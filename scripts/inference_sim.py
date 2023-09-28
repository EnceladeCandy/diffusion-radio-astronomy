import torch 
import numpy as np
import os
from torch.func import vmap, grad
from tqdm import tqdm
import h5py

from score_models import ScoreModel
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

def main(args): 
    def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")

    # Importing and loading the weights of the score of the prior 
    prior = args.prior
    score_model = ScoreModel(checkpoints_directory=prior)
    S = torch.tensor(np.load(args.sampling_function).astype(bool)).to(device)
    img_size = args.model_pixels

    if "probes" in prior:
        print("WARNING: RUNNING WITH OLD FUNCTION PROBES (OK IF 64*64)") 
        C = 1/2
        B = 1/2

    elif "skirt" in prior: 
        C = 1 # VP prior
        B = 0 # VP
    

    def link_function(x):
        return C * x + B
    
    def sigma(t):
        return score_model.sde.sigma(t)

    def mu(t):
        return score_model.sde.marginal_prob_scalars(t)[0]
    
    pad = args.pad
    def noise_padding(x, pad, sigma):
        H, W = x.shape
        out = torch.nn.functional.pad(x, (pad, pad, pad, pad)) 
        # Create a mask for padding region
        mask = torch.ones_like(out)
        mask[pad:pad + H, pad:pad+W] = 0.
        # Noise pad around the model
        z = torch.randn_like(out) * sigma
        out = out + z * mask
        return out
    
    def model(x, t):
        x = x.reshape(img_size, img_size) # for the FFT 
        x = noise_padding(x, pad=pad, sigma=sigma(t))
        x = link_function(x) # map from model unit space to real unit space
        vis_full = ft(torch.fft.fftshift(x)).flatten() 
        vis_sampled = vis_full
        vis_sampled = torch.cat([vis_sampled.real, vis_sampled.imag])
        return vis_sampled[S]
    

    def log_likelihood(y, x, t, sigma_y):
        """
        Calculate the log-likelihood of a gaussian distribution 
        Arguments: 
            y = processed gridded visibilities (real part and imaginary part concatenated)
            x = sky brightness 
            t = diffusion temperature
            A = linear model (sampling function and FT)  
        
        Returns: 
            log-likelihood of a gaussian distribution
        """ 
        y_hat = model(x, t)
        var = sigma(t) **2/2 + mu(t)**2 * sigma_y**2
        log_prob = -0.5 * torch.sum((mu(t) * y - y_hat)**2 / var)
        return log_prob

    def score_likelihood(y, x, t, sigma_y): 
        x = x.flatten(start_dim = 1)
        return vmap(grad(lambda x, t: log_likelihood(y, x, t, sigma_y)), randomness = "different")(x, t)

    def score_posterior(y, x, t, sigma_y): 
        x = x.reshape(-1, 1, img_size, img_size)
        return score_model.score(t, x).flatten(start_dim = 1) + score_likelihood(y, x, t, sigma_y) 

    def g(t, x):
        return score_model.sde.diffusion(t, x)

    def drift_fn(t, x):
        return score_model.sde.drift(t, x)

    def pc_sampler(y, sigma_y, num_samples, num_pred_steps, num_corr_steps, score_function, snr = 1e-2, img_size = 28): 
        t = torch.ones(size = (num_samples, 1)).to(device)
        x = sigma(t) * torch.randn([num_samples, img_size ** 2]).to(device)
        dt = -1/num_pred_steps


        with torch.no_grad(): 
            for _ in tqdm(range(num_pred_steps-1)): 
                # Corrector step: (Only if we are not at 0 temperature )
                gradient = score_function(y, x, t, sigma_y)
                for _ in range(num_corr_steps): 
                    z = torch.randn_like(x)
                    grad_norm = torch.mean(torch.norm(gradient, dim = -1)) # mean of the norm of the score over the batch 
                    noise_norm = torch.mean(torch.norm(z, dim = -1))
                    epsilon =  2 * (snr * noise_norm / grad_norm) ** 2
                    x = x + epsilon * gradient + (2 * epsilon) ** 0.5 * z * dt  

            
                # Predictor step: 
                z = torch.randn_like(x).to(device)
                gradient = score_function(y, x, t, sigma_y)
                drift = drift_fn(t, x)
                diffusion = g(t, x)
                x_mean = x + drift * dt - diffusion**2 * gradient * dt  
                noise = diffusion * (-dt) ** 0.5 * z
                x = x_mean + noise
                t += dt
                
        return link_function(x_mean).reshape(-1, 1, img_size, img_size), chain

    def euler_sampler(y, sigma_y, num_samples, num_steps, score_function, img_size = 28): 
        t = torch.ones(size = (num_samples, 1)).to(device)
        x = sigma(t) * torch.randn([num_samples, img_size ** 2]).to(device)
        dt = -1/num_steps

        with torch.no_grad(): 
            for i in tqdm(range(num_steps - 1)): 
                z = torch.randn_like(x).to(device)
                gradient = score_function(y, x, t, sigma_y)
                drift = drift_fn(t, x)
                diffusion = g(t, x)
                x_mean = x + drift * dt - diffusion**2 * gradient * dt  
                noise = diffusion * (-dt) ** 0.5 * z
                x = x_mean + noise
                t += dt

        return link_function(x_mean).reshape(-1, 1, img_size, img_size)

    sampler = args.sampler
    pred = args.num_pred
    corr = args.num_corr
    snr = args.snr
    
    batch_size = args.batch_size
    num_samples = args.num_samples
    
    path = args.results_dir + f"{sampler}/"

    filename = os.path.join(path, args.experiment_name + f"_{THIS_WORKER}" + ".h5")
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("model", [args.num_samples, 1, args.model_pixels, args.model_pixels], dtype=np.float32)

        ground_truth = score_model.sample([1, 1, args.model_pixels, args.model_pixels], steps=pred)
        observation = model(x = ground_truth.flatten(), t = torch.zeros(1).to(device))
        sigma_y = args.sigma_likelihood
        observation += torch.randn_like(observation) * sigma_y

        hf.create_dataset("reconstruction", [args.num_samples, observation.shape[0]], dtype=np.float32)
        hf["observation"] = observation.cpu().numpy().astype(np.float32).squeeze()
        hf["ground_truth"] = link_function(ground_truth).cpu().numpy().astype(np.float32).squeeze()
        
        plt.imshow(ground_truth.squeeze().cpu(), cmap = "magma")
        plt.savefig("new.jpeg")
        plt.show()
        for i in range(int(num_samples//batch_size)):
            if sampler.lower() == "euler":    
                samples = euler_sampler(
                    y = observation,
                    sigma_y = sigma_y,
                    num_samples = batch_size,
                    num_steps = pred, 
                    score_function = score_posterior, 
                    img_size = img_size
                )

            elif sampler.lower() == "pc":
                # pc_params = [(1000, 10, 1e-2), (1000, 100, 1e-2), (1000, 1000, 1e-3)]
                # #pc_params = [(1000, 1000, 1e-3)]
                # idx = int(THIS_WORKER//100)
                # pred, corr, snr = pc_params[idx]

                print(f"Sampling pc pred = {pred}, corr = {corr}, snr = {snr}")
                samples = pc_sampler(
                    y = observation,
                    sigma_y = sigma_y,
                    num_samples = batch_size,
                    num_pred_steps = pred,
                    num_corr_steps = corr,
                    snr = snr,
                    score_function = score_posterior,
                    img_size = img_size
                )
                
                
            else : 
                raise ValueError("The sampler specified is not implemented or does not exist. Choose between 'euler' and 'pc'")
            B = batch_size
            hf["model"][i*B: (i+1)*B] = samples.cpu().numpy().astype(np.float32)

            # Let's hope it doesn't take too much time compared to the posterior sampling:
            y_hat = torch.empty(size = (B, 1, img_size, img_size)).to(device)
            for j in range(batch_size):
                y_hat = model(samples[j], torch.zeros(1).to(device))
            hf["reconstruction"][i*B: (i+1)*B] = y_hat.cpu().numpy().astype(np.float32)
  
    # plt.imshow(ground_truth.squeeze().cpu(), cmap = "magma")
    # plt.savefig("new.jpeg", cmap = "magma")
    # fig, axs = plt.subplots(1, 2, figsize = (8, 4))
    # x = link_function(ground_truth).cpu().numpy().astype(np.float32).squeeze()
    # im = axs[0].imshow(x, cmap = "magma")
    # plt.colorbar(im, fraction = 0.046,ax = axs[0])

    # im = axs[1].imshow(samples[0].squeeze().cpu().numpy().astype(np.float32), cmap="magma")
    # plt.colorbar(im, fraction = 0.046, ax = axs[1])
    # plt.subplots_adjust(wspace=0.5)
    # plt.savefig("what.jpeg", bbox_inches="tight")
    # print(x.sum())
    # print(samples.sum())


if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Likelihood parameters
    parser.add_argument("--sigma_likelihood",   required = True,                    type = float,   help = "The square root of the multiplier of the isotropic gaussian matrix")
    
    # Experiments spec
    parser.add_argument("--results_dir",        required = True,                                    help = "Directory where to save the TARP files")
    parser.add_argument("--experiment_name",    required = True,                                    help = "Prefix for the name of the file")
    

    parser.add_argument("--model_pixels",       required = True,                    type = int)
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "pc", type = str,     help = "Sampling procedure used ('pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,   type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--batch_size",         required = False,   default = 20,   type = int)
    parser.add_argument("--num_pred",           required = False,   default = 1000, type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--num_corr",           required = False,   default = 20,   type = int,     help ="Number of corrector steps for the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2, type = float)
    parser.add_argument("--pad",                required = False,   default = 0,    type = int)
    parser.add_argument("--sampling_function",  required = True)
    parser.add_argument("--prior",              required = True)
    
    args = parser.parse_args()
    main(args) 




