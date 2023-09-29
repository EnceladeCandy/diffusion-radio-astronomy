import os
import sys
sys.path.append("..\\scripts\\")
import torch
from torch.func import vmap, grad 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mpol.constants as const
from typing import Callable
import h5py


from score_models import ScoreModel
import json

N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args): 
    
    
    #Function definitions
    def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")

    # C = 0.5 # VE prior
    # B = 0.5 # VE
    C = 1 # VP prior
    B = 0 # VP
    def link_function(x):
        return C * x + B
    

    def grid(pixel_scale, img_size): 
        """Given a pixel scale and a number of pixels in image space, grid the associated Fourier space

        Args:
            pixel_scale (float): Pixel resolution (in arcsec)
            img_size (float/int): Size of the image 

        Returns:
            
        """

        # Arcsec to radians: 
        dl = pixel_scale * const.arcsec
        dm = pixel_scale * const.arcsec

        du = 1 / (img_size * dl) * 1e-3 # klambda
        dv = 1 / (img_size * dm) * 1e-3 # klambda

        u_min = -img_size//2 * du 
        u_max = img_size//2 * du 

        v_min = -img_size//2 * dv
        v_max = img_size//2 * dv

        u_edges = np.linspace(u_min, u_max, img_size + 1)
        v_edges = np.linspace(v_min, v_max, img_size + 1)

        return u_edges, v_edges
    
    #CONSTANTS
    # TODO : Put all the functions in a helper file so that the code is more readable.
    pixel_scale = 0.0015
    pad = (args.padding - args.model_pixels)//2 
    img_size = args.model_pixels
    npix = img_size + 2 * pad
    u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = npix)
    pixel_size = u_edges[1] - u_edges[0] # this is delta_u, and we should probably call it that in the future. My bad
    
    # OLD DATA IMPORT
    # vis_bin_re = np.load(os.path.join(args.data_dir, "allspw_re.npy"))
    # std_bin_re = np.load(os.path.join(args.data_dir,  "allspw_re_std.npy"))
    # counts = np.load(os.path.join(args.data_dir, "allspw_counts.npy"))
    # vis_bin_imag = np.load(os.path.join(args.data_dir, "allspw_imag.npy"))
    # std_bin_imag = np.load(os.path.join(args.data_dir, "allspw_imag_std.npy"))

    # NEW DATA IMPORT: 
    with np.load(args.data_dir) as data_gridded: 

        vis_bin_re = data_gridded["vis_bin_re"]
        vis_bin_imag = data_gridded["vis_bin_imag"]
        std_bin_re = data_gridded["std_bin_re"]
        std_bin_imag = data_gridded["std_bin_imag"]
        counts = data_gridded["counts"]

    # From object type to float
    vis_bin_re = vis_bin_re.astype(float)
    vis_bin_imag = vis_bin_imag.astype(float)
    std_bin_re = std_bin_re.astype(float) 
    std_bin_imag = std_bin_imag.astype(float)
    counts = counts.astype(float)

    # i.e. The sampling function where there is data in the uv plane
    mask = counts > 0

    # Upsample std 
    from scipy import interpolate
    def upsample(input_data, d):
        # Get the shape of the input grid
        rows, cols = input_data.shape
        # Define the x and y coordinates of the original grid
        x = np.linspace(0, cols - 1, cols)
        y = np.linspace(0, rows - 1, rows)

        # Create a bilinear interpolator object
        f_interp = interpolate.interp2d(x, y, input_data, kind='linear')

        # Define the x and y coordinates of the upsampled grid
        x_new = np.linspace(0, cols - 1, cols * d)
        y_new = np.linspace(0, rows - 1, rows * d)

        # Perform the interpolation to upsample the grid
        upsampled_data = f_interp(x_new, y_new)
        return upsampled_data

    # upsample std if needed
    if std_bin_re.shape[0] < vis_bin_re.shape[0]:
        assert vis_bin_re.shape[0] // std_bin_re.shape[0] == downsampling_factor
        std_bin_re = upsample(std_bin_re, downsampling_factor)
    if std_bin_imag.shape[0] < vis_bin_imag.shape[0]:
        assert vis_bin_imag.shape[0] // std_bin_imag.shape[0] == downsampling_factor
        std_bin_imag = upsample(std_bin_imag, downsampling_factor)

    #I am commenting this out and performing this process inside the binning function
    #std_bin_re /= (counts + 1)**0.5
    #std_bin_imag /= (counts + 1)**0.5

    # For the inference, fftshift is done in the forward model
    vis_gridded_re = np.fft.fftshift(vis_bin_re).flatten()
    vis_gridded_imag = np.fft.fftshift(vis_bin_imag).flatten()
    std_gridded_re = np.fft.fftshift(std_bin_re).flatten()
    std_gridded_imag = np.fft.fftshift(std_bin_imag).flatten()
    S_grid = np.fft.fftshift(mask).flatten()
    
    S_cat = np.concatenate([S_grid, S_grid])
    vis_gridded = np.concatenate([vis_gridded_re, vis_gridded_imag])
    std_gridded = np.concatenate([std_gridded_re, std_gridded_imag])

    # Numpy to torch: 
    S = torch.tensor(S_cat).to(device)
    y = torch.tensor(vis_gridded, device = S.device)[S].to(device) * npix
    sigma_y = torch.tensor(std_gridded, device = S.device)[S].to(device) * npix

    # Importing and loading the weights of the score of the prior 
    score_model = ScoreModel(checkpoints_directory=args.prior)
    
    def sigma(t):
        return score_model.sde.sigma(t)

    def mu(t):
        return score_model.sde.marginal_prob_scalars(t)[0]

    def sigma(t): 
        return score_model.sde.sigma(t)

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
        var = sigma(t) **2 / 2 * C**2 + mu(t)**2 * sigma_y**2
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
            for i in tqdm(range(num_pred_steps-1)): 
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

                # To check the time for sampling:
                # if i == 20:
                #     break
        return link_function(x_mean).reshape(-1, 1, img_size, img_size)

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

               #To check the time for sampling:
                # if i == 20:
                #     break

        return link_function(x_mean).reshape(-1, 1, img_size, img_size)

    sampler = args.sampler
    pred = args.num_pred
    corr = args.num_corr
    snr = args.snr
    num_samples = args.num_samples 
    batch_size = args.batch_size
    
    path = args.results_dir + f"/{sampler}/"
    # if not os.path.exists(path):
    #         # os.makedirs(path)

    filename = os.path.join(path, args.experiment_name + f"_{THIS_WORKER}" + ".h5")
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("model", [args.num_samples, 1, args.model_pixels, args.model_pixels], dtype=np.float32)
        hf.create_dataset("reconstruction", [args.num_samples, len(y)], dtype=np.float32)
        hf["observation"] = y.cpu().numpy().astype(np.float32).squeeze()

        for i in range(int(num_samples//batch_size)):
            if sampler.lower() == "euler":    
                samples = euler_sampler(
                    y = y,
                    sigma_y = sigma_y,
                    num_samples = batch_size,
                    num_steps = pred, 
                    score_function = score_posterior, 
                    img_size = img_size
                )
                

            elif sampler.lower() == "pc":
                #pc_params = [(1001, 10, 1e-2), (1000, 100, 1e-2), (1000, 1000, 1e-3)]
                #pc_params = [(1001, 1000, 1e-3)]
                # idx = int(THIS_WORKER//101)
                # pred, corr, snr = pc_params[idx]

                print(f"Sampling pc pred = {pred}, corr = {corr}, snr = {snr}")
                samples = pc_sampler(
                    y = y,
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

    import matplotlib.pyplot as plt
    plt.imshow(samples[0].squeeze().cpu().numpy().astype(np.float32), cmap="magma")
    plt.savefig("sanity2.jpeg", bbox_inches="tight")

if __name__ == "__main__": 
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--prior",              required = True,                    type = str,     help = "Path to the checkpoints directory of the prior")
    parser.add_argument("--data_dir",           required = True,                                    help = "Path where to find all the data files")
    parser.add_argument("--padding",            required = False,   default = 4096, type = int,     help = "Size of the padded image.")
    parser.add_argument("--model_pixels",       required = False,   default = 256,  type = int,     help = "Size of the pixel grid for the prior")
    parser.add_argument("--sampler",            required = False,   default = "pc", type = str,     help = "Sampling procedure used ('pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,   type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--num_pred",           required = False,   default = 1000, type = int,     help = "Number of predictor/euler steps in the loop")
    parser.add_argument("--num_corr",           required = False,   default = 20,   type = int,     help = "Number of corrector steps for the PC sampling")
    parser.add_argument("--snr",                required = False,   default = 1e-2, type = float,   help = "Signal-to-noise ratio for the PC sampling")
    parser.add_argument("--batch_size",         required = True,    default = 25,   type = int,     help = "Number of samples created per iteration")
    parser.add_argument("--results_dir",        required = True,                                    help = "Directory where to save the TARP files")
    parser.add_argument("--experiment_name",    required = True,                                    help = "Prefix for the name of the file")

    args = parser.parse_args()
    main(args) 
