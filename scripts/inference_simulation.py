import torch 
from torch.func import vmap, grad
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import h5py
from score_models import ScoreModel, NCSNpp
import json
import os
import sys
import matplotlib.pyplot as plt
sys.path.append("../..")

from utils import link_function, probes_256, fits_to_tensor, resize, probes_64

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


device = "cuda" if torch.cuda.is_available() else "cpu"

""" # 256 probes
# Importing the models hparams, weights + loading them
file = open("/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942/model_hparams.json")
model_hparams = json.load(file)
sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]
score_model = ScoreModel(checkpoints_directory="/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942")
"""



# Path ground-truth prior
dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
hdf = h5py.File(dir_galaxies, "r")
hdf.keys()
dataset = hdf['galaxies']


def main(args):
    

    if N_WORKERS!=1: 
        idx = THIS_WORKER
    else: 
        idx = np.random.randint(len(dataset))

    img_size = args.sim_size
    if img_size == 64:
        # 64 probes
        file = open("/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652/model_hparams.json")
        model_hparams = json.load(file)
        sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]
        score_model = ScoreModel(checkpoints_directory="/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652") 
        
        # Loading the psf
        #path_ms + "_psf.fits").
        ms = args.ms
        header_psf, psf = fits_to_tensor("../../data_targets2/" + ms + ".fits")
    
        psf = resize(psf, target_size = 64).to(device)
        img = probes_64(dataset, idx).to(device)
    
    elif img_size == 256: 
        # Importing the models hparams, weights + loading them
        file = open("/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942/model_hparams.json")
        model_hparams = json.load(file)
        sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]
        score_model = ScoreModel(checkpoints_directory="/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942")
        
        # Loading the psf
        ms = args.ms
        header_psf, psf = fits_to_tensor("../../data_targets2/" + ms + "_psf.fits")
        psf = psf.to(device)
        img = torch.tensor(dataset[idx])[..., 1].to(device) # green 

    else: 
        raise ValueError("The image size for probes must be either 64 or 256 (only simulation size for the trained scored models)")

    def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")

    def ift(x):
        return torch.fft.ifft2(x, norm = "ortho")

    vis_full = ft(img).flatten()
    sampling_function= ft(torch.fft.ifftshift(psf)).flatten()

    # To prevent problems with grad for complex functions in torch
    real_part = sampling_function.real * vis_full.real - sampling_function.imag * vis_full.imag
    im_part = sampling_function.real * vis_full.imag + sampling_function.imag * vis_full.real
    vis_sampled = torch.cat([real_part, im_part])

    sigma_likelihood = args.sigma_likelihood
    
    sigma_likelihood = args.sigma_likelihood
    y = vis_sampled + sigma_likelihood * torch.randn_like(vis_sampled)
    

    
    def model(x):
        x = link_function(x)
        vis_full = ft(x.reshape(img_size, img_size)).flatten()
        sampling_function= ft(torch.fft.ifftshift(psf)).flatten()

        # To prevent problems with grad for complex functions in torch
        real_part = sampling_function.real * vis_full.real - sampling_function.imag * vis_full.imag
        im_part = sampling_function.real * vis_full.imag + sampling_function.imag * vis_full.real
        vis_sampled = torch.cat([real_part, im_part])
        return vis_sampled

    # VE SDE
    def sigma(t): 
        return sigma_min * (sigma_max/sigma_min) ** t
    
    def logprob_likelihood(x, sigma): 
      D = x.shape[-1]
      val = -torch.sum(x**2, axis = -1)/(2*sigma**2) - D/2 * np.log((2*torch.pi))- D * torch.log(sigma)
      return val.squeeze(0) # needs to be without diemensions   

    def score_likelihood(x, t): 
        return vmap(grad(lambda x, t: logprob_likelihood(y -model(x), (sigma_likelihood ** 2 + sigma(t)** 2)**0.5)))(x, t)

    def score_posterior(x, t): 
        return score_model.score(t, x.reshape(-1, 1, img_size, img_size)).flatten(start_dim = 1) + score_likelihood(x, t)

    def g(t): 
        return sigma(t) * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))

    def pc_sampler(num_samples, num_pred_steps, num_corr_steps, score_function, snr = 1e-2, img_size = 28): 
        t = torch.ones(size = (num_samples, 1)).to(device)
        x = torch.randn([num_samples, img_size ** 2]).to(device)
        dt = -1/num_pred_steps
        with torch.no_grad(): 
            for i in tqdm(range(num_pred_steps-1)): 
                # Corrector step: (Only if we are not at 0 temperature )
                gradient = score_function(x, t)
                for _ in range(num_corr_steps): 
                    z = torch.randn_like(x)
                    grad_norm = torch.mean(torch.norm(gradient, dim = -1)) # mean of the norm of the score over the batch 
                    noise_norm = torch.mean(torch.norm(z, dim = -1))
                    epsilon =  2 * (snr * noise_norm / grad_norm) ** 2
                    x = x + epsilon * gradient + (2 * epsilon) ** 0.5 * z * dt  

            
                # Predictor step: 
                z = torch.randn_like(x).to(device)
                gradient = score_function(x, t)
                drift = 0
                diffusion = g(t)
                x_mean = x - diffusion**2 * gradient * dt  
                noise = diffusion * (-dt) ** 0.5 * z
                x = x_mean + noise
                t += dt
        return link_function(x_mean)

    def euler_sampler(num_samples, num_steps, score_function, img_size = 28): 
        t = torch.ones(size = (num_samples, 1)).to(device)
        x = sigma_max * torch.randn([num_samples, img_size ** 2]).to(device)
        dt = -1/num_steps
        with torch.no_grad(): 
            for i in tqdm(range(num_steps - 1)): 
                z = torch.randn_like(x).to(device)
                gradient = score_function(x, t)
                drift = 0
                diffusion = g(t)
                x_mean = x - diffusion**2 * gradient * dt  
                noise = diffusion * (-dt) ** 0.5 * z
                x = x_mean + noise
                t += dt
        
        return link_function(x_mean)

    sampler = args.sampler
    pred = args.num_pred
    corr = args.num_corr
    snr = args.snr
    num_samples = args.num_samples


    batch_size = args.batchsize # Maximum number of posterior samples per gpu with 16GB (beluga)
    samples_tot = torch.empty(size = (num_samples, img_size, img_size))


    if sampler.lower() == "euler":    
        for i in tqdm(range(int(num_samples//batch_size))):
            samples = euler_sampler(
                num_samples = batch_size,
                num_steps = pred, 
                score_function = score_posterior, 
                img_size = img_size
            )

            # Filling gradually the samples
            samples_tot[i * batch_size : (i + 1) * batch_size] = samples.reshape(-1, img_size, img_size)

        # Saving 
        folder_dir = f"../../samples_probes/euler"
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        torch.save(samples_tot, folder_dir + f"/idx_{idx}.pt")

    elif sampler.lower() == "pc":
        for i in tqdm(range(int(num_samples//batch_size))):
            samples = pc_sampler(
                num_samples = batch_size,
                num_pred_steps = pred,
                num_corr_steps = corr,
                snr = snr,
                score_function = score_posterior,
                img_size = img_size
            )
            # Filling gradually the samples
            samples_tot[i * batch_size : (i + 1) * batch_size] = samples.reshape(-1, img_size, img_size)

        # Saving 
        folder_dir = f"../../samples_probes/pc/corr_{corr}_snr{snr}"
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        torch.save(samples_tot, folder_dir + f"/idx_{idx}.pt")
        
    
    else: 
        raise ValueError("The sampler does not exist ! Use either 'pc' or 'euler'")
    
    vis_gridded = y[:int(img_size**2)] + 1j * y[int(img_size**2):]
    dirty_image = ift(vis_gridded.reshape(img_size, img_size))
    fig, axs = plt.subplots(1, 10, figsize = (10*3.5, 3.5))
    for i in range(len(axs)):
        axs[i].axis("off")
    axs[0].imshow(img.cpu(), cmap = "magma")
    axs[1].imshow(dirty_image.real.cpu(), cmap = "magma")
    for i in range(8):
        axs[i+2].imshow(samples_tot[i].cpu(), cmap = "magma")
    plt.savefig("../../images/sanity.jpeg", bbox_inches="tight", pad_inches = 0.2)
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Likelihood parameters
    parser.add_argument("--sigma_likelihood",   required = True,                    type = float,   help = "The square root of the multiplier of the isotropic gaussian matrix")
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "pc", type = str,      help = "Sampling procedure used ('pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,   type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--num_pred",          required = False,   default = 1000, type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--num_corr",     required = False,   default = 20,   type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2, type = float)
    
    parser.add_argument("--sim_size",           required = True,   type = int)
    # Ground-truth parameters
    parser.add_argument("--ms",                required = False,   default = "HTLup_COcube" ,    type = str,     help = "Name of the target") 
    
    parser.add_argument("--batchsize",          required = False,  type = int,  default = 1)
    args = parser.parse_args()
    main(args) 
