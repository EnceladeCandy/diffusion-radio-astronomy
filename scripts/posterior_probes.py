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
sys.path.append("../..")
import h5py

from utils import probes_64, link_function

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


device = "cuda" if torch.cuda.is_available() else "cpu"

# Importing the models hparams, weights + loading them
file = open("/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652/model_hparams.json")
model_hparams = json.load(file)
sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]
score_model = ScoreModel(checkpoints_directory="/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652")

# Path ground-truth prior
dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
hdf = h5py.File(dir_galaxies, "r")
hdf.keys()
dataset = hdf['galaxies']


def main(args):
    hparams_sampler = [(20, 1e-2), (500, 1e-3), (2000, 1e-4)] # Format = (num_corr_step, snr)

    # Args of the scripts
    if N_WORKERS>1:
        idx = THIS_WORKER
        num_samples = args.num_samples
        num_iters = args.num_iters
        num_corr_steps, snr = hparams_sampler[int(idx/100)] # Every 100 workers we change parameters (100 simulations per hparams)
        marker = f"/corr{num_corr_steps}_snr{snr:.1g}"
        folder_dir = "../../" + args.samples_folder + marker
        if idx%100==0:    
            os.mkdir(folder_dir)
    else: 
        idx = args.idx
        num_samples = args.num_samples
        num_iters = args.num_iters
        num_corr_steps = args.num_corr_steps
        snr = args.snr

    def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")
    
    # Loading an image in the dataset:
    img = probes_64(dataset, idx)
    #img = link_function(torch.load("../../prior.pt")[idx]) # PRIOR SAMPLES
    img_size = img.shape[-1]

    # Loading the psf
    psf = torch.load("../psf64.pt")

    # Calculating the sampled visibilities given the full visibility and the psf
    vis_full = ft(img).flatten()
    sampling_function= ft(torch.fft.ifftshift(psf)).flatten()
    vis_sampled = sampling_function * vis_full
    vis_sampled = vis_sampled.flatten()
    vis_sampled = torch.cat([vis_sampled.real, vis_sampled.imag])

    
    samples_per_loop = 20 # Maximum capacity for beluga given their 16GB of RAM for the GPUs
    samples = torch.empty(size = (num_samples, img_size**2), requires_grad = False).to(device)
    
    for i in tqdm(range(int(num_samples//samples_per_loop))):
        sigma_likelihood = args.sigma_likelihood
        
        y_dim = len(vis_sampled)  
        dist_likelihood = MultivariateNormal(loc = torch.zeros(y_dim).to(device), covariance_matrix=sigma_likelihood **2 * torch.eye(y_dim).to(device))
        eta = dist_likelihood.sample([])

        # Creating an observation:
        y = vis_sampled + eta 

        # VE SDE
        def sigma(t): 
            return sigma_min * (sigma_max/sigma_min) ** t

        def logprob_likelihood(x, cov_mat): 
            dist = torch.distributions.MultivariateNormal(loc = torch.zeros(y_dim, device = x.device), covariance_matrix = cov_mat, validate_args=False)
            return dist.log_prob(x)

        def model(x):
            x = link_function(x)
            vis_full = ft(x.reshape(img_size, img_size)).flatten()
            # vis_sampled = sampling_function * vis_full
            # To prevent problems with grad for complex functions in torch
            real_part = sampling_function.real * vis_full.real - sampling_function.imag * vis_full.imag
            im_part = sampling_function.real * vis_full.imag + sampling_function.imag * vis_full.real
            vis_sampled = torch.cat([real_part, im_part])
            return vis_sampled

        def score_likelihood(x, t): 
            return vmap(grad(lambda x, t: logprob_likelihood(y- model(x),  (sigma_likelihood ** 2 + sigma(t)**2) * torch.eye(y_dim, device = x.device))))(x, t)

        def score_posterior(x, t): 
            return score_model.score(t, x.reshape(-1, 1, img_size, img_size)).flatten(start_dim = 1) + score_likelihood(x, t)

        def g(t): 
            return sigma(t) * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))

        def pc_sampler(num_samples, num_pred_steps, num_corr_steps, score_function, snr = 1e-2, img_size = 28): 
            t = torch.ones(size = (num_samples, 1)).to(device)
            x = torch.randn([num_samples, img_size ** 2]).to(device)
            dt = -1/num_pred_steps
            with torch.no_grad(): 
                for _ in tqdm(range(num_pred_steps-1)): 
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
                for _ in range(num_steps - 1): 
                    z = torch.randn_like(x).to(device)
                    gradient = score_function(x, t)
                    drift = 0
                    diffusion = g(t)
                    x_mean = x - diffusion**2 * gradient * dt  
                    noise = diffusion * (-dt) ** 0.5 * z
                    x = x_mean + noise
                    t += dt
            
            return link_function(x_mean)

        if args.sampler == "euler": 
            samples_loop = euler_sampler(num_samples = samples_per_loop, num_steps = num_iters, score_function = score_posterior, img_size = img_size)
        elif args.sampler == "pc":
            samples_loop = pc_sampler(num_samples = samples_per_loop, num_pred_steps = num_iters, num_corr_steps = args.num_corr_steps, snr = args.snr, score_function = score_posterior, img_size = img_size)
            path = folder_dir
        samples[i*samples_per_loop:(i+1)*samples_per_loop] = samples_loop
    
    #path = f"../../"+ args.samples_folder + f"/sigma_{sigma_likelihood:.1g}/"
    torch.save(samples, path + f"/idx_{idx}.pt")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Likelihood parameters
    parser.add_argument("--sigma_likelihood",   required = True,                    type = float,   help = "The square root of the multiplier of the isotropic gaussian matrix")
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "pc", type = str,      help = "Sampling procedure used ('pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,   type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--num_iters",          required = False,   default = 1000, type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--num_corr_steps",     required = False,   default = 20,   type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2, type = float)
    
    # Ground-truth parameters
    parser.add_argument("--idx",                required = False,   default = 0,    type = int,     help = "Idx of the image in the probes dataset (between 0 and approx 2000)")

    # Output files parameter
    parser.add_argument("--samples_folder",     required = False,   default ="samples_probes",      help = "Folder where to save the samples")
    
    args = parser.parse_args()
    main(args) 
