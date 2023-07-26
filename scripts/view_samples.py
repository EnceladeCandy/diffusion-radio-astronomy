import torch 
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import numpy as np
import h5py
plt.style.use("dark_background")
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
This script creates plots to view the posterior samples in the samples_probes folder or the samples folder.

Args:
sigma_likelihood = sqrt(variance) of the gaussian noise added in the forward model
idx = indice of the image in the probes dataset
num_samples = number of posterior samples to create for the second plot
samples_folder = samples_probes or samples (samples_probes = posterior samples using the real prior, samples = posterior samples using a sample from the learnt prior)

"""

def main(args):
    def preprocess_probes_g_channel(img, inv_link = False):  # channel 0
        img = torch.clamp(img, 0, 1.48)
        
        if inv_link:
            img = 2 * img / 1.48 - 1.
        return img
    
    # Image from the probes dataset: 
    dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
    hdf = h5py.File(dir_galaxies, "r")
    hdf.keys()
    dataset = hdf['galaxies']

    # Idx of the image in the dataset and 
    idx = args.idx
    sigma_likelihood = args.sigma_likelihood

    # Importing the posterior samples
    path = f"../../"+ args.samples_folder + f"/sigma_{sigma_likelihood:.1g}/"
    samples = torch.load(path + f"idx_{idx}.pt", map_location=device)
    num_samples = samples.shape[0]
    img_size = int((samples.shape[1])**0.5) # Assuming images with height = width
    samples = samples.reshape(-1, img_size, img_size)


    ############ FIRST PLOT ###################
    # Plot naive reconstruction: 
    def ft(x): 
            return torch.fft.fft2(x, norm = "ortho")

    def ift(x): 
        return torch.fft.ifft2(x, norm = "ortho")


    psf = torch.load("../psf64.pt", map_location = device)
    img = torch.tensor(dataset[0, ..., 1]) # green channel
    img = preprocess_probes_g_channel(img) # probes (N, 256, 256, 3)
    img = F.avg_pool2d(img[None, None, ...], (4, 4))[0, 0].to(device)
    #img = torch.load("../ground_truth.pt")
    #img = torch.load("../../prior.pt", map_location = device)[idx]
    img_size = img.shape[-1]

    # Calculating the sampled visibilities: V_s = FT(psf) * FT(img)
    vis_full = ft(img).flatten()
    sampling_function= ft(torch.fft.ifftshift(psf)).flatten()
    vis_sampled = sampling_function * vis_full

    vis_sampled = vis_sampled.flatten()
    vis_sampled = torch.cat([vis_sampled.real, vis_sampled.imag]) # trik for torch

    y_dim = len(vis_sampled)  
    dist_likelihood = MultivariateNormal(loc = torch.zeros(y_dim).to(device), covariance_matrix=sigma_likelihood **2 * torch.eye(y_dim).to(device))
    eta = dist_likelihood.sample([])
    
    # Creaing an observation
    y = vis_sampled + eta 

    print("Plotting naive reconstruction...\n")
    fig, axs = plt.subplots(1, 2, figsize = (8, 4))
    dirty_image_noise = ift((y[:img_size**2] + 1j * y[img_size**2:]).reshape(img_size, img_size)).real

    for i in range(len(axs)): 
        axs[i].axis("off")

    axs[0].imshow(img.reshape(img_size, img_size).cpu(), cmap = "hot")
    axs[0].set_title("Ground-truth")
    axs[1].imshow(dirty_image_noise.cpu(), cmap = "hot")
    axs[1].set_title("Dirty image with noise")
    plt.subplots_adjust(wspace = 0.1)
    fig.suptitle(r"$\sigma_{lh}$ = " + f"{sigma_likelihood:.1g}", y = 0.1)
    plt.savefig(f"../../images/naive_rec/{sigma_likelihood:.1g}_{idx}.jpeg", bbox_inches = "tight", pad_inches = 0.1)


    print("Plotting posterior samples...\n")
    ############ SECOND PLOT ################### 
    # Creating the plot
    grid_size = int((args.num_samples) ** 0.5)
    if grid_size==1:
        fig = plt.figure(figsize= (8,8), dpi = 150)
        plt.imshow(samples[0].cpu(), cmap = "hot")
        plt.axis("off")
        plt.savefig(f"../../images/samples/posterior_{sigma_likelihood:.1g}_{idx}.jpg", bbox_inches = "tight", pad_inches = 0.1)

    else:
        fig, axs = plt.subplots(grid_size, grid_size, figsize = (10, 10), dpi = 150)

        k = 0
        for i in range(grid_size): 
            for j in range(grid_size): 
                axs[i, j].imshow(samples[k].cpu(), cmap = "hot")
                axs[i, j].axis("off")
                k += 1
        fig.suptitle(r"$\sigma_{lh}$ = " + f"{sigma_likelihood:.1g}", y = 0.1)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
        plt.savefig(f"../../images/samples/posterior_{sigma_likelihood:.1g}_{idx}.jpg", bbox_inches = "tight", pad_inches = 0.1)

if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sigma_likelihood",        required = True,    type = float)
    parser.add_argument("--idx",                required = True,    type = int)
    parser.add_argument("--num_samples",    required = True,    type=int, help = "Number of samples to view in the plot (ideally a perfect square)")
    parser.add_argument("--samples_folder",        required = False,    default ="samples_probes")
    args = parser.parse_args()
    main(args) 

