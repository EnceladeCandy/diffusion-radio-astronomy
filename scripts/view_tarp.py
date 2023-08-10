import matplotlib.pyplot as plt
import numpy as np 
import os
import torch
import h5py
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
from tarp_perso import bootstrapping, get_drp_coverage



def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Importing the dataset
    dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
    hdf = h5py.File(dir_galaxies, "r")
    hdf.keys()
    dataset = hdf['galaxies']
    img_size = 64

    sampler = args.sampler    
    
    path = f"../../samples_probes/{sampler}"
    samples_files = os.listdir(path)

    batch_size = 250
    num_samples = 1000
    num_sims = len(samples_files)
    num_dims = img_size ** 2
    samples = torch.empty(size = (num_samples, num_sims, num_dims)) # (n_samples, n_sims, n_dims)
    theta = torch.empty(size = (num_sims, num_dims)) # (n_sims, n_dims)

    print("Importing the samples and the ground-truths...")
    for i in tqdm(range(num_sims)):
        samples[:, i, :] = torch.load(path + "/" + samples_files[i], map_location=torch.device(device)).flatten(start_dim = 1)
        k = int(samples_files[i].split("_")[-1].replace(".pt", ""))
        theta[i, :] = resize(torch.tensor(dataset[k, ..., 1]), target_size=64).flatten()

    # tarp is coded for numpy
    samples = samples.numpy()
    theta = theta.numpy()
    
    # Sanity check: 

    fig, axs = plt.subplots(1, 5, figsize = (5*3.5, 3.5))
    for i in range(len(axs)):
        axs[i].axis("off")
    
    k = np.random.randint(num_sims)
    axs[0].imshow(theta[k].reshape(img_size, img_size), cmap = "magma")

    for i in range(1, 5):
        axs[i].imshow(samples[i, k].reshape(img_size, img_size), cmap = "magma")
    
    plt.savefig("../../images/tarp/sanity.jpeg", bbox_inches="tight", pad_inches=0.2)
    print("Running the tarp test...")
    print(samples.shape, theta.shape)
    
    
    if args.bootstrapping: 
        print("Applying bootstrapping method")
        k = args.uncertainty
        ecp, ecp_std, alpha = bootstrapping(samples, theta, references = "random", metric = "euclidean", norm = False)
        #ecp, ecp_std, alpha = np.insert(ecp, 0, 0), np.insert(ecp_std, 0, 0), np.insert(alpha, 0, 0)
        #print(ecp.shape)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi = 150)
        ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
        ax.plot(alpha, ecp, label='DRP')
        ax.fill_between(alpha, ecp - k*ecp_std, ecp + k* ecp_std, alpha = 0.4, color = "orange", label = "Uncertainty zone ($3\sigma$)")
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")
        plt.title(args.title, fontsize = 10)
    else: 
        print("Applying a regular method")
        ecp, alpha = get_drp_coverage(samples, theta, references = "random", metric = "euclidean", norm = False)
        #ecp, alpha = np.insert(ecp, 0, 0), np.insert(alpha, 0, 0)
        print(ecp.shape)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi = 150)
        ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
        ax.plot(alpha, ecp, label='DRP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")
        plt.title(args.title, fontsize = 10)
    
    
    plt.savefig("../../images/tarp/test_euler.jpeg", bbox_inches = "tight", pad_inches = 0.2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--sampler",        required = True,       help = "Sampler used to create the posterior samples")
    parser.add_argument("--title",          required = False,       help = "Title of the plot",                                      default = "Euler")
    parser.add_argument("--bootstrapping",  required = False,  default = False, help = "Whether to apply bootstrapping or not for the tarp test", type = bool)
    parser.add_argument("--uncertainty",    required = False,      help = "Size of the uncertainty zone in the plot", type = float, default = 3)

    # ADD PARAMETER TO CHANGE THE SIZE OF THE UNCERTAINTY ZONE:
    args = parser.parse_args()
    main(args) 