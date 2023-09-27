import matplotlib.pyplot as plt
import numpy as np 
import os
from glob import glob
import h5py
from tqdm import tqdm
from typing import Tuple, Union
from tarp_perso import bootstrapping, get_drp_coverage



def main(args):

    # Importing the dataset
    sampler = args.sampler
    experiment_name = args.experiment_name
    img_size = args.img_size
    dir_results = f"/home/noedia/scratch/tarp_samples/{sampler}/"

    pattern = experiment_name + "*.h5"
    paths = glob(dir_results + pattern)

    num_samples = 300
    num_sims = len(paths)
    num_dims = args.img_size ** 2

    # Posterior samples
    samples = np.empty(shape = (num_samples, num_sims, num_dims)) # (n_samples, n_sims, n_dims)

    # Ground-truths
    theta = np.empty(shape = (num_sims, num_dims)) # (n_sims, n_dims)
    


    for i, path in tqdm(enumerate(paths)):
        with h5py.File(path, "r") as hf:
            hf.keys()
            samples[:, i, :] = np.array(hf["model"]).reshape(600, num_dims)[:300, :]
            theta[i, :] = np.array(hf["ground_truth"]).flatten()

 
    print("Importing the samples and the ground-truths...")
    # for i in tqdm(range(num_sims)):
    #     samples[:, i, :] = np.load(path + "/" + samples_files[i], map_location=torch.device(device)).flatten(start_dim = 1)
    #     k = int(samples_files[i].split("_")[-1].replace(".pt", ""))
    #     theta[i, :] = probes_64(dataset, k).flatten()

    # # tarp is coded for numpy
    # samples = samples.numpy()
    # theta = theta.numpy()
    
    # Sanity check: 

    fig, axs = plt.subplots(1, 5, figsize = (5*3.5, 3.5))
    for i in range(len(axs)):
        axs[i].axis("off")
    
    k = np.random.randint(num_sims)
    axs[0].imshow(theta[k].reshape(img_size, img_size), cmap = "magma")

    for i in range(1, 5):
        axs[i].imshow(samples[i, k].reshape(img_size, img_size), cmap = "magma")
    
    tarp_folder = "../../plots_tarp/"
    plt.savefig(tarp_folder + f"{sampler}_{args.file_name}2.jpeg", bbox_inches="tight", pad_inches=0.2)
    print("Running the tarp test...")
    
    
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
        #plt.title(args.title, fontsize = 10)
    else: 
        print("Applying a regular method")
        ecp, alpha = get_drp_coverage(samples, theta, references = "random", metric = "euclidean", norm = True)
        #ecp, alpha = np.insert(ecp, 0, 0), np.insert(alpha, 0, 0)
        print(ecp.shape)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi = 150)
        ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
        ax.plot(alpha, ecp, label='DRP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")
        plt.title(args.title, fontsize = 10)
    
    
    plt.savefig(tarp_folder + f"bootstrap{sampler}_{args.file_name}.jpeg", bbox_inches = "tight", pad_inches = 0.2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--sampler",        required = True,       help = "Sampler used to create the posterior samples")
    parser.add_argument("--title",          required = False,       help = "Title of the plot",                                      default = "Euler")
    parser.add_argument("--bootstrapping",  required = False,  default = False, help = "Whether to apply bootstrapping or not for the tarp test", type = bool)
    parser.add_argument("--uncertainty",    required = False,      help = "Size of the uncertainty zone in the plot", type = float, default = 3)
    parser.add_argument("--experiment_name")
    parser.add_argument("--file_name",       required = False)
    parser.add_argument("--img_size",   type = int)
    # ADD PARAMETER TO CHANGE THE SIZE OF THE UNCERTAINTY ZONE:
    args = parser.parse_args()
    main(args) 
