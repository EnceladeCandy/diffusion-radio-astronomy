import matplotlib.pyplot as plt
import numpy as np 
import os
import torch
import h5py
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
from tarp_perso import bootstrapping 



def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Importing the dataset
    dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
    hdf = h5py.File(dir_galaxies, "r")
    hdf.keys()
    dataset = hdf['galaxies']
    img_size = 64
    

    path = args.samples_dir
    samples_files = os.listdir(path)

    num_samples = 500
    num_sims = len(samples_files)
    num_dims = img_size ** 2
    samples = torch.empty(size = (num_samples, num_sims, num_dims)) # (n_samples, n_sims, n_dims)
    theta = torch.empty(size = (num_sims, num_dims)) # (n_sims, n_dims)

    print("Importing the samples and the ground-truths...")
    for i in tqdm(range(num_sims)):
        samples[:, i, :] = torch.load(path + "/" + samples_files[i], map_location=torch.device(device))
        k = int(samples_files[i].split("_")[-1].replace(".pt", ""))
        theta[i, :] = probes_64(dataset, k).flatten()

    # tarp is coded for numpy
    samples = samples.numpy()
    theta = theta.numpy()

    print("Running the tarp test...")
    ecp, ecp_std, alpha = bootstrapping(samples, theta, references = "random", metric = "euclidean", norm = False)
    ecp, ecp_std, alpha = np.insert(ecp, 0, 0), np.insert(ecp_std, 0, 0), np.insert(alpha, 0, 0)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi = 150)
    ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    ax.plot(alpha, ecp, label='DRP')
    ax.fill_between(alpha, ecp - 3*ecp_std, ecp + 3* ecp_std, alpha = 0.4, color = "orange", label = "Uncertainty zone ($3\sigma$)")
    ax.legend()
    ax.set_ylabel("Expected Coverage")
    ax.set_xlabel("Credibility Level")
    plt.title(args.title, fontsize = 10)
    plt.savefig("images/tarp/200sims_sigma1e-4.jpeg", bbox_inches = "tight", pad_inches = 0.2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--samples_dir",    required = True,       help = "Folder directory where the samples for each simulation are saved")
    parser.add_argument("--title",    required = True,       help = "Title of the plot")
    # ADD PARAMETER TO CHANGE THE SIZE OF THE UNCERTAINTY ZONE:
    args = parser.parse_args()
    main(args) 