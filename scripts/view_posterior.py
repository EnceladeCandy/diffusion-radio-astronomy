import torch
import h5py
import os 
import matplotlib.pyplot as plt
from utils import probes_64
from tqdm import tqdm
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


# Importing the dataset
dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
hdf = h5py.File(dir_galaxies, "r")
hdf.keys()
dataset = hdf['galaxies']
img_size = 64

sampler = "euler"

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
    theta[i, :] = probes_64(dataset, k).flatten()

# tarp is coded for numpy
samples = samples.numpy()
theta = theta.numpy()

# Sanity check: 

fig, axs = plt.subplots(5, 5, figsize = (7, 7))
for i in range(5):
    for j in range(5):
        axs[i, j].axis("off")

k = np.random.randint(num_sims, size = (5,))

for i in range(5):
    axs[i, 0].imshow(theta[k[i]].reshape(img_size, img_size), cmap = "magma")
    for j in range(1, 5):
        axs[i, j].imshow(samples[j, k[i]].reshape(img_size, img_size), cmap = "magma")
plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
plt.savefig("../../images/tarp/sanity.jpeg", bbox_inches="tight", pad_inches=0.2)