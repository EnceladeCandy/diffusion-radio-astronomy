import torch 
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import numpy as np
import h5py
plt.style.use("dark_background")
device = "cuda" if torch.cuda.is_available() else "cpu"

dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
hdf = h5py.File(dir_galaxies, "r")
hdf.keys()
dataset = hdf['galaxies']


############ FIRST PLOT ###################
# Plot naive reconstruction: 
def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")

def ift(x): 
    return torch.fft.ifft2(x, norm = "ortho")


psf = torch.load("../psf64.pt")
img = dataset[101,...,0]
img = F.avg_pool2d(torch.tensor(img[None, None, ...]), (4, 4))[0, 0].to(device)
img = torch.load("../ground_truth.pt")
img_size = img.shape[-1]

vis_full = ft(img).flatten()
sampling_function= ft(torch.fft.ifftshift(psf)).flatten()
vis_sampled = sampling_function * vis_full

sigma_likelihood = 1e-5
vis_sampled = vis_sampled.flatten()
vis_sampled = torch.cat([vis_sampled.real, vis_sampled.imag])

y_dim = len(vis_sampled)  
dist_likelihood = MultivariateNormal(loc = torch.zeros(y_dim).to(device), covariance_matrix=sigma_likelihood **2 * torch.eye(y_dim).to(device))
eta = dist_likelihood.sample([])

y = vis_sampled + eta 
fig, axs = plt.subplots(1, 2, figsize = (10, 4))
dirty_image_noise = ift((y[:img_size**2] + 1j * y[img_size**2:]).reshape(img_size, img_size)).real

for i in range(len(axs)): 
    axs[i].axis("off")

axs[0].imshow(img.reshape(img_size, img_size).cpu(), cmap = "hot")
axs[0].set_title("Ground-truth")
axs[1].imshow(dirty_image_noise.cpu(), cmap = "hot")
axs[1].set_title("Dirty image with noise")
plt.subplots_adjust(wspace = 0.1)
plt.savefig("../images/naive_rec_1e-5.jpeg", bbox_inches = "tight", pad_inches = 0.1)



############ SECOND PLOT ################### 
# Importing the posterior samples
samples = torch.load("../../samples/idx_101.pt").reshape(-1, 64, 64)
num_samples = samples.shape[0]
img_size = int((samples.shape[1]))

# Creating the plot
grid_size = int((num_samples)**0.5) # Square grid, same number of rows and cols:
print(grid_size)
if grid_size==1:
     fig = plt.figure(figsize= (8,8), dpi = 150)
     plt.imshow(samples[0].cpu(), cmap = "hot")
     plt.axis("off")
     plt.savefig("../images/posterior_1e-5.jpg", bbox_inches = "tight", pad_inches = 0.1)

else:
    fig, axs = plt.subplots(grid_size, grid_size, figsize = (10, 10), dpi = 150)

    k = 0
    for i in range(grid_size): 
        for j in range(grid_size): 
            axs[i, j].imshow(samples[k].cpu(), cmap = "hot")
            axs[i, j].axis("off")
            k += 1

    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    plt.savefig("../images/posterior_idx101.jpg", bbox_inches = "tight", pad_inches = 0.1)
