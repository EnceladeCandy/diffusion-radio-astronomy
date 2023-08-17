import torch 
import os
from utils import fits_to_tensor, ft
from astropy.io import fits
import matplotlib.pyplot as plt

path = "../../data_targets2/noisy/"


files = []
for file in os.listdir(path): 
    if file.endswith(".fits"): 
        files.append(file)

img_size = int(files[0].split("_")[3])
noise_vis = torch.empty(size = (len(files), img_size, img_size), dtype = torch.complex64)
noise_vis_scaled = torch.empty(size = (len(files), img_size, img_size), dtype = torch.complex64)

for i, file in enumerate(files): 
    file_path = path + file 

    # Importing dirty image
    header, dirty_image = fits_to_tensor(file_path)

    # Calculating the gridded visibilities V = F(I)
    vis = ft(dirty_image)
    scaled_vis = ft(dirty_image/dirty_image.max())
    
    noise_vis[i] = vis 
    noise_vis_scaled[i] = scaled_vis
    

std_vis_real = noise_vis.real.std(dim = 0)
std_vis_imag = noise_vis.imag.std(dim = 0)
std_scaled_vis = noise_vis_scaled.std(dim = 0)

print(std_vis_real)
plt.imshow(dirty_image.cpu(), cmap = "magma")
plt.savefig("test.jpeg", bbox_inches = "tight", pad_inches = 0.2)

















































































































































































































