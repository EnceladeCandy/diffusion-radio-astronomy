from astropy.visualization import ImageNormalize, AsinhStretch, LogStretch
import scienceplots
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
import colorcet as cc
import h5py
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from astropy.wcs import WCS
import os
import sys

sys.path.append("..\\")
plt.style.use("science") # Need SciencePLots
params = {
         'axes.labelsize': 25,
         'axes.titlesize': 30,
         'ytick.labelsize' :20,
         'xtick.labelsize' :20,
         'xtick.major.size': 8,
         'xtick.minor.size': 4,
         'xtick.major.width': 1,
         'xtick.minor.width': 1,
         'ytick.color': "k",
         'xtick.color': "k",
         'axes.labelcolor': "k",
         'ytick.labelcolor' : "k",
         'xtick.labelcolor' : "k",
         }
pylab.rcParams.update(params)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    ms = args.ms
    sampler = args.sampler

    dir_results = f"/home/noedia/scratch/tarp_samples/{sampler}/"

    pattern = args.experiment_name + "*.h5"
    paths = glob(dir_results + pattern)
    
    # 40 samples per file
    N = 40
    samples = np.empty(shape = (N * len(paths), 256, 256)) # (N_samples, img_size, img_size)

    for i, path in tqdm(enumerate(paths)):
        with h5py.File(path, "r") as hf:
            hf.keys()
            samples[N*i:N*(i+1), :] = np.array(hf["model"])
    

    # Creating the dirty image
    vis_bin_re = np.load(os.path.join(args.data_dir, "allspw_re.npy"))
    vis_bin_imag = np.load(os.path.join(args.data_dir, "allspw_imag.npy"))
    dirty_image = np.fft.ifft2(vis_bin_re + 1j * vis_bin_imag, norm = "ortho")


if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Ground-truth parameters
    parser.add_argument("--ms",                required = False,   default = "HTLup_continuum0.0015arcsec2" ,    type = str,     help = "Name of the target") 
    parser.add_argument("--sampler",           required = False,   default = "euler")
    parser.add_argument("--results_dir",       required = True)
    parser.add_argument("--experiment_name",   required = True)
    args = parser.parse_args()
    main(args) 
