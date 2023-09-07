    

def main(args):
    import sys
    sys.path.append("..\\scripts\\")
    import torch 
    from astropy.io import fits
    from astropy.visualization import ImageNormalize, AsinhStretch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.func import vmap, grad
    from tqdm import tqdm
    from scipy.stats import binned_statistic_2d

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 10
    from score_models import ScoreModel, NCSNpp
    import json
    from mpol import coordinates
    from mpol.gridding import DirtyImager

    plt.style.use("dark_background")

    from utils import fits_to_tensor, link_function
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Importing the models hparams and weights
    file = open("../../ncsnpp_ct_g_220912024942/model_hparams.json")
    model_hparams = json.load(file)
    sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]

    # Importing the weights
    score_model = ScoreModel(checkpoints_directory="../../ncsnpp_ct_g_220912024942")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")

    def ift(x): 
        return torch.fft.ifft2(x, norm = "ortho")


    # Observation
    data = np.load("../../HTLup_continuum2.npz")
    u = data["uu"] # klambda units
    v = data["vv"]
    vis = data["data"]
    #wavelength = data["wavelength"]
    weight = data["weight"]

    # Loading the psf and the dirty image
    pixel_scale = 0.0015 # arcsec
    npix = 256 # Number of pixels in the grid
    coords = coordinates.GridCoords(cell_size=pixel_scale, npix=npix)
    img_size = 256 # Number of pixels in the reconstructed image

    imager = DirtyImager(
        coords=coords,
        uu=u,
        vv=v,
        weight=weight,
        data_re=vis.real,
        data_im=vis.imag
    )

    robust = -0.4
    dirty_image, beam = imager.get_dirty_image(weighting=args.weighting, robust=args.robust)
    dirty_image, beam = torch.tensor(dirty_image.copy()).to(device), torch.tensor(beam.copy()).to(device)
    # imgs.append(dirty_image)
    # beams.append(beam)


    # Creating the complex conjugate for each visibility
    uu = np.concatenate([u, u])
    vv = np.concatenate([v, v])
    vis_re = np.concatenate([vis.real, vis.real])
    vis_imag = np.concatenate([vis.imag, -vis.imag])
    weight = np.concatenate([weight, weight])

    bin_x = coords.u_edges
    bin_y = coords.v_edges
    std_bin_real, edgex, edgey, binumber = binned_statistic_2d(vv, uu, vis_re, "std", (bin_y, bin_x))
    std_bin_imag, edgex, edgey, binumber = binned_statistic_2d(vv, uu, vis_imag,  "std", (bin_y, bin_x))

    vis_bin_real, edgex, edgey, binumber = binned_statistic_2d(vv, uu, vis_re, "mean", (bin_y, bin_x))
    vis_bin_imag, edgex, edgey, binumber = binned_statistic_2d(vv, uu, vis_imag,  "mean", (bin_y, bin_x))
    
    count, *_ = binned_statistic_2d(vv, uu, vis_re, "count", (bin_y, bin_x))

    std_real = (std_bin_real / (count + 1))
    std_imag = (std_bin_imag / (count + 1))

    std_bin_real[np.isnan(std_bin_real)] = 0
    std_bin_imag[np.isnan(std_bin_imag)] = 0

    S = std_bin_real>0.0

    std_bin_real = torch.fft.fftshift(torch.tensor(std_bin_real)).to(device)
    std_bin_imag = torch.fft.fftshift(torch.tensor(std_bin_imag)).to(device)


if __name__ == "main": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "pc", type = str,      help = "Sampling procedure used ('pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,   type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--num_pred",          required = False,   default = 1000, type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--num_corr",     required = False,   default = 20,   type = int,     help ="Number of iterations in the loop to compute the reverse sde")
    parser.add_argument("--snr",                required = False,   default = 1e-2, type = float)
    
    # Ground-truth parameters
    parser.add_argument("--ms",                required = False,   default = "HTLup_COcube" ,    type = str,     help = "Name of the target") 
    parser.add_argument("--batchsize",          required = False,  type = int,  default = 1)

    # Gridding parameters: 
    parser.add_argument("--weighting",         required = True,   type = str,   help = "Weighting procedure to follow using tclean's convention")
    parser.add_argument("--robust",            required = False,  type = float, help = "Robust parameter for briggs weighting, must be between -2 and 2", default = None)
    args = parser.parse_args()
    main(args) 










