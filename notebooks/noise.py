from tqdm import tqdm
from mpol import coordinates
from mpol.gridding import DirtyImager
import numpy as np

pixel_scale = 0.0015 # arcsec
npix = 256 # Number of pixels in the grid
coords = coordinates.GridCoords(cell_size=pixel_scale, npix=npix)
img_size = 256 # Number of pixels in the reconstructed image


fname = "../../HTLup_continuum.npz"
data = np.load(fname)
vis = data["data"]
weight = data["weight"]
u = data["u"]
v = data["v"] # Normalized weights

noisy_vis_gridded = np.empty(shape = (1000, npix, npix), dtype = np.complex128)
N_vis = len(vis)
for i in tqdm(range(1000)):

    sigma = weight ** -0.5
    eta_re = np.random.normal(loc = np.zeros(N_vis), scale = sigma)
    eta_im = np.random.normal(loc = np.zeros(N_vis), scale = sigma)
    imager = DirtyImager(
    coords=coords,
    uu=u,
    vv=v,
    weight=weight,
    data_re=eta_re,
    data_im=eta_im
    )
    robust = -0.4
    imager._grid_visibilities(weighting = "briggs", robust = robust, taper_function = None)
    vis_gridded = imager.vis_gridded[0] * imager.C
    noisy_vis_gridded[i] = vis_gridded

np.save("noise_robust-04.npy", noisy_vis_gridded.std(axis = 0))