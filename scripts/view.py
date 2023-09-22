from astropy.visualization import ImageNormalize, AsinhStretch, LogStretch
import scienceplots
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
import colorcet as cc
import numpy as np
import torch
import os
import mpol.constants as const
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d

device = "cuda" if torch.cuda.is_available() else "cpu"


N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))

spw = THIS_WORKER
data = np.load("../../data_npz/HTLup_continuum_full.npz")

# Creating the dirty image: 
u = data["uu"]
v = data["vv"]
vis = data["data"]
weight = data["weight"]
vis_per_spw = data["vis_per_spw"]


# Choosing a specific spectral window
indices = vis_per_spw.cumsum()
spw = THIS_WORKER

if spw == 0: 
    idx_inf = 0
else: 
    idx_inf = indices[spw-1]

idx_sup = indices[spw]
u = u[idx_inf:idx_sup]
v = v[idx_inf:idx_sup]
vis = vis[idx_inf:idx_sup]
weight = weight[idx_inf:idx_sup]

uu = np.concatenate([u, -u])
vv = np.concatenate([v, -v])

vis_re = np.concatenate([vis.real, vis.real])
vis_imag = np.concatenate([vis.imag, -vis.imag])
weight_ = np.concatenate([weight, weight])

def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")

def link_function(x): 
    return (x+1) / 2 

def flip(x): 
    return x[:, ::-1]

def grid(pixel_scale, img_size): 
    """Given a pixel scale and a number of pixels in image space, grid the associated Fourier space

    Args:
        pixel_scale (float): Pixel resolution (in arcsec)
        img_size (float/int): Size of the image 

    Returns:
        
    """

    # Arcsec to radians: 
    dl = pixel_scale * const.arcsec
    dm = pixel_scale * const.arcsec

    du = 1 / (img_size * dl) * 1e-3 # klambda
    dv = 1 / (img_size * dm) * 1e-3 # klambda

    u_min = -img_size//2 * du 
    u_max = img_size//2 * du 

    v_min = -img_size//2 * dv
    v_max = img_size//2 * dv

    u_edges = np.linspace(u_min, u_max, img_size + 1)
    v_edges = np.linspace(v_min, v_max, img_size + 1)

    return u_edges, v_edges


# Gridding visibilities
pixel_scale = 0.0015
img_size = 256
u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = img_size)

bin_x = u_edges
bin_y = v_edges
vis_gridded_re, edgex, edgey, binnumber = binned_statistic_2d(vv, uu, vis_re, "mean", (bin_y, bin_x))
vis_gridded_img, edgex, edgey, binnumber = binned_statistic_2d(vv, uu, vis_imag, "mean", (bin_y, bin_x))

vis_gridded_re[np.isnan(vis_gridded_re)] = 0
vis_gridded_img[np.isnan(vis_gridded_img)] = 0

vis_grid = vis_gridded_re + 1j * vis_gridded_img

dirty_image = img_size ** 2 * flip(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(vis_grid)))).real

folder_dir = f"../../results/euler/spw_{spw}"
folder_plot = "../../results/plots"

if not os.path.exists(folder_plot):
    os.makedirs(folder_plot)


samples = torch.load(folder_dir + "/100samples.pt", map_location= "cpu").numpy()

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
plt.rcParams["text.usetex"] = True

def draw_scale_and_compass(ax, cutout, wcs, rotation=0, scale=1, x0=0.1, y0=0.1, compass_size=0.1, arrow_size=0.001, color="k", lw=2, textpad=5, fontsize=15):
    hdr = wcs.to_header()
    pixel_scale_x, pixel_scale_y, *_ = proj_plane_pixel_scales(wcs) # in degrees /pixels
    pixel_scale = np.mean([pixel_scale_x, pixel_scale_y]) # for drawing arrows
    M, N = cutout.shape
    
    # nice doc here http://montage.ipac.caltech.edu/docs/headers.html
    # Remember that PC maps pixels to world, so it include a mirror flip of the j coordinate (East-West)
    pc_matrix = wcs.pixel_scale_matrix
    north = np.array([0, 1])
    east = np.array([1, 0]) # accounts for the mirror flip of that coordinate to stay in pixel space
    assert rotation % 90 == 0, "Rotation should only be used for images flipped by 90 or 180 degrees"
    theta = rotation * np.pi / 180 / 2 # divide by two to apply this to the PC matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    pc_matrix = R @ pc_matrix @ R.T
    
    # North arrow 
    ndx = (pc_matrix @ north)[0] * compass_size * N / pixel_scale
    ndy = (pc_matrix @ north)[1] * compass_size * N / pixel_scale
    # East arrow
    edx = (pc_matrix @ east)[0] * compass_size * N / pixel_scale
    edy = (pc_matrix @ east)[1] * compass_size * N / pixel_scale
    # some convenient padding solutions
    nrx = abs(ndx) / np.sqrt(ndx**2 + ndy**2) 
    nry = abs(ndy) / np.sqrt(ndx**2 + ndy**2)
    erx = edx / np.sqrt(edx**2 + edy**2)
    ery = -edy / np.sqrt(edx**2 + edy**2)
    # Orientation of the letters N and E
    angle = (np.arctan2(np.abs(ndy), ndx) - np.pi / 2) * 180 / np.pi
    
    # Scale bar
    ax.plot([x0*N, x0*N + scale/pixel_scale_y/3600], [y0*M, y0*M], linewidth=lw, color=color)
    ax.text(x0*N + scale/pixel_scale_x/2/3600, y0*M+textpad/2, f"{abs(scale)}''", fontsize=fontsize, color=color, ha='center', weight='bold')

    # North arrow, on the bottom right corner, pointing up
    ax.arrow(x=(1 - x0)*N, y=y0*M, dx=ndx, dy=ndy, head_width=abs(arrow_size)*N, head_length=abs(arrow_size)*N, fc=color, ec=color, linewidth=lw)
    ax.text((1 - x0)*N+ndx+textpad*nrx, y0*M+ndy+textpad*nry, "N", color=color, fontsize=fontsize, ha='center', rotation=angle, weight='bold')
    
    # East arrow, on the bottom right corner, pointing left
    ax.arrow(x=(1 - x0)*N, y=y0*M, dx=edx, dy=edy, head_width=abs(arrow_size)*N, head_length=abs(arrow_size)*N, fc=color, ec=color, linewidth=lw)
    ax.text((1 - x0)*N+edx+4*textpad*erx/3, y0*M+edy+4*textpad*ery/3, "E", color=color, fontsize=fontsize, ha='center', rotation=angle, weight='bold')
    return ax

from astropy.wcs import WCS
hdr_dict = {'WCSAXES': 2,
 'CRPIX1': 256 / 2 - 0.5,
 'CRPIX2': 256 / 2 - 0.5,
 'CDELT1': -0.0015 / 3600, # pixel scale
 'CDELT2': 0.0015 / 3600,
 'CUNIT1': 'deg',
 'CUNIT2': 'deg',
 'CTYPE1': 'RA---SIN',
 'CTYPE2': 'DEC--SIN',
 'CRVAL1': 236.3035304583,
 'CRVAL2': -34.29194894083,
 'PV2_1': 0.0,
 'PV2_2': 0.0,
 'LONPOLE': 180.0,
 'LATPOLE': -34.29194894083,
 'RESTFRQ': 230538000000.0,
 'TIMESYS': 'utc',
 'MJDREF': 0.0,
 'DATE-OBS': '2017-05-14T04:29:02.640000',
 'MJD-OBS': 57887.186836111,
 'OBSGEO-X': 2225142.180269,
 'OBSGEO-Y': -5440307.370349,
 'OBSGEO-Z': -2481029.851874,
 'RADESYS': 'FK5',
 'EQUINOX': 2000.0,
 'SPECSYS': 'LSRK'}
my_wcs = WCS(hdr_dict)

fig = plt.figure(figsize=(30, 6))
#cmap = cc.cm["CET_CBL3"]
cmap_std = cc.cm["fire"]
cmap = "magma"
#norm = plt.cm.colors.Normalize(vmin=0, vmax=samples.max()) # change so it makes sense with your units
norm_std = ImageNormalize(vmin=0, vmax=samples.std(axis = 0).max()) # also change this as well
norm = ImageNormalize(samples, vmin = 0, stretch = AsinhStretch())
wcs = my_wcs # World coordinate system, specified by fits header above

# Dirty Image
img = dirty_image
ax = fig.add_subplot(151, projection=wcs)
im = ax.imshow(img, cmap=cmap, norm=norm)
draw_scale_and_compass(ax, img, wcs, scale=0.1, x0=0.1, y0=0.1, compass_size=0.1, arrow_size=0.01, color="w", lw=2, textpad=7, fontsize=25)
ax.tick_params(axis='both', which='both', color='white')
ax.set_ylabel("Declination")
ax.set_xlabel("Right ascension")
ax.set_title("Dirty image")

for i in range(2):
    img = samples[i] # change for a posterior sample
    ax = fig.add_subplot(int(f"15{i+2}"), projection=wcs)
    ax.imshow(img, cmap=cmap, norm=norm)
    ax.axis("off")
    ax.tick_params(axis='both', which='both', color='white')
    ax.set_title("Posterior sample")


# Mean image (or mode img)
img = samples.mean(axis = 0)
ax = fig.add_subplot(int(f"154"), projection=wcs)
im1 = ax.imshow(img, cmap=cmap, norm=norm)
ax.axis("off")
ax.tick_params(axis='both', which='both', color='white')
ax.set_title("Posterior mean") # or mode

# Standard deviation
img = samples.std(axis = 0)
ax = fig.add_subplot(int(f"155"), projection=wcs)
im2 = ax.imshow(img, cmap=cmap_std, norm=norm_std)
ax.axis("off")
ax.tick_params(axis='both', which='both', color='white')
ax.set_title("Posterior deviation") # or mode

cbar_ax = fig.add_axes([0.35, 0.05, 0.18, 0.055])  # Position colorbar at [left, bottom, width, height]
cb1 = plt.colorbar(im1, cax=cbar_ax, orientation="horizontal")
cbar_ax.set_xlabel(r"Probes units")

cbar_ax2 = fig.add_axes([0.905, 0.12, 0.01, 0.75])
fig.colorbar(im2, cax=cbar_ax2)
cbar_ax2.set_ylabel(r"Probes units")

plt.subplots_adjust(hspace=0, wspace=0.01)
plt.savefig(folder_plot + f"/posterior_spw{spw}.jpeg")