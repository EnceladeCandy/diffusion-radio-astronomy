import os
import sys
sys.path.append("..\\scripts\\")
import torch
from torch.func import vmap, grad 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import binned_statistic_2d
import mpol.constants as const
from typing import Callable


from score_models import ScoreModel, NCSNpp
import json

N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args): 
    
    # Importing the models hparams and weights
    file = open("/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942/model_hparams.json")
    model_hparams = json.load(file)
    sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]
    score_model = ScoreModel(checkpoints_directory="/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942")

    # Loading the visibilities, uv coverage, weights... 
    data = np.load("../../data_npz/HTLup_continuum_full.npz")

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


    # TODO : Put all the functions in a helper file so that the code is more readable.
    pixel_scale = 0.0015
    # pad = 768//2 # padding to 1024 pixels
    pad = (4096 - 256)//2 # padding to 2048
    # pad = 128 # padding to 512
    img_size = 256
    npix = img_size + 2 * pad
    u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = npix)
    pixel_size = u_edges[1] - u_edges[0] # this is delta_u, and we should probably call it that in the future. My bad
    
    def ft(x): 
        return torch.fft.fft2(x, norm = "ortho")

    # C = 0.5 # VE prior
    # B = 0.5 # VE
    C = 1 # VP prior
    B = 0 # VP
    def link_function(x):
        return C * x + B

    def flip(x): 
        return x[:, ::-1]
    
    def noise_padding(x, pad, sigma):
        H, W = x.shape
        out = torch.nn.functional.pad(x, (pad, pad, pad, pad)) 
        # Create a mask for padding region
        mask = torch.ones_like(out)
        mask[pad:pad + H, pad:pad+W] = 0.
        # Noise pad around the model
        z = torch.randn_like(out) * sigma
        out = out + z * mask
        return out

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
    
    def pillbox_window(u, center, pixel_size=pixel_size, m=1):
        """
        u: coordinate of the data points to be aggregated (u or v)
        center: coordinate of the center of the pixel considered. 
        pixel_size: Size of a pixel in the (u,v)-plane
        m: size of the truncation of this window (in term of pixel_size)
        """
        return np.where(np.abs(u - center) <= m * pixel_size / 2, 1, 0)


    def sinc_window(u, center, pixel_size=pixel_size, m=1):
        """
        u: coordinate of the data points to be aggregated (u or v)
        center: coordinate of the center of the pixel considered. 
        pixel_size: Size of a pixel in the (u,v)-plane
        m: size of the truncation of this window (in term of pixel_size)
        """
        return np.sinc(np.abs(u - center) / m / pixel_size)

    def bin_data(u, v, values, weights, bins, window_fn, truncation_radius, statistics_fn="mean", verbose=0):
        u_edges = bins[0]
        v_edges = bins[1]
        n_coarse = 0
        grid = np.zeros((len(u_edges)-1, len(v_edges)-1))
        if verbose:
            print("Fitting the KD Tree on the data...")
        # Build a cKDTree from the data points coordinates to query uv points in our truncation radius
        uv_grid = np.vstack((u.ravel(), v.ravel())).T
        tree = cKDTree(uv_grid)
        if verbose:
            print("Gridding...")
        for i in tqdm(range(len(u_edges)-1), disable=not verbose):
            for j in range(len(v_edges)-1):
                # Calculate the coordinates of the center of the current cell in our grid
                u_center = (u_edges[i] + u_edges[i+1])/2
                v_center = (v_edges[j] + v_edges[j+1])/2
                # Query the tree to find the points within the truncation radius of the cell
                indices = tree.query_ball_point([u_center, v_center], truncation_radius, p=1) # p=1 is the Manhattan distance (L1)
                # Apply the convolutional window and weighted averaging
                if len(indices) > 0:
                    value = values[indices]
                    weight = weights[indices] * window_fn(u[indices], u_center) * window_fn(v[indices], v_center)
                    #if len(indices) == 1 and verbose > 1:
                        #print(f"Cell {(i, j)} has a single visibility and weight {weight.sum()} {weight}...")
                    if weight.sum() > 0.: # avoid dividing by a normalization = 0
                        if statistics_fn=="mean":
                            grid[j, i] = (value * weight).sum() / weight.sum()
                        elif statistics_fn=="std":
                            m = 1
                            if (weight > 0).sum() < 5:
                                # run statistics on a larger neighborhood
                                while (weight > 0).sum() < 5: # this number is a bit arbitrary, we just hope to get better statistics
                                    m += 0.1
                                    indices = tree.query_ball_point([u_center, v_center], m*truncation_radius, p=1) # p=1 is the Manhattan distance (L1)
                                    value = values[indices]
                                    weight = weights[indices] * window_fn(u[indices], u_center, m = m) * window_fn(v[indices], v_center, m = m)
                                #print(f"Coarsened pixel to {m} times its size, now has {len(indices)} for statistics")
                                n_coarse += 1
                            # See https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
                            # See more specifically the bootstrapping
                            if np.sum(weight > 0) < 2:
                                print("Low weight")

                            #N_eff taken from Bevington, see the page: http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
                            importance_weights = window_fn(u[indices], u_center, m = m) * window_fn(v[indices], v_center, m = m)
                            n_eff = np.sum(importance_weights)**2 / np.sum(importance_weights**2)
                            grid[j, i] = np.sqrt(np.cov(value, aweights=weight, ddof = 0)) * (n_eff / (n_eff - 1)) * 1/(np.sqrt(n_eff))
                        elif statistics_fn=="count":
                            grid[j, i] = (weight > 0).sum()
                        elif isinstance(statistics_fn, Callable):
                            grid[j, i] = statistics_fn(value, weight)
        print(f"number of coarsened pix: {n_coarse}")
        return grid

    # Gridding visibilities
    pixel_scale = 0.0015
    img_size = 256
    u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = img_size)

    truncation_radius = pixel_size
    window_fn = partial(sinc_window, pixel_size=pixel_size)

    # Real part mean and count
    args = (uu, vv, vis_re, weight_, (u_edges, v_edges), window_fn, truncation_radius)
    vis_bin_re = bin_data(*args, statistics_fn="mean", verbose=1)
    std_bin_re = bin_data(*args, statistics_fn="std", verbose=2)
    counts = bin_data(*args, statistics_fn="count", verbose=1)

    # Image part mean
    args = (uu, vv, vis_imag, weight_, (u_edges, v_edges), window_fn, truncation_radius)
    vis_bin_imag = bin_data(*args, statistics_fn="mean", verbose=1)
    std_bin_imag = bin_data(*args, statistics_fn="std", verbose=2)

    # From object type to float
    vis_bin_re = vis_bin_re.astype(float)
    vis_bin_imag = vis_bin_imag.astype(float)
    std_bin_re = std_bin_re.astype(float) 
    std_bin_imag = std_bin_imag.astype(float)
    counts = counts.astype(float)

    # i.e. The sampling function where there is data in the uv plane
    mask = counts > 0

    # Upsample std 
    from scipy import interpolate
    def upsample(input_data, d):
        # Get the shape of the input grid
        rows, cols = input_data.shape
        # Define the x and y coordinates of the original grid
        x = np.linspace(0, cols - 1, cols)
        y = np.linspace(0, rows - 1, rows)

        # Create a bilinear interpolator object
        f_interp = interpolate.interp2d(x, y, input_data, kind='linear')

        # Define the x and y coordinates of the upsampled grid
        x_new = np.linspace(0, cols - 1, cols * d)
        y_new = np.linspace(0, rows - 1, rows * d)

        # Perform the interpolation to upsample the grid
        upsampled_data = f_interp(x_new, y_new)
        return upsampled_data

    # upsample std if needed
    if std_bin_re.shape[0] < vis_bin_re.shape[0]:
        assert vis_bin_re.shape[0] // std_bin_re.shape[0] == downsampling_factor
        std_bin_re = upsample(std_bin_re, downsampling_factor)
    if std_bin_imag.shape[0] < vis_bin_imag.shape[0]:
        assert vis_bin_imag.shape[0] // std_bin_imag.shape[0] == downsampling_factor
        std_bin_imag = upsample(std_bin_imag, downsampling_factor)

    #I am commenting this out and performing this process inside the binning function
    #std_bin_re /= (counts + 1)**0.5
    #std_bin_imag /= (counts + 1)**0.5

    # For the inference, fftshift is done in the forward model
    vis_gridded_re = np.fft.fftshift(vis_bin_re).flatten()
    vis_gridded_imag = np.fft.fftshift(vis_bin_imag).flatten()
    std_gridded_re = np.fft.fftshift(std_bin_re).flatten()
    std_gridded_imag = np.fft.fftshift(std_bin_imag).flatten()
    S_grid = np.fft.fftshift(mask).flatten()
    
    S_cat = np.concatenate([S_grid, S_grid])
    vis_gridded = np.concatenate([vis_gridded_re, vis_gridded_imag])
    std_gridded = np.concatenate([std_gridded_re, std_gridded_imag])

    # Numpy to torch: 
    S = torch.tensor(S_cat).to(device)
    y = torch.tensor(vis_gridded, device = S.device)[S].to(device) * npix
    sigma_y = torch.tensor(std_gridded, device = S.device)[S].to(device) * npix


    def sigma(t): 
        return sigma_min * (sigma_max/sigma_min) ** t


    def model(x):
        x = x.reshape(img_size, img_size) # for the FFT 
        x = link_function(x) # map from model unit space to real unit space
        x = torch.flip(x, dims = (1,))

        # Padding: 
        #pad_size = int((npix - img_size)/2)
        #x = torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size)) 
        vis_full = ft(torch.fft.fftshift(x)).flatten() 
        vis_sampled = vis_full
        vis_sampled = torch.cat([vis_sampled.real, vis_sampled.imag])
        return vis_sampled[S]


    def log_likelihood(y, x, t, sigma_y):
        """
        Calculate the log-likelihood of a gaussian distribution 
        Arguments: 
            y = processed gridded visibilities (real part and imaginary part concatenated)
            x = sky brightness 
            t = diffusion temperature
            A = linear model (sampling function and FT)  
        
        Returns: 
            log-likelihood of a gaussian distribution
        """ 
        y_hat = model(x)
        var = sigma(t) **2 / 2  + sigma_y**2
        log_prob = -0.5 * torch.sum((y - y_hat)**2 / var)
        return log_prob


    # GIVE THE GOOD COVARIANCE MATRIX
    def score_likelihood(y, x, t, sigma_y): 
        x = x.flatten(start_dim = 1) 
        return vmap(grad(lambda x, t: log_likelihood(y, x, t, sigma_y)))(x, t)

    #torch.manual_seed(0)
    def score_posterior(x, t): 
        x = x.reshape(-1, 1, img_size, img_size)
        return score_model.score(t, x).flatten(start_dim = 1) + score_likelihood(y, x, t, sigma_y) 

    def g(t): 
        return sigma(t) * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))


    def pc_sampler(num_samples, num_pred_steps, num_corr_steps, score_function, snr = 1e-2, img_size = 28): 
        t = torch.ones(size = (num_samples, 1)).to(device)
        x = torch.randn([num_samples, img_size ** 2]).to(device)
        dt = -1/num_pred_steps
        with torch.no_grad(): 
            for _ in tqdm(range(num_pred_steps-1)): 
                # Corrector step: (Only if we are not at 0 temperature )
                gradient = score_function(x, t)
                for _ in range(num_corr_steps): 
                    z = torch.randn_like(x)
                    grad_norm = torch.mean(torch.norm(gradient, dim = -1)) # mean of the norm of the score over the batch 
                    noise_norm = torch.mean(torch.norm(z, dim = -1))
                    epsilon =  2 * (snr * noise_norm / grad_norm) ** 2
                    x = x + epsilon * gradient + (2 * epsilon) ** 0.5 * z * dt  

            
                # Predictor step: 
                z = torch.randn_like(x).to(device)
                gradient = score_function(x, t)
                drift = 0
                diffusion = g(t)
                x_mean = x - diffusion**2 * gradient * dt  
                noise = diffusion * (-dt) ** 0.5 * z
                x = x_mean + noise
                t += dt
                if torch.any(torch.isnan(x_mean)):
                    print("Nans appearing")
                    break
                
        return link_function(x_mean)

    def euler_sampler(num_samples, num_steps, score_function, img_size = 28): 
        t = torch.ones(size = (num_samples, 1)).to(device)
        x = sigma_max * torch.randn([num_samples, img_size ** 2]).to(device)
        dt = -1/num_steps
        with torch.no_grad(): 
            for i in tqdm(range(num_steps - 1)): 
                z = torch.randn_like(x).to(device)
                gradient = score_function(x, t)
                drift = 0
                diffusion = g(t)
                x_mean = x - diffusion**2 * gradient * dt  
                noise = diffusion * (-dt) ** 0.5 * z
                x = x_mean + noise
                t += dt

                # if i==20:
                #     break

                if torch.any(torch.isnan(x_mean)):
                    print("Nans appearing")
                    break
        
        return link_function(x_mean)

    sampler = args.sampler
    pred = args.num_pred
    corr = args.num_corr
    snr = args.snr
    num_samples = args.num_samples 
    
    batch_size = args.batchsize # Maximum number of posterior samples per gpu with 16GB (beluga)
    samples_tot = torch.empty(size = (num_samples, img_size, img_size))

    if sampler.lower() == "euler":    
        for i in range(int(num_samples//batch_size)):
            print('Starting sampling procedure...')
            samples = euler_sampler(
                num_samples = batch_size,
                num_steps = pred, 
                score_function = score_posterior, 
                img_size = img_size
            )

            # Filling gradually the samples
            samples_tot[i * batch_size : (i + 1) * batch_size] = samples.reshape(-1, img_size, img_size)

        # Saving 
        folder_dir = f"../../results/euler/spw_{spw}"
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        torch.save(samples_tot, folder_dir + f"/{num_samples}samples.pt")

    elif sampler.lower() == "pc":
        for i in range(int(num_samples//batch_size)):
            samples = pc_sampler(
                num_samples = batch_size,
                num_pred_steps = pred,
                num_corr_steps = corr,
                snr = snr,
                score_function = score_posterior,
                img_size = img_size
            )
            # Filling gradually the samples
            samples_tot[i * batch_size : (i + 1) * batch_size] = samples.reshape(-1, img_size, img_size)

        # Saving 
        folder_dir = f"../../results/pc/spw_{spw}"
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        torch.save(samples_tot, folder_dir + f"/{num_samples}samples.pt")

    else: 
        raise ValueError("The sampler specified is not implemented or does not exist. Choose between 'euler' and 'pc'")

    # Plotting the dirty image, for my sanity
    fig = plt.figure(figsize = (8, 8), dpi = 200)
    plt.imshow(dirty_image.real, cmap = "magma", origin = "lower")
    plt.axis("off")
    plt.savefig(folder_dir + "/dirty_image.jpeg", bbox_inches = "tight", pad_inches = 0.01)

    

if __name__ == "__main__": 
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--sampler",            required = False,   default = "pc", type = str,      help = "Sampling procedure used ('pc' or 'euler')")
    parser.add_argument("--num_samples",        required = False,   default = 20,   type = int,     help = "Number of samples from the posterior to create")
    parser.add_argument("--num_pred",           required = False,   default = 1000, type = int,     help ="Number of predictor/euler steps in the loop")
    parser.add_argument("--num_corr",           required = False,   default = 20,   type = int,     help ="Number of corrector steps for the PC sampling")
    parser.add_argument("--snr",                required = False,   default = 1e-2, type = float,   help = "Signal-to-noise ratio for the PC sampling")
    parser.add_argument("--batchsize",          required = True,    default = 25,   type = int,     help = "Number of samples created per iteration")
    args = parser.parse_args()
    main(args) 