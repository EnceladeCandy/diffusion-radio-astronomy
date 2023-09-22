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

    if N_WORKERS>1:
        spw = THIS_WORKER

    else: 
        spw = args.spw

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
    npix = args.npix # To have a finer grid make it bigger than img_size
    img_size = 256 # fixed for this score model
    u_edges, v_edges = grid(pixel_scale = pixel_scale, img_size = img_size)

    def process_binstat(data, weight):
        y = list(zip(data, weight))
        out = np.empty(len(data), dtype = object)
        out[:] = y
        return out

    data_real = process_binstat(vis_re, weight_)
    data_imag = process_binstat(vis_imag, weight_)

    

    def binned_mean(y):
        y = np.stack(y)
        values = y[:, 0]
        w = y[:, 1]
        return np.average(values, weights = w)

    def binned_std(y):
        y = np.stack(y)
        values = y[:, 0]
        w = y[:, 1]
        return np.sqrt(np.cov(values, aweights=w)) #have to use cov since std doesn't accept weights :(
    
    bin_x = u_edges
    bin_y = v_edges
    vis_bin_re, _, _, _ = binned_statistic_2d(vv, uu, values = data_real, bins = (bin_y, bin_x), statistic = binned_mean)
    vis_bin_img, _, _, _ = binned_statistic_2d(vv, uu, values = data_imag, bins = (bin_y, bin_x), statistic = binned_mean)
    std_bin_re, _, _, _ = binned_statistic_2d(vv, uu, values = data_real, bins = (bin_y, bin_x), statistic = binned_std)
    std_bin_img, _, _, _ = binned_statistic_2d(vv, uu, values = data_imag, bins = (bin_y, bin_x), statistic = binned_std)
    counts, _, _, _ = binned_statistic_2d(vv, uu, values = weight_, bins = (bin_y, bin_x), statistic = "count")


    # From object type to float
    vis_bin_re = vis_bin_re.astype(float)
    vis_bin_img = vis_bin_img.astype(float)
    std_bin_re = std_bin_re.astype(float) 
    std_bin_img = std_bin_img.astype(float)
    counts = counts.astype(float)

    # i.e. The sampling function where there is data in the uv plane
    mask = counts>0 

    # binned_stat outputs nans, we put everything to zero instead
    vis_bin_re[~mask] = 0.
    vis_bin_img[~mask] = 0.
    std_bin_re[~mask] = 0.
    std_bin_img[~mask] = 0.
    counts[~mask] = 0.

    std_bin_re /= (counts + 1)**0.5
    std_bin_img /= (counts + 1)**0.5

    # Computing the dirty image: 
    vis_grid = np.fft.fftshift(vis_bin_re + 1j * vis_bin_img)
    dirty_image = npix**2 * flip(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(vis_grid))))

    # For the inference, fftshift + flatten everything for the fft
    vis_gridded_re = np.fft.fftshift(vis_bin_re).flatten()
    vis_gridded_img = np.fft.fftshift(vis_bin_img).flatten()
    std_gridded_re = np.fft.fftshift(std_bin_re).flatten()
    std_gridded_img = np.fft.fftshift(std_bin_re).flatten()
    S_grid = np.fft.fftshift(mask).flatten()

    # Numpy to torch: 
    S_cat = np.concatenate([S_grid, S_grid])
    vis_gridded = np.concatenate([vis_gridded_re, vis_gridded_img])
    std_gridded = np.concatenate([std_gridded_re, std_gridded_img])

    S = torch.tensor(S_cat).to(device)
    y = torch.tensor(vis_gridded, device = S.device)[S].to(device) * img_size
    sigma_y = torch.tensor(std_gridded, device = S.device)[S].to(device) * img_size 


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

    def sigma(t): 
        return sigma_min * (sigma_max/sigma_min) ** t

    def model(x, t):
        x = x.reshape(img_size, img_size) # for the FFT 
        x = link_function(x) # map from model unit space to real unit space
        # Padding: 
        pad_size = int((npix - img_size)/2)
        #x = torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size)) 
        x = noise_padding(x, pad = pad_size, sigma = sigma(t))
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
        y_hat = model(x, t)
        var = sigma(t) **2 / 2  + sigma_y**2
        log_prob = -0.5 * torch.sum((y - y_hat)**2 / var)
        return log_prob


    # GIVE THE GOOD COVARIANCE MATRIX
    def score_likelihood(y, x, t, sigma_y): 
        x = x.flatten(start_dim = 1) 
        return vmap(grad(lambda x, t: log_likelihood(y, x, t, sigma_y)), randomness = "different")(x, t)


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
        folder_dir = f"../../results/noise_pad/euler/spw_{spw}"
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
        folder_dir = f"../../results/noise_pad/pc/spw_{spw}"
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        torch.save(samples_tot, folder_dir + f"/{num_samples}samples.pt")

    else: 
        raise ValueError("The sampler specified is not implemented or does not exist. Choose between 'euler' and 'pc'")

    np.save("dirty_image.npy", dirty_image)

    

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
    parser.add_argument("--spw",                required = False,   default = 10,   type = int,     help = "Spectral window to select from the measurement set"  )
    args = parser.parse_args()
    main(args) 