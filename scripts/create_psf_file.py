import torch
from astropy.io import fits
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
def resize(x, target_size=64):
        x_size = x.shape[-1] 
        start = int((x_size-target_size)/2)
        end = start + target_size
        x = x[start:end, start:end]
        return x

# Path to your .fits file
fits_file_path = '../../psf_256.fits'

# Open the .fits file using Astropy
with fits.open(fits_file_path) as hdul:
    # Get the header and data from the primary HDU (Extension 0)
    header = hdul[0].header
    psf = torch.tensor((hdul[0].data).astype(np.float32))[0,0, ...].to(device)

psf = resize(psf, target_size = 64)
torch.save(psf, "../psf64.pt")