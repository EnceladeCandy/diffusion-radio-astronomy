import torch 
from astropy.io import fits
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_probes_g_channel(img, inv_link = False):  # channel 0
        img = torch.clamp(img, 0, 1.48)
        
        if inv_link:
            img = 2 * img / 1.48 - 1.
        return img

def link_function(x):
    return (x + 1)/2

def resize(x, target_size=64):
        x_size = x.shape[-1] 
        start = int((x_size-target_size)/2)
        end = start + target_size
        x = x[start:end, start:end]
        return x

def probes_64(dataset, idx):
    """
    dataset = probes 
    idx must be a tensor of size (256, 256) (green channel of the probes dataset)
    """ 
    img = torch.tensor(dataset[idx, ..., 1])
    img = preprocess_probes_g_channel(img) 
    img = F.avg_pool2d(img[None, None, ...], (4, 4))[0, 0].to(device) # Need for the image to be (1, 1, 64, 64)
    return img

def probes_256(dataset, idx): 
    img = torch.tensor(dataset[idx, ..., 1])
    img = preprocess_probes_g_channel(img, inv_link = False)

def fits_to_tensor(file): 
    with fits.open(file) as hdul: 
        header = hdul[0].header
        data = torch.tensor((hdul[0].data).astype(np.float32))[0,0, ...].to(device)
    return header, data