import torch 
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_probes_g_channel(img, inv_link = False):  # channel 0
        img = torch.clamp(img, 0, 1.48)
        
        if inv_link:
            img = 2 * img / 1.48 - 1.
        return img

def link_function(x):
    return (x + 1)/2

def probes_64(dataset, idx):
    """
    dataset = probes 
    idx must be a tensor of size (256, 256) (green channel of the probes dataset)
    """ 
    img = torch.tensor(dataset[idx, ..., 1])
    img = preprocess_probes_g_channel(img) 
    img = F.avg_pool2d(img[None, None, ...], (4, 4))[0, 0].to(device) # Need for the image to be (1, 1, 64, 64)
    return img