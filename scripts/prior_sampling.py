import torch 
import h5py
device = "cuda" if torch.cuda.is_available() else "cpu"
dir_galaxies = "/home/noedia/projects/rrg-lplevass/data/probes.h5"
hdf = h5py.File(dir_galaxies, "r")
hdf.keys()
dataset = hdf['galaxies']

samples = torch.tensor(dataset[:200]).to(device)
torch.save(samples, "../true_prior.pt")