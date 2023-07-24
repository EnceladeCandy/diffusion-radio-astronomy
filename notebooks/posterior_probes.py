import torch 
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch.func import vmap, grad
from tqdm import tqdm
from torch.distributions import MultivariateNormal
import h5py
from score_models import ScoreModel, NCSNpp
import json

plt.style.use("dark_background")
sys.path.append("..\\")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Importing the models hparams and weights
file = open("/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652/model_hparams.json")
model_hparams = json.load(file)
sigma_min, sigma_max = model_hparams["sigma_min"], model_hparams["sigma_max"]

torch.manual_seed(2)
score_model = ScoreModel(checkpoints_directory="/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652")
x = score_model.sample(1, shape = (1, 64, 64), steps = 500)[0,0]
img_size = x.shape[-1]
plt.imshow(x.cpu(), cmap = "hot")
plt.savefig("../ground_truth.jpg")