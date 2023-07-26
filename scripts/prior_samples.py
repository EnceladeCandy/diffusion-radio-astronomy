import torch 
from score_models import ScoreModel, NCSNpp
device = "cuda" if torch.cuda.is_available() else "cpu"
score_model = ScoreModel(checkpoints_directory="/home/noedia/projects/rrg-lplevass/data/score_models/ncsnpp_probes_g_64_230604024652")
x = score_model.sample(100, shape = (1, 64, 64), steps = 500)
torch.save(x[:,0,...], "../../prior.pt")