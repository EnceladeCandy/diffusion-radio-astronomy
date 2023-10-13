#!/bin/bash
#SBATCH --tasks=1
#SBATCH --array=0-200%100
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=40G         # memory per node
#SBATCH --time=00-02:59   # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Posterior_probes_truemodel
module load python
cd $HOME/projects/rrg-lplevass/noedia/diffusion-radio-astronomy/scripts
source $HOME/diffusion/bin/activate

python $HOME/projects/rrg-lplevass/noedia/diffusion-radio-astronomy/scripts/inference_sim.py \
    --sigma_likelihood=1e-2\
    --sampling_function=$HOME/projects/rrg-lplevass/data/sampling_function3.npy \
    --prior=$HOME/projects/rrg-lplevass/data/score_models/ncsnpp_vp_skirt_y_64_230813225149 \
    --pad=96\
    --model_pixels=64\
    --sampler=euler\
    --num_pred=4000\
    --num_samples=300\
    --batch_size=300\
    --results_dir=/home/noedia/scratch/tarp_samples/ \
    --experiment_name=vp_skirt
