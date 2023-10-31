#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
<<<<<<< HEAD
#SBATCH --output=/vol/bitbucket/jq619/individual-project/slurm_outputs/sweep-%j.out
=======
#SBATCH --output=/vol/bitbucket/jq619/adaptor-thesis/slurm_outputs/sweep-%j.out
>>>>>>> 1c83f5c387ded7b17e35c4b09c8401035e3ef837
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export WANDB_DIR=/vol/bitbucket/jq619/
export WANDB_DATA_DIR=/vol/bitbucket/jq619/wandb/
export WANDB__SERVICE_WAIT=300
<<<<<<< HEAD
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/individual-project/trained_models/clf_sweep"
=======
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/adaptor-thesis/trained_models/clf_sweep"
>>>>>>> 1c83f5c387ded7b17e35c4b09c8401035e3ef837
export DATASET="rsna"
export DATA_PCT=1.0
export VISION_MODEL="dinov2-b"
export TEXT_MODEL="clinicalbert"
srun wandb agent --count 1 holajoa/finetune-sweep_dinov2-b_clinicalbert_rsna_1.0_rsna/7j9g9z60
wandb artifact cache cleanup 1GB
/usr/bin/nvidia-smi
uptime
