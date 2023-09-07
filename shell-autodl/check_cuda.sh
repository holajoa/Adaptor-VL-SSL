#!/bin/bash
#SBATCH  --gpus-per-node=2
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
python -m torch.utils.collect_env
/usr/bin/nvidia-smi
uptime
