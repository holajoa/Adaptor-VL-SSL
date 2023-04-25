#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jq619
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
python3 ./pretrain.py --vision_model_type timm --vision_pretrained swin_base_patch4_window7_224 --text_pretrained "./weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint" --batch_size 32 --data_pct 0.01 --num_hidden_layers 1 --seed 42 
/usr/bin/nvidia-smi
uptime