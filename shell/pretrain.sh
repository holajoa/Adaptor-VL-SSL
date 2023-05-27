#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/individual-project/slurm_outputs/pretrain-from-embeds-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/individual-project/trained_models/pretrain"
export VISION_MODEL="swin-base" 
export TEXT_MODEL="biobert"
# export TEXT_MODEL="bert"
# export TEXT_MODEL="clinicalbert"
python ./pretrain.py --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 128 --data_pct 1.0 --num_workers 1 --num_hidden_layers 1 --num_train_epochs 30 --seed 42 --lr 1e-4 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}
/usr/bin/nvidia-smi
uptime