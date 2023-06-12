#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/individual-project/slurm_outputs/finetune-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export WANDB_DIR=/vol/bitbucket/jq619/
export WANDB_DATA_DIR=/vol/bitbucket/jq619/wandb/
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/individual-project/trained_models/clf"
export DATASET="covidx"

for VISION_MODEL in "dinov2-b" "dinov2-s" "resnet-ae" 
do
    for TEXT_MODEL in "clinicalbert" "bert" "biobert" "pubmedbert" "cxrbert"
    do
        for DATA_PCT in "0.01" "0.1" "1.0"
            do
            python ./finetune.py --dataset $DATASET --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 64 --data_pct $DATA_PCT --num_workers 1 --num_layers 1 --num_train_epochs 50 --seed 42 --lr 1e-4 --weight_decay 1e-2 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --wandb
            wandb artifact cache cleanup 1GB
        done
    done
done
/usr/bin/nvidia-smi
uptime