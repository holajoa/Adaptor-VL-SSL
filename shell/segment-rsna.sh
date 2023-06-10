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
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/individual-project/trained_models/segment"
export DATASET="rsna"
for DATA_PCT in 0.01 0.1 1.0
do
    for VISION_MODEL in "resnet-ae" 
    do
        for TEXT_MODEL in "clinicalbert" "bert" "biobert" "pubmedbert" "cxrbert"
        do
            python ./segmentation.py --dataset $DATASET --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 16 --data_pct $DATA_PCT --num_workers 1 --num_train_epochs 50 --seed 42 --lr 2e-5 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --wandb
            wandb artifact cache cleanup 1GB
        done
    done
done
/usr/bin/nvidia-smi
uptime