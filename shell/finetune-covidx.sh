#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/adaptor-thesis/slurm_outputs/finetune-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export WANDB_DIR=/vol/bitbucket/jq619/
export WANDB_DATA_DIR=/vol/bitbucket/jq619/wandb/
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/adaptor-thesis/trained_models/clf"
export DATASET="covidx"
export VERSION=3
for DATA_PCT in 0.01 0.1 1
do
    for TEXT_MODEL in "clinicalbert" "pubmedbert" "bert" "biobert" "cxrbert" 
    do
        for VISION_MODEL in "dinov2-b"  "resnet-ae" "dinov2-s" 
        do
            python ./finetune.py --seed 6 --n_gpus 2 --dataset $DATASET --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 128 --data_pct $DATA_PCT --num_workers 8 --num_train_epochs 50 --lr 5e-4 --weight_decay 0.05 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --postfix v${VERSION} --pretrain_wandb_project_name adaptor_pretrain_v${VERSION} --wandb --project_name adaptor_finetune_v${VERSION}${POSTFIX} --check_val_every_n_epochs 1 --patience_epochs 5  | tee logs/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}.txt
            wandb artifact cache cleanup 1GB
        done
    done
done
/usr/bin/nvidia-smi
uptime8