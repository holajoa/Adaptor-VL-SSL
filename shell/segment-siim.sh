#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/adaptor-thesis/slurm_outputs/segment-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export WANDB_DIR=/vol/bitbucket/jq619/
export WANDB_DATA_DIR=/vol/bitbucket/jq619/wandb/
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/adaptor-thesis/trained_models/segment"
export DATASET="siim"
export SEED=7
export VERSION=3
for DATA_PCT in 0.1 0.01
do
    for TEXT_MODEL in "cxrbert" "pubmedbert" # "bert" "biobert" "clinicalbert" 
    do
        for VISION_MODEL in   "dinov2-b" "dinov2-s" # "resnet-ae"  
        do  
            # source /etc/network_turbo
            echo ${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}
            python ./segment.py --original_dice_loss --n_gpus 2 --seed $SEED --dataset $DATASET --crop_size 896 --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 4 --data_pct $DATA_PCT --num_workers 8 --num_train_epochs 100 --lr 5e-4 --weight_decay 1e-4 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --wandb --postfix v${VERSION} --pretrain_wandb_project_name adaptor_pretrain_v${VERSION} --wandb --project_name adaptor_segment_v${VERSION} --check_val_every_n_epochs 3 --patience_epochs 10 | tee logs/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}.txt
            wandb artifact cache cleanup 1GB
        done
    done
done
/usr/bin/nvidia-smi
uptime