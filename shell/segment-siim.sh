#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/individual-project/slurm_outputs/segment-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export WANDB_DIR=/vol/bitbucket/jq619/
export WANDB_DATA_DIR=/vol/bitbucket/jq619/wandb/
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/individual-project/trained_models/segment"
export DATASET="siim"
# export PROJECT_POSTFIX="_update_loss"
# export SEED=1024
export VERSION=0
export SEED=7
export DATA_PCT=0.01
for TEXT_MODEL in "biobert" # "clinicalbert" "cxrbert" "bert" "pubmedbert" 
do
    for VISION_MODEL in  "dinov2-b" # "resnet-ae" "dinov2-s" 
    do
        echo ${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}
        python ./segment.py --n_gpus 2 --seed $SEED --dataset $DATASET --crop_size 896 --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 4 --data_pct $DATA_PCT --num_workers 8 --num_train_epochs 100 --lr 5e-4 --weight_decay 1e-4 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --wandb --postfix v${VERSION} --pretrain_wandb_project_name adaptor_pretrain_v${VERSION} --wandb --project_name adaptor_segment_v${VERSION} --check_val_every_n_epochs 5 --patience_epochs 20 | tee logs/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}.txt
        wandb artifact cache cleanup 1GB
    done
done
/usr/bin/nvidia-smi
uptime
