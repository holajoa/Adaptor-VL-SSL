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
export DATASET="rsna"
export PROJECT_POSTFIX="_batchsize8"
for DATA_PCT in 0.01
do
    for VISION_MODEL in "dinov2-b" "dinov2-s" "resnet-ae"
    do
        for TEXT_MODEL in "clinicalbert" "bert" "biobert" "pubmedbert" "cxrbert"
        do
            echo ${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}
            python ./segment.py --dataset $DATASET --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 8 --data_pct $DATA_PCT --num_workers 1 --num_train_epochs 200 --seed 1117 --lr 5e-4 --weight_decay 1e-4 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --postfix v2 --pretrain_wandb_project_name adaptor_pretrain_2_layers --wandb --project_name adaptor_segment_2_layers${PROJECT_POSTFIX} --check_val_every_n_epochs 5 
            wandb artifact cache cleanup 1GB
        done
    done
done
/usr/bin/nvidia-smi
uptime
