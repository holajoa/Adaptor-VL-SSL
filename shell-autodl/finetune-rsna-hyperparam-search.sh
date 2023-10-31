#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/adaptor-thesis/slurm_outputs/finetune-sweep-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export WANDB_DIR=/vol/bitbucket/jq619/
export WANDB_DATA_DIR=/vol/bitbucket/jq619/wandb/
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/adaptor-thesis/trained_models/clf_sweep"
export DATASET="rsna"
export DATA_PCT=0.01
export VISION_MODEL="dinov2-s"
export TEXT_MODEL="clinicalbert"
for BATCH_SIZE in 512 256 128
do
    for LR in 5e-5 1e-4 5e-4
    do
        for WD in 1e-4
        do
            python ./finetune.py --dataset $DATASET --vision_model $VISION_MODEL --text_model $TEXT_MODEL --data_pct $DATA_PCT --num_workers 1 --num_layers 2 --num_train_epochs 100 --seed 42 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT} --wandb --project_name "finetune-sweep_${VISION_MODEL}_${TEXT_MODEL}_${DATASET}_${DATA_PCT}" --batch_size $BATCH_SIZE --lr $LR --weight_decay $WD --postfix v2 --pretrain_wandb_project_name adaptor_pretrain_2_layers --wandb --check_val_every_n_epochs 2
            wandb artifact cache cleanup 1GB
        done
    done
done
/usr/bin/nvidia-smi
uptime
