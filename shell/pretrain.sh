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
export VISION_MODEL="dinov2-s" 
export TEXT_MODEL="biobert"

for VISION_MODEL in "dinov2-b" "dinov2-s" "resnet-ae"
do
    for TEXT_MODEL in "biobert" "bert" "clinicalbert" "cxrbert" "pubmedbert"
    do
        python ./pretrain.py --vision_model $VISION_MODEL --text_model $TEXT_MODEL --batch_size 128 --data_pct 1.0 --num_workers 1 --num_train_epochs 50 --seed 42 --lr 1e-4 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL} --wandb
        wandb artifact cache cleanup 1GB
    done
done
/usr/bin/nvidia-smi
uptime