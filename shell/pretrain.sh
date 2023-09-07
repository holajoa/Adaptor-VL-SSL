#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/adaptor-thesis/slurm_outputs/pretrain-from-embeds-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
TERM=vt100 # or TERM=xterm
export SAVED_MODEL_DIR="/vol/bitbucket/jq619/adaptor-thesis/trained_models/pretrain"
export VISION_MODEL="dinov2-s"
export TEXT_MODEL="biobert"

for VISION_MODEL in "dinov2-b" # "resnet-ae" "dinov2-s"
do
    for TEXT_MODEL in  "biobert" "clinicalbert" "cxrbert" "pubmedbert" "bert"
    do
        python ./pretrain.py --vision_model $VISION_MODEL --text_model $TEXT_MODEL --num_layers 2 --batch_size 1024 --data_pct 1.0 --num_workers 4 --num_train_epochs 50 --seed 42 --lr 2e-5 --output_dir $SAVED_MODEL_DIR/${VISION_MODEL}_${TEXT_MODEL}_v${VERSION} --project_name adaptor_pretrain_v${VERSION} --wandb | tee logs/pretrain_${VISION_MODEL}_${TEXT_MODEL}.txt 
        wandb artifact cache cleanup 1MB
    done
done
/usr/bin/nvidia-smi
uptime