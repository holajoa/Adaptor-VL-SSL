#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
SAVED_EMBEDDINGS_DIR="/vol/bitbucket/jq619/individual-project/saved_embeddings"
python3 ./pretrain_from_scratch.py --vision_model_type ae --vision_pretrained 101-elastic --text_pretrained "dmis-lab/biobert-v1.1" --batch_size 128 --data_pct 1.0 --num_hidden_layers 1 --num_train_epochs 10 --seed 42 --output_dir ./results/BioBERT_ResNetAE --num_workers 8 --lr 5e-5
/usr/bin/nvidia-smi
uptime