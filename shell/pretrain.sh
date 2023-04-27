#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
SAVED_EMBEDDINGS_DIR="/vol/bitbucket/jq619/individual-project/saved_embeddings"
# python3 ./scripts/pretrain.py --vision_model_type timm --vision_pretrained swin_base_patch4_window7_224 --text_pretrained "./weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint" --batch_size 32 --data_pct 0.01 --num_hidden_layers 1 --seed 42 --text_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/text_embeds/ClinicalBert --image_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/image_embeds/Swin-Base --num_of_batches 100
python3 ./pretrain.py --vision_model_type ae --vision_pretrained 101-elastic --text_pretrained "dmis-lab/biobert-v1.1" --batch_size 256 --data_pct 0.01 --num_hidden_layers 1 --num_of_batches -1 --num_train_epochs 10 --seed 42 --text_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/text_embeds/BioBERT --image_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/image_embeds/ResNetAE --output_dir ./results/BioBERT_ResNetAE
/usr/bin/nvidia-smi
uptime