#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=
SAVED_EMBEDDINGS_DIR="/vol/bitbucket/jq619/individual-project/saved_embeddings"
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --vision_model_type ae --vision_pretrained 101-elastic --text_pretrained "./weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint" --batch_size 128 --data_pct 1.0 --seed 42 --text_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/text_embeds/ClinicalBERT --image_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/image_embeds/ResNetAE --num_workers 0 --force_rebuild_dataset
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --vision_model_type timm --vision_pretrained swin_base_patch4_window7_224 --text_pretrained dmis-lab/biobert-v1.1 --batch_size 128 --data_pct 1.0 --seed 42 --text_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/text_embeds/BioBERT --image_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/image_embeds/Swin_Base --num_workers 0 --force_rebuild_dataset
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --text_pretrained microsoft/BiomedVLP-CXR-BERT-general --batch_size 128 --data_pct 1.0 --seed 42 --text_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/text_embeds/CXR-BERT --num_workers 0 --force_rebuild_dataset
python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --vision_model_type ae --vision_pretrained 101-elastic --batch_size 64 --data_pct 1.0 --seed 42 --image_embeds_raw_dir $SAVED_EMBEDDINGS_DIR/image_embeds/ResNetAE --num_workers 8 # --force_rebuild_dataset
/usr/bin/nvidia-smi
uptime