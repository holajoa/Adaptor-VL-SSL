#!/bin/bash
#SBATCH  --gpus-per-node=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jq619
#SBATCH --output=/vol/bitbucket/jq619/individual-project/slurm_outputs/get-embeds-%j.out
export PATH=/vol/bitbucket/jq619/idv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=
python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --vision_model resnet-ae --text_model bert --batch_size 64 --data_pct 1.0 --seed 42  --num_workers 8 --force_rebuild_dataset
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --vision_model swin-base --text_model biobert --batch_size 64 --data_pct 1.0 --seed 42  --num_workers 8 --force_rebuild_dataset
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --text_model pubmedbert --batch_size 64 --data_pct 1.0 --seed 42  --num_workers 8 --force_rebuild_dataset
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --text_model cxrbert --batch_size 64 --data_pct 1.0 --seed 42  --num_workers 8 --force_rebuild_dataset
python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --text_model clinicalbert --batch_size 64 --data_pct 1.0 --seed 42 --num_workers 8 --force_rebuild_dataset
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --vision_model dinov2-s --batch_size 64 --data_pct 1.0 --seed 42  --num_workers 8 --force_rebuild_dataset
# python3 -m torch.distributed.launch --nproc_per_node 2 ./get_pretrained_embeddings.py --vision_model dinov2-b --batch_size 64 --data_pct 1.0 --seed 42  --num_workers 8 --force_rebuild_dataset
/usr/bin/nvidia-smi
uptime