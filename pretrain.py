import torch
import torch.nn as nn

from dataset.dataset import MultimodalPretrainedEmbeddingsIterableDataset

from models.adaptor import Adaptor, AdaptorTrainer, AdaptorTrainingArguments, ExternalLoggingCallback
from models.configurations import (
    TEXT_PRETRAINED_AVAILABLE,
    VISION_PRETRAINED_AVAILABLE,
    VISION_MODEL_TYPE_2_DATA_TRANSFORM,
    VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM, 
)
from utils.utils import load_timm_model, freeze_encoder
from utils.model_utils import load_vision_model
from utils.dataset_utils import torch2huggingface_dataset
from transformers import BertModel

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import argparse
import logging

# SAVED_EMBEDDINGS_DIR = '/vol/bitbucket/jq619/individual-project/saved_embeddings'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vision_pretrained', type=str, help='Choose from [101-elastic, swin_base_patch4_window7_224]')
parser.add_argument('--vision_model_type', type=str, help='Choose from [timm, ae, huggingface]')
parser.add_argument('--text_pretrained', type=str, 
                    help='Choose from [bert-base-uncased, dmis-lab/biobert-v1.1, '
                    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, '
                    'microsoft/BiomedVLP-CXR-BERT-general, '
                    './weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint]')

parser.add_argument('--text_embeds_raw_dir', type=str, help='path to raw text embeddings')
parser.add_argument('--image_embeds_raw_dir', type=str, help='path to raw image embeddings')
parser.add_argument('--num_of_batches', type=int, default=100, help='number of batches to use for training')

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--force_rebuild_dataset', action='store_true', help='Whether to force rebuild dataset, if not can load pickled file if available')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--data_pct', type=float, default=0.01, help='percentage of data to use')
parser.add_argument('--crop_size', type=int, default=224)

parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of transformer layers to use in adaptor')
parser.add_argument('--projection_dim', type=int, default=768, help='dimension of projection head')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_train_epochs', type=int, default=1)

parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

parser.add_argument('--seed', type=int, default=1117)

parser.add_argument('--output_dir', type=str, default='./results', help='path to save model')

args = parser.parse_args()

torch.manual_seed(args.seed)

from time import gmtime, strftime
log_fn = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
logging.basicConfig(filename=f'{log_fn}.log', encoding='utf-8', level=logging.INFO)

num_of_gpus = torch.cuda.device_count()
logging.info(f"Number of available GPUs = {num_of_gpus}: "
             f"{', '.join([torch.cuda.get_device_properties(i).name for i in range(num_of_gpus)])}.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ### Enable distributed training
# if device != torch.device("cpu"):
#     def ddp_setup(rank:int, world_size:int):
#         """_summary_

#         Args:
#             rank (int): unique identifier assigned to each process (0 ~ world_size-1)
#             world_size (int): Total number of processes in a group
#         """
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '12355'
#         init_process_group(backend="nccl", rank=rank, world_size=world_size)
    

### Load vision model (not used in training actually, just for model definition)
if args.vision_pretrained in VISION_PRETRAINED_AVAILABLE.keys():
    assert VISION_PRETRAINED_AVAILABLE[args.vision_pretrained] == args.vision_model_type, \
        'Vision model type does not match pretrained model'
vision_model = load_vision_model(args.vision_model_type, args.vision_pretrained)

### Load text model (not used in training actually, just for model definition)
text_model = BertModel.from_pretrained(args.text_pretrained)

### Define model
add_cls_token = args.vision_model_type == 'ae'
vision_output_dim = VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM[args.vision_model_type]
model = Adaptor(
    text_model=text_model,
    vision_model=vision_model,
    vision_model_type=args.vision_model_type, 
    vision_output_dim=vision_output_dim,
    projection_dim=args.projection_dim,
    num_hidden_layers=args.num_hidden_layers, 
    add_cls_token=add_cls_token,
)
freeze_encoder(model)  # freeze encoder
# model = nn.DataParallel(model)
# model = DDP(model, device_ids=[0, 1])
model.to(device)


### Load dataset
dataset_device = 'cpu' if args.num_workers > 0 else device
train_dataset = MultimodalPretrainedEmbeddingsIterableDataset(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
                                                              split='train', num_of_batches=args.num_of_batches, 
                                                              device=dataset_device)
args.max_steps = len(train_dataset) // args.batch_size * args.num_train_epochs

train_dataset = torch2huggingface_dataset(train_dataset, streaming=True)
train_dataset.with_format('torch')

val_dataset = MultimodalPretrainedEmbeddingsIterableDataset(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
                                                            split='valid', num_of_batches=args.num_of_batches, 
                                                            device=dataset_device)
val_dataset = torch2huggingface_dataset(val_dataset, streaming=True)
val_dataset.with_format('torch')

### Training
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2, eta_min=args.lr/10)

arguments = AdaptorTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size, 
    per_device_eval_batch_size=args.batch_size,  
    num_train_epochs=args.num_train_epochs,
    dataloader_num_workers=args.num_workers,
    dataloader_drop_last=True,
    logging_steps=20, 
    evaluation_strategy='steps', 
    eval_steps=100, 
    save_strategy="epoch",
    # learning_rate=args.lr, 
    seed=args.seed, 
    push_to_hub=False, 
    dataloader_pin_memory=False, # True, 
    num_of_batches=args.num_of_batches,
    max_steps=args.max_steps,
)

trainer = AdaptorTrainer(
    model=model, 
    args=arguments,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,  
    data_collator=None, 
    optimizers=(optimizer, lr_schedule),
    callbacks=[ExternalLoggingCallback()],
)

# torch.multiprocessing.set_start_method('spawn')
trainer.train()
