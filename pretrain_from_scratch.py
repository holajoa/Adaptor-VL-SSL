import torch 
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from typing import List, Union, Tuple, Dict, Optional

import numpy as np

from transformers import AutoTokenizer
from transformers import BertModel

import torchxrayvision as xrv
from models.adaptor import Adaptor, AdaptorTrainer, ExternalLoggingCallback
from models.configurations import (
    TEXT_PRETRAINED_AVAILABLE,
    VISION_PRETRAINED_AVAILABLE,
    VISION_MODEL_TYPE_2_DATA_TRANSFORM,
    VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM, 
)
from utils.utils import load_timm_model, freeze_encoder
from utils.dataset_utils import ae_image_processor, pickle_dataset
from utils.model_utils import load_vision_model

from transformers import TrainingArguments, Trainer
from dataset.dataset import multimodal_collator
import argparse 
import logging


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vision_pretrained', type=str, help='Choose from [101-elastic, swin_base_patch4_window7_224]')
parser.add_argument('--vision_model_type', type=str, help='Choose from [timm, ae, huggingface]')
parser.add_argument('--text_pretrained', type=str, 
                    help='Choose from [bert-base-uncased, dmis-lab/biobert-v1.1, '
                    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, '
                    'microsoft/BiomedVLP-CXR-BERT-general, '
                    './weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint]')
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--force_rebuild_dataset', action='store_true', help='Whether to force rebuild dataset, if not can load pickled file if available')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--data_pct', type=float, default=0.01, help='percentage of data to use')
parser.add_argument('--crop_size', type=int, default=224)

parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of transformer layers to use in adaptor')
parser.add_argument('--projection_dim', type=int, default=768, help='dimension of projection head')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_train_epochs', type=int, default=1)

parser.add_argument('--seed', type=int, default=1117)
parser.add_argument('--output_dir', type=str, default='./results', help='path to save model')

parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')


args = parser.parse_args()

torch.manual_seed(args.seed)


from time import gmtime, strftime
log_fn = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
logging.basicConfig(filename=f'{log_fn}.log', encoding='utf-8', level=logging.INFO)

num_of_gpus = torch.cuda.device_count()
logging.info(f"Number of available GPUs = {num_of_gpus}: "
             f"{', '.join([torch.cuda.get_device_properties(i).name for i in range(num_of_gpus)])}.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Load vision model
if args.vision_pretrained in VISION_PRETRAINED_AVAILABLE.keys():
    assert VISION_PRETRAINED_AVAILABLE[args.vision_pretrained] == args.vision_model_type, \
        'Vision model type does not match pretrained model'
vision_model = load_vision_model(args.vision_model_type, args.vision_pretrained)

### Load text model
text_model = BertModel.from_pretrained(args.text_pretrained)
tokenizer = AutoTokenizer.from_pretrained(args.text_pretrained)

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
model = nn.DataParallel(model)
model.to(device)


### Load dataset
data_transforms = VISION_MODEL_TYPE_2_DATA_TRANSFORM[args.vision_model_type]

train_dataset_pkl = f'saved_datasets/train_dataset_{args.vision_model_type}.pkl'
val_dataset_pkl = f'saved_datasets/val_dataset_{args.vision_model_type}.pkl'

train_dataset = pickle_dataset(
    train_dataset_pkl, 
    split='train', 
    transform=data_transforms(True, args.crop_size), 
    data_pct=args.data_pct, 
    force_rebuild=args.force_rebuild_dataset, 
    tokenizer=tokenizer,
)
val_dataset = pickle_dataset(
    val_dataset_pkl,
    split='valid',
    transform=data_transforms(False, args.crop_size),
    data_pct=args.data_pct, 
    force_rebuild=args.force_rebuild_dataset, 
    tokenizer=tokenizer,
)

### Training
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2, eta_min=args.lr/10)

arguments = TrainingArguments(
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
    dataloader_pin_memory=True, 
)

trainer = AdaptorTrainer(
    model=model, 
    args=arguments,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    tokenizer=tokenizer, 
    data_collator=multimodal_collator, 
    optimizers=(optimizer, lr_schedule),
    callbacks=[ExternalLoggingCallback()],
)
trainer.train()
