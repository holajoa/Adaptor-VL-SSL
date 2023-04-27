import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

import os
from tqdm import tqdm

from dataset.dataset import (
    # MultimodalPretrainedEmbeddingsDatasetLoader, 
    MultimodalPretrainedEmbeddingsDataset, 
)

from models.adaptor import Adaptor, AdaptorTrainer, AdaptorTrainingArguments, ExternalLoggingCallback
from models.configurations import (
    TEXT_PRETRAINED_AVAILABLE,
    VISION_PRETRAINED_AVAILABLE,
    VISION_MODEL_TYPE_2_DATA_TRANSFORM,
    VISION_MODEL_TYPE_2_VISION_OUTPUT_DIM, 
)
from utils.utils import load_timm_model, freeze_encoder
from utils.model_utils import load_vision_model
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import TrainingArguments

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from datasets import Dataset

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

parser.add_argument('--seed', type=int, default=1117)

parser.add_argument('--output_dir', type=str, default='./results', help='path to save model')

args = parser.parse_args()


from time import gmtime, strftime
log_fn = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
logging.basicConfig(filename=f'{log_fn}.log', encoding='utf-8', level=logging.INFO)

### Load dataset
# train_dataset_loader = MultimodalPretrainedEmbeddingsDatasetLoader(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
#                                                                    split='train', num_of_batches=args.num_of_batches,)
# train_dataset = train_dataset_loader.load_data()

# val_dataset_loader = MultimodalPretrainedEmbeddingsDatasetLoader(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
#                                                                  split='valid', num_of_batches=args.num_of_batches,)
# val_dataset = val_dataset_loader.load_data()

train_dataset = MultimodalPretrainedEmbeddingsDataset(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
                                                      split='train', num_of_batches=args.num_of_batches,)
val_dataset = MultimodalPretrainedEmbeddingsDataset(args.text_embeds_raw_dir, args.image_embeds_raw_dir, 
                                                    split='valid', num_of_batches=args.num_of_batches,)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model = nn.DataParallel(model)
model.to(device)


### Training
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-5)

arguments = AdaptorTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size, 
    per_device_eval_batch_size=args.batch_size,  
    num_train_epochs=args.num_train_epochs,
    dataloader_num_workers=args.num_workers,
    logging_steps=20, 
    save_strategy="epoch",
    learning_rate=args.lr, 
    seed=args.seed, 
    push_to_hub=False, 
    dataloader_pin_memory=True, 
    num_of_batches=args.num_of_batches,
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
trainer.train()
