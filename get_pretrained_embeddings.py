import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import List, Union, Tuple, Dict, Optional

import numpy as np

from transformers import AutoTokenizer
from transformers import BertModel, AutoModel, ViTImageProcessor

from models.adaptor import Adaptor
from utils.utils import (
    load_timm_model, freeze_encoder, 
    get_text_embeds_raw, get_image_embeds_raw,
    get_dataloader,
)
# from utils.dataset_utils import ae_image_processor, timm_image_processor

from utils.dataset_utils import (
    pickle_dataset, 
    VISION_MODEL_TYPE_2_DATA_TRANSFORM, 
)

import logging

from mgca.datasets.transforms import DataTransforms
from datasets.dataset import (
    MultimodalPretrainingDatasetForAdaptor, 
    multimodal_collator, 
)

import skimage

seed = 1117
batch_size = 32
num_workers = 16
data_pct = 0.01
crop_size = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')
print(f'Using device: {device}')


# Load pretrained models
vision_model = load_timm_model('swin_base_patch4_window7_224', pretrained=True, retain_head=False)
vision_model_type = 'timm'

text_pretrained = "./weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint"
text_model = BertModel.from_pretrained(text_pretrained)

### Load dataset
data_transforms = VISION_MODEL_TYPE_2_DATA_TRANSFORM[vision_model_type]

train_dataset_pkl = f'saved_datasets/train_dataset_{vision_model_type}.pkl'
val_dataset_pkl = f'saved_datasets/val_dataset_{vision_model_type}.pkl'
test_dataset_pkl = f'saved_datasets/test_dataset_{vision_model_type}.pkl'

train_dataset = pickle_dataset(
    train_dataset_pkl, 
    split='train', 
    transform=data_transforms(True, crop_size), 
    data_pct=data_pct, 
    # force_rebuild=True, 
)
val_dataset = pickle_dataset(
    val_dataset_pkl,
    split='valid',
    transform=data_transforms(False, crop_size),
    data_pct=data_pct, 
    # force_rebuild=True, 
)

test_dataset = pickle_dataset(
    test_dataset_pkl,
    split='test',
    transform=data_transforms(False, crop_size),
    data_pct=data_pct, 
    # force_rebuild=True, 
)

# Get dataloaders
train_dataloader = get_dataloader(
    train_dataset, 
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=multimodal_collator,
)
val_dataloader = get_dataloader(
    val_dataset, 
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=multimodal_collator,
)
test_dataloader = get_dataloader(
    test_dataset, 
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=multimodal_collator,
)

for split, dataloader in zip(['train', 'valid', 'test'], 
                             [train_dataloader, val_dataloader, test_dataloader]):
    get_text_embeds_raw(
        dataloader,
        text_model=text_model,
        save_path='./saved_embeddings/text_embeds',
        model_name='ClinicalBERT',
        split=split,
    )
    get_image_embeds_raw(
        dataloader,
        vision_model=vision_model,
        vision_model_type='timm', 
        save_path='./saved_embeddings/image_embeds',
        model_name='Swin-Base',
        split=split,
    )

