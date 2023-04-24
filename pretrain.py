import torch 
import torch.nn as nn
from typing import List, Union, Tuple, Dict, Optional

import numpy as np

from transformers import AutoTokenizer
from transformers import BertModel, AutoModel, ViTImageProcessor

import torchxrayvision as xrv
from models.adaptor import Adaptor
from utils.utils import load_timm_model, freeze_encoder
from utils.dataset_utils import ae_image_processor, pickle_dataset, VISION_MODEL_TYPE_2_DATA_TRANSFORM


from transformers import TrainingArguments, Trainer
from datasets.dataset import multimodal_collator
from mgca.datasets.transforms import DataTransforms

from pathlib import Path


seed = 1117
batch_size = 32
num_workers = 16
data_pct = 0.01
crop_size = 224
num_hidden_layers = 1


text_pretrained_available = [
    "bert-base-uncased", 
    "dmis-lab/biobert-v1.1", 
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
    "microsoft/BiomedVLP-CXR-BERT-general", 
]

### Load vision model
vision_model = xrv.autoencoders.ResNetAE(weights="101-elastic")
vision_model_type = 'ae'
# vision_model = load_timm_model('swin_base_patch4_window7_224', pretrained=True, retain_head=False)

### Load text model
# text_pretrained = "microsoft/BiomedVLP-CXR-BERT-general"
text_pretrained = "./weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint"
text_model = BertModel.from_pretrained(text_pretrained)

tokenizer = AutoTokenizer.from_pretrained(text_pretrained)
# if vision_model_type == 'ae':
#     iamge_processor = lambda x: ae_image_processor(x, return_dict=True)
# else:
#     image_processor = lambda x: ViTImageProcessor()(x, return_tensors="pt", return_dict=True)


### Define model
add_cls_token = vision_model_type == 'ae'
vision_output_dim = 512 if vision_model_type == 'ae' else 1024
model = Adaptor(
    text_model=text_model,
    vision_model=vision_model,
    vision_model_type=vision_model_type, 
    vision_output_dim=vision_output_dim,
    projection_dim=768,
    num_hidden_layers=1, 
    add_cls_token=add_cls_token,
)
freeze_encoder(model)


### Load dataset
data_transforms = VISION_MODEL_TYPE_2_DATA_TRANSFORM[vision_model_type]

train_dataset_pkl = f'saved_datasets/train_dataset_{vision_model_type}.pkl'
val_dataset_pkl = f'saved_datasets/val_dataset_{vision_model_type}.pkl'

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

# delattr(train_dataset, 'crop_size')
# delattr(val_dataset, 'crop_size')
# train_dataset.transform.crop_size = crop_size
# val_dataset.transform.crop_size = crop_size

# import pickle
# with open(train_dataset_pkl, "wb") as f:
#     pickle.dump(train_dataset, f, protocol=2)
# with open(val_dataset_pkl, "wb") as f:
#     pickle.dump(val_dataset, f, protocol=2)


### Training
arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,  
    num_train_epochs=1, 
    save_strategy="epoch",
    learning_rate=5e-5, 
    seed=1117, 
    push_to_hub=False, 
)

trainer = Trainer(
    model=model, 
    args=arguments,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    tokenizer=tokenizer, 
    data_collator=multimodal_collator, 
)
trainer.train()
