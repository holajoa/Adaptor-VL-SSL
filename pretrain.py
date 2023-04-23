import torch 
import torch.nn as nn
from typing import List, Union, Tuple, Dict, Optional

import numpy as np

from transformers import AutoTokenizer
from transformers import BertModel, AutoModel, ViTImageProcessor

import torchxrayvision as xrv
from adaptor import Adaptor
from utils import load_timm_model, freeze_encoder

from image_processor import ae_image_processor, timm_image_processor

import skimage

from transformers import TrainingArguments, Trainer

from mgca.datasets.pretrain_dataset import MultimodalPretrainingDataset, multimodal_collate_fn
from mgca.datasets.data_module import DataModule
from mgca.datasets.classification_dataset import MIMICImageDataset

from mgca.datasets.transforms import DataTransforms

def multimodal_collator(*args, **kwargs):
    d = multimodal_collate_fn(*args, **kwargs)
    d['input_ids'] = d.pop('caption_ids')
    d['pixel_values'] = d.pop('imgs')
    return d

seed = 1117
batch_size = 48
num_workers = 16
data_pct = 0.01
crop_size = 224

datamodule = DataModule(
    dataset=MultimodalPretrainingDataset,
    collate_fn=None, 
    transforms=DataTransforms, 
    data_pct=data_pct, 
    batch_size=batch_size, 
    num_workers=num_workers,
    crop_size=224, 
)

train_dataset = MultimodalPretrainingDataset(
    split='train', 
    transform=DataTransforms(True, crop_size), 
    data_pct=data_pct
)

val_dataset = MultimodalPretrainingDataset(
    split='valid', 
    transform=DataTransforms(False, crop_size), 
    data_pct=data_pct
)

text_pretrained_available = [
    "bert-base-uncased", 
    "dmis-lab/biobert-v1.1", 
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
    "microsoft/BiomedVLP-CXR-BERT-general", 
]

### Load vision model
# vision_model = xrv.autoencoders.ResNetAE(weights="101-elastic")
vision_model = load_timm_model('swin_base_patch4_window7_224', pretrained=True, retain_head=False)

### Load text model
# text_pretrained = "microsoft/BiomedVLP-CXR-BERT-general"
text_pretrained = "./weights/ClinicalBERT_checkpoint/ClinicalBERT_pretraining_pytorch_checkpoint"
text_model = BertModel.from_pretrained(text_pretrained)

tokenizer = AutoTokenizer.from_pretrained(text_pretrained)
image_processor = lambda x: ViTImageProcessor()(x, return_tensors="pt", return_dict=True)

### Load sample input
img_path = 'sample.jpeg'
img = skimage.io.imread(img_path)
imgs = np.stack([img, img])
vision_inputs = image_processor(imgs)

### Define model
model = Adaptor(
    text_model=text_model,
    vision_model=vision_model,
    vision_model_type='timm', 
    vision_output_dim=1024,
    projection_dim=768,
)

### Obtain inputs
vision_inputs = image_processor(imgs)
text_inputs = tokenizer(
    text=["Nodule", "Lung Lesion"], 
    return_tensors="pt", padding=True, 
)
inputs = {**vision_inputs, **text_inputs}

### Forward to get output
outputs = model(**inputs, return_dict=True, return_loss=True)

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
