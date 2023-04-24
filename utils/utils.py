import timm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from models.adaptor import Adaptor

from tqdm import tqdm
import os

from typing import List, Union, Tuple, Dict, Optional


def load_timm_model(model_name='swin_base_patch4_window7_224', retain_head=False, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    if not retain_head:
        return nn.Sequential(*list(model.children())[:-2])
    return model

def freeze_encoder(model:Adaptor):
    for encoder in [model.text_model, model.vision_model]:
        for param in encoder.parameters():
            param.requires_grad = False

def get_dataloader(
    dataset, 
    batch_size,
    num_workers,
    collate_fn,
):
    return DataLoader(
        dataset, 
        pin_memory=True, 
        drop_last=True,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

def get_image_embeds_raw(
    dataloader,
    vision_model,
    vision_model_type='huggingface', 
    save_path='./weights/image_embeds',
    model_name='',
    split='train', 
):    
    if model_name:
        save_path = os.path.join(save_path, model_name)
    save_path = os.path.join(save_path, split)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for batch_idx, inputs in enumerate(tqdm(dataloader)):
        if vision_model_type == 'huggingface':
            vision_outputs = vision_model(
                inputs.pixel_values,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            image_embeds_raw = vision_outputs.last_hidden_state
        elif vision_model_type == 'timm':
            image_embeds_raw = vision_model(inputs['pixel_values'])
        elif vision_model_type == 'ae':
            vision_outputs = vision_model(inputs)
            image_embeds_raw = torch.flatten(vision_outputs['z'], start_dim=2).permute((0, 2, 1))
        else: 
            logging.error(f'{vision_model_type} is not supported.')
        pt_filename = f'{split}_{batch_idx}.pt'
        torch.save(image_embeds_raw, os.path.join(save_path, pt_filename))
        

def get_text_embeds_raw(
    dataloader, 
    text_model,
    save_path='./weights/text_embeds',
    removed_arguments=['cap_lens', 'pixel_values', 'path'], 
    model_name='', 
    split='train', 
):  
    if model_name:
        save_path = os.path.join(save_path, model_name)
    save_path = os.path.join(save_path, split)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for batch_idx, inputs in enumerate(tqdm(dataloader)):
        [inputs.pop(key, None) for key in removed_arguments]
        text_outputs = text_model(
            **inputs, 
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        text_embeds_raw = text_outputs.last_hidden_state
        pt_filename = f'{split}_{batch_idx}.pt'
        torch.save(text_embeds_raw, os.path.join(save_path, pt_filename))
    