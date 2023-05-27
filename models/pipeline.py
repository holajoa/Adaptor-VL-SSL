from typing import List, Union, Tuple, Dict, Optional

import torch 
import torch.nn as nn

from torchmetrics import Accuracy, AUROC

import pytorch_lightning as pl

from transformers import AutoTokenizer
from transformers import BertModel

from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.clip.modeling_clip import CLIPOutput

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.adaptor import Adaptor
from models.configurations import TEXT_PRETRAINED, VISION_PRETRAINED
from utils.model_utils import load_vision_model

import logging


def freeze_encoder(model:pl.LightningModule):
    for encoder in [model.text_model, model.vision_model]:
        for param in encoder.parameters():
            param.requires_grad = False

class AdaptorPipelineBase(pl.LightningModule):
    def __init__(
        self, 
        text_model:str,
        vision_model:str,
        adaptor_ckpt:str,
        lr:float=1e-4,
    ):
        super(AdaptorPipelineBase, self).__init__()
        
        vision_model_config = VISION_PRETRAINED[vision_model]
        vision_pretrained = vision_model_config['pretrained_weight']
        self.vision_model_type = vision_model_config['vision_model_type']
        self.text_pretrained = TEXT_PRETRAINED[text_model]
        
        self.vision_model = load_vision_model(
            vision_model_type=self.vision_model_type, 
            vision_pretrained=vision_pretrained, 
            retain_head=False,
        )
        self.text_model = BertModel.from_pretrained(self.text_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_pretrained)
        self.adaptor = Adaptor.load_from_checkpoint(adaptor_ckpt)
        
        self.lr = lr
        freeze_encoder(self)
        self.save_hyperparameters(ignore=["text_model", "vision_model"])
        
    def forward(
        self,
        pixel_values:torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs, 
    ) -> Union[Tuple[torch.Tensor], CLIPOutput, torch.Tensor]:
        assert pixel_values is not None, "Must pass pixel_values."
        if self.vision_model_type == 'huggingface':
            vision_outputs = self.vision_model(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds_raw = vision_outputs.pooler_output
        elif self.vision_model_type == 'timm':
            image_embeds_raw = self.vision_model(pixel_values)[:, 0, :]
        elif self.vision_model_type == 'ae':
            vision_outputs = self.vision_model(pixel_values)
            image_embeds_raw = torch.flatten(vision_outputs['z'], start_dim=2).permute((0, 2, 1)).mean(1)
        else: 
            logging.ERROR(f'{self.vision_model_type} is not supported.')
        assert len(image_embeds_raw.shape) == 2
        
        outputs = self.adaptor(image_embeds_raw)
        image_embeds = outputs  #.last_hidden_state[:, 0, :]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        return image_embeds  # ignore return_dict and return_loss
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.lr_schedulers().step()
        return loss
    
    def _shared_eval(self, batch, batch_idx, prefix):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(f'{prefix}_loss', loss,  on_epoch=True, logger=True)
        
    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        lr_schedule = CosineAnnealingWarmRestarts(
            optimizer=optimizer, 
            T_0=int(self.training_steps*0.4), 
            T_mult=1, 
            eta_min=1e-8, 
        )
        return {'optimizer':optimizer, 'lr_scheduler':lr_schedule}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    # @staticmethod
    # def training_steps(train_dataset, batch_size, num_train_epochs):
    #     return ceil(len(train_dataset) / batch_size) * num_train_epochs
    
    # @staticmethod
    # def val_steps(val_dataset, batch_size, num_train_epochs):
    #     return ceil(len(val_dataset) / batch_size)
        
    

class AdaptorPipelineWithClassificationHead(AdaptorPipelineBase):
    def __init__(
        self, 
        text_model:str,
        vision_model:str,
        adaptor_ckpt:str,
        num_classes:int, 
        lr:float=1e-4,
    ):
        super(AdaptorPipelineWithClassificationHead, self).__init__(
            text_model=text_model, 
            vision_model=vision_model, 
            adaptor_ckpt=adaptor_ckpt, 
            lr=lr, 
        )
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.adaptor.projection_dim, num_classes)
        self.loss_func = nn.BCELoss(reduction='mean')
        
        if num_classes > 1:
            self.metrics = AUROC(task="multiclass", num_classes=num_classes)
            self.val_metrics = AUROC(task="multiclass", num_classes=num_classes)
        else:
            self.metrics = AUROC(task='binary')
            self.val_metrics = AUROC(task='binary')
        
        self.save_hyperparameters(ignore=["text_model", "vision_model"])
        
    def forward(
        self,
        pixel_values:torch.FloatTensor,
        labels:torch.LongTensor, 
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs, 
    ) -> Union[Tuple[torch.Tensor], CLIPOutput, torch.Tensor]:
        adaptor_output = super(AdaptorPipelineWithClassificationHead, self).forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict, 
            **kwargs, 
        )
        if isinstance(adaptor_output, CLIPOutput):
            image_embeds = adaptor_output.image_embeds
        elif isinstance(adaptor_output, Tuple):
            if len(adaptor_output) == 2:
                image_embeds = adaptor_output[1][-1]
            else:
                image_embeds = adaptor_output[-1]
        else:  # should always run this
            image_embeds = adaptor_output
        
        logits = self.classifier(image_embeds)
        probs = nn.Sigmoid()(logits)
        loss = self.loss_func(probs.float(), labels.float())
        
        if not return_dict:
            return loss, logits if return_loss else logits
        
        return ImageClassifierOutput(loss=loss, logits=logits)
        
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        y = batch['labels']
        preds = nn.Sigmoid()(outputs.logits)
        self.metrics(preds, y)
        self.log('train_auroc_step', self.metrics, on_step=True, prog_bar=True)
        
        self.lr_schedulers().step()
        return loss
    
    def _shared_eval(self, batch, batch_idx, prefix):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(f'{prefix}_loss', loss, on_step=True)
        
        y = batch['labels']
        preds = nn.Sigmoid()(outputs.logits)
        self.val_metrics.update(preds, y)
        
    def training_epoch_end(self, outputs):
        self.metrics.reset()
    
    def validation_epoch_end(self, outputs):
        self.log('val_auroc_epoch', self.val_metrics.compute(), on_epoch=True, logger=True)
        self.val_metrics.reset()
        