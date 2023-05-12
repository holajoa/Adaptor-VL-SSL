from typing import List, Union, Tuple, Dict, Optional

import torch 
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions, 
    BaseModelOutputWithPoolingAndCrossAttentions, 
)
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers.models.clip.modeling_clip import clip_loss

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import logging
from tqdm import tqdm
import sys


class AdaptorModule(nn.Module, ModuleUtilsMixin):
    def __init__(
        self, 
        config:Optional[BertConfig]=None, 
        num_hidden_layers:int=2, 
    ):
        super(AdaptorModule, self).__init__()
        if config is not None:
            self.config = config 
        else:
            self.config = BertConfig(num_hidden_layers=num_hidden_layers)
        
        self.embeddings = lambda t, i: torch.stack([t, i], dim=1)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

    def forward(
        self, 
        text_embeds:torch.Tensor, 
        image_embeds:torch.Tensor, 
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs, 
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        assert text_embeds.device == image_embeds.device, "text and image embeddings must be on the same device"
        inputs_embeds = self.embeddings(text_embeds, image_embeds)
        
        # Copied from transformers/models/bert/modeling_bert.py, BertModel.forward()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        use_cache = False
        
        device = inputs_embeds.device
        
        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
            
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        encoder_outputs = self.encoder(
            inputs_embeds,   # concatenated text and image embeddings
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) # if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class Project(nn.Module):
    def __init__(
        self, 
        text_embed_dim:int,
        vision_embed_dim:int,
        projection_dim:int=512,
    ):
        super(Project, self).__init__()
        self.text_embed_dim = text_embed_dim
        self.vision_embed_dim = vision_embed_dim
        self.projection_dim = projection_dim
        self.visual_projection = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_embed_dim, projection_dim, bias=False)

    def forward(
        self,
        text_embeds_:torch.Tensor,
        image_embeds_:torch.Tensor,
    ):
        text_embeds = self.text_projection(text_embeds_)
        image_embeds = self.visual_projection(image_embeds_)
        
        # normalized features
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        return image_embeds, text_embeds


class Adaptor(pl.LightningModule):
    def __init__(
        self, 
        text_model:nn.Module,
        vision_model:nn.Module,
        adaptor_config:Optional[BertConfig]=None, 
        vision_model_type:str='huggingface',
        vision_output_dim:Optional[int]=None,  # ignored if vision_model_type is huggingface
        logit_scale_init_value:float=2.6592,  # logit_scale = 1 / temperature
        projection_dim:int=512,
        num_hidden_layers:int=2,
        lr:float=1e-4,
    ):
        super(Adaptor, self).__init__()
        
        self.vision_model = vision_model
        self.text_model = text_model
        self.projection = Project(
            text_embed_dim=text_model.config.hidden_size, 
            vision_embed_dim=vision_output_dim, 
            projection_dim=projection_dim, 
        )
        self.adaptor_module = AdaptorModule(adaptor_config, num_hidden_layers)
        self.vision_model_type = vision_model_type
        self.vision_output_dim = vision_output_dim
        self.projection_dim = projection_dim
                
        self.logit_scale_init_value = logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        
        if self.vision_model_type != 'transformer':
            assert vision_output_dim is not None, \
                'Please provide vision_output_dim for non transformer vision models'
        
        self.lr = lr
        freeze_encoder(self)
        
        self.save_hyperparameters(ignore=["text_model", "vision_model"])
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeds_raw: Optional[torch.FloatTensor] = None, 
        text_embeds_raw: Optional[torch.FloatTensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = True,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs, 
    ) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        if image_embeds_raw is None:
            assert pixel_values is not None, \
                "Must pass pixel_values if no precomputed image_embeds_raw is provided."
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
        else:
            # logging.info('Using precomputed image_embeds_raw.')
            if self.vision_model_type == 'ae' and len(image_embeds_raw.shape) == 4:
                image_embeds_raw = torch.flatten(image_embeds_raw, start_dim=2).permute((0, 2, 1)).mean(1)
        assert len(image_embeds_raw.shape) == 2
        
        if text_embeds_raw is None:
            assert input_ids is not None, \
                "Must pass input_ids if no precomputed text_embeds_raw is provided."
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            text_embeds_raw = text_outputs.pooler_output
        # else:
        #     logging.info('Using precomputed text_embeds_raw.')
        
        image_embeds, text_embeds = self.projection(text_embeds_raw, image_embeds_raw)
        outputs = self.adaptor_module(text_embeds, image_embeds)
        
        text_embeds = outputs.last_hidden_state[:, 0, :]
        image_embeds = outputs.last_hidden_state[:, 1, :]
        
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale   # [batch_size, batch_size]
        logits_per_image = logits_per_text.T
        
        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)
        
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds)
            return ((loss,) + output) if loss is not None else output
        
        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.lr_schedulers().step()
        return loss
    
    def _shared_eval(self, batch, batch_idx, prefix):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(f'{prefix}_loss', loss)
        
    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        lr_schedule = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2, eta_min=self.lr/10)
        return {'optimizer':optimizer, 'lr_scheduler':lr_schedule}
    
        
class AdaptorTrainer(Trainer):
    def __init__(self, num_of_batches=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_of_batches = num_of_batches
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from transformers.trainer_utils import seed_worker
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = self._get_train_sampler()
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        
class StreamingProgressBar(TQDMProgressBar):
    def __init__(self, total:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total = total
         
    def init_train_tqdm(self):
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            total=self._total,
        )
        return bar


def freeze_encoder(model:Adaptor):
    for encoder in [model.text_model, model.vision_model]:
        for param in encoder.parameters():
            param.requires_grad = False
            
