from typing import List, Union, Tuple, Dict, Optional

import torch 
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions, 
    BaseModelOutputWithPoolingAndCrossAttentions, 
)
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers.models.clip.modeling_clip import clip_loss

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
        
        self.embeddings = lambda i, t: torch.stack([i, t], dim=1)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

    def forward(
        self, 
        image_embeds:torch.Tensor, 
        text_embeds:Optional[torch.Tensor]=None, 
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
        
        
        # Optional text embedding inputs
        if text_embeds is not None:
            assert text_embeds.device == image_embeds.device, "text and image embeddings must be on the same device"
            inputs_embeds = self.embeddings(image_embeds, text_embeds)
        else:
            inputs_embeds = image_embeds
        if len(inputs_embeds.shape) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(1)
        
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
        image_embeds_:torch.Tensor,
        text_embeds_:Optional[torch.Tensor]=None,
    ):  
        
        image_embeds = self.visual_projection(image_embeds_)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # normalized features
        
        if text_embeds_ is not None:
            text_embeds = self.text_projection(text_embeds_)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # normalized features
            return image_embeds, text_embeds
        
        return image_embeds


class Adaptor(pl.LightningModule):
    def __init__(
        self, 
        adaptor_config:Optional[BertConfig]=None, 
        text_output_dim:int=768, 
        vision_output_dim:Optional[int]=None,  # ignored if vision_model_type is huggingface
        logit_scale_init_value:float=2.6592,  # logit_scale = 1 / temperature
        projection_dim:int=512,
        num_hidden_layers:int=2,
        lr:float=1e-4,
    ):
        super(Adaptor, self).__init__()
        
        self.projection = Project(
            text_embed_dim=text_output_dim, 
            vision_embed_dim=vision_output_dim, 
            projection_dim=projection_dim, 
        )
        self.adaptor_module = AdaptorModule(adaptor_config, num_hidden_layers)
        self.vision_output_dim = vision_output_dim
        self.text_output_dim = text_output_dim
        self.projection_dim = projection_dim
                
        self.logit_scale_init_value = logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        
        self.lr = lr
        self.save_hyperparameters()
        
    def forward(
        self,
        image_embeds_raw:torch.FloatTensor, 
        text_embeds_raw:Optional[torch.FloatTensor]=None, 
        return_loss:Optional[bool]=True,
        return_dict:Optional[bool]=True,
        **kwargs, 
    ) -> Union[Tuple[torch.Tensor], torch.Tensor, CLIPOutput]:
        
        assert len(image_embeds_raw.shape) == 2
        
        if text_embeds_raw is not None:
            assert len(text_embeds_raw.shape) == 2
        
            image_embeds, text_embeds = self.projection(image_embeds_raw, text_embeds_raw)
            outputs = self.adaptor_module(image_embeds, text_embeds)
            
            image_embeds = outputs.last_hidden_state[:, 0, :]
            text_embeds = outputs.last_hidden_state[:, 1, :]
            
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
        
        else:   # text_embeds is none
            image_embeds = self.projection(image_embeds_raw)
            outputs = self.adaptor_module(image_embeds)
            image_embeds = outputs.last_hidden_state[:, 0, :]
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            
            return image_embeds  # ignore return_dict and return_loss
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.lr_schedulers().step()
        return loss
    
    def _shared_eval(self, batch, batch_idx, prefix):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(f'{prefix}_loss', loss, prog_bar=True, logger=True)
        
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
        