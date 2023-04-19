import torch 
import torch.nn as nn
from typing import List, Union, Tuple, Dict, Optional

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions, 
    BaseModelOutputWithPoolingAndCrossAttentions, 
)
from transformers.modeling_utils import ModuleUtilsMixin


class FusionModule(nn.Module, ModuleUtilsMixin):
    def __init__(self, config:Optional[BertConfig]=None):
        super(FusionModule, self).__init__()
        self.config = config if config is not None else BertConfig()
        
        self.embeddings = lambda t, i: torch.cat([t, i], dim=1)
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
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        

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
        
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
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
