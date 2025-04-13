import argparse
import json
import os

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput
import torch
from torch import nn
from nnAudio import features as nnAudioFeatures
from transformers.models.hubert.modeling_hubert import (
    HubertFeatureEncoder,
    HubertModel,
    HubertEncoderStableLayerNorm,
    HubertEncoder,
    HubertEncoderLayer,
    HubertPositionalConvEmbedding,
    HubertAttention,
    HubertFeedForward,
)

    
from .configuration_musichubert import MusicHubertConfig
# from .configuration_musichubert_for_intervention import MusicHubertConfigForIntervention
from .configuration_musichubert_for_intervention import *

class MusicHubertFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        self.feature_extractor_cqt = config.feature_extractor_cqt

        if self.feature_extractor_cqt:
            # v3 concat features
            self.feature_dimension = config.conv_dim[-1] + config.feature_extractor_cqt_bins
            print(f"feature dimention: {self.feature_dimension}")
        else:
            self.feature_dimension = config.conv_dim[-1]
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(self.feature_dimension, eps=config.layer_norm_eps)
        self.projection = nn.Linear(self.feature_dimension, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MusicHubertModel(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfig
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfig,
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        super(HubertModel, self).__init__(config)

        self.config = config
        # print('config: ', config)
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature

        if self.config.feature_extractor_cqt: # false(lz)
            print('initializing cqt extractor for MusicHubert')
            self.feature_extractor_cqt = nnAudioFeatures.cqt.CQT(sr=self.config.sample_rate, hop_length=self.config.sample_rate//50, fmin=32.7, 
                    fmax=None, n_bins=self.config.feature_extractor_cqt_bins, bins_per_octave=self.config.feature_extractor_cqt_bins//7, 
                    filter_scale=1, norm=1, window='hann', center=True, 
                    pad_mode='constant', trainable=False, 
                    output_format='Magnitude', verbose=True)
            # mapping dimension from cqt bins to 1d CNN output
            # self.post_cqt_feature_proj = nn.Linear(self.config.feature_extractor_cqt_bins, self.config.conv_dim[-1])
            
            # cqtfeat v2
            # self.feature_layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
            # self.feature_projection.feat_proj_layer_norm = False # set to false and norm by hand
            # self.feature_projection = MusicHubertFeatureProjection(config)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm: # false(lz)
            assert not config.deepnorm, "must use post-layer_norm with deepnorm"
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            if config.deepnorm: # false(lz)
                self.encoder = HubertEncoder_extend(config)
            else: # true(lz)
                self.encoder = HubertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        # add additional cqt features for transformer input
        if self.config.feature_extractor_cqt: # false(lz)
            features_cqt = self.feature_extractor_cqt(input_values).transpose(1, 2)
            features_cqt = features_cqt[:,:extract_features.shape[1],:] # align shape
            # # v2
            # features_cqt = self.post_cqt_feature_proj(features_cqt)
            # extract_features = self.feature_projection.layer_norm(extract_features) + self.feature_projection.layer_norm(features_cqt) #v2
            # v3
            extract_features = torch.cat([extract_features,features_cqt], 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MusicHubertModelForIntervention(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # if config.do_stable_layer_norm: # false(lz)
        #     assert not config.deepnorm, "must use post-layer_norm with deepnorm"
        #     self.encoder = HubertEncoderStableLayerNorm(config)
        # else:
        #     if config.deepnorm: # false(lz)
        #         self.encoder = HubertEncoder_extend(config)
        #     else: # true(lz)
        #         self.encoder = HubertEncoder(config)

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class HubertEncoderForIntervention(HubertEncoder):
    def __init__(self, config):
        # super(HubertEncoder, self).__init__(config)
        super().__init__(config)
        self.config = config
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.probe_layer = self.config.probe_layer
        self.probe_layer_from = self.config.probe_layer_from
        self.probe_layer_to = self.config.probe_layer_to
        # self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(self.probe_layer_to)])
        # self.layers_2nd_stage = nn.ModuleList([HubertEncoderLayer(config) for _ in range(1)])
        self.gradient_checkpointing = False

    def forward_1st_stage(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        deepspeed_zero3_is_enabled = False

        for layer in self.layers[:self.probe_layer_from]:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else: # 进入这条
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer: # false(lz)
                layer_outputs = (None, None)

            if output_attentions: # false(lz)
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward_2nd_stage(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        layer_forward: int = 0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

       
        # if layer_forward != 0:
        #     position_embeddings = self.pos_conv_embed(hidden_states)
        #     hidden_states = hidden_states + position_embeddings
        #     hidden_states = self.layer_norm(hidden_states)
        #     hidden_states = self.dropout(hidden_states)

        start = self.probe_layer_from + layer_forward
        # for layer in self.layers[self.probe_layer_from:self.probe_layer_from+1]:
        for layer in self.layers[start:start+1]:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )




class MusicHubertModelForIntervention_from1(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from1
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from1,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MusicHubertModelForIntervention_from2(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from2
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from2,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MusicHubertModelForIntervention_from3(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from3
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from3,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MusicHubertModelForIntervention_from4(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from4
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from4,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MusicHubertModelForIntervention_from5(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from5
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from5,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MusicHubertModelForIntervention_from6(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from6
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from6,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MusicHubertModelForIntervention_from7(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from7
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from7,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MusicHubertModelForIntervention_from8(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from8
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from8,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MusicHubertModelForIntervention_from9(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from9
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from9,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MusicHubertModelForIntervention_from10(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from10
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from10,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MusicHubertModelForIntervention_from11(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from11
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from11,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MusicHubertModelForIntervention_from12(HubertModel):
    # overwrite config class
    config_class = MusicHubertConfigForIntervention_from12
    base_model_prefix = "music_hubert"
    def __init__(
        self,
        config: MusicHubertConfigForIntervention_from12,
        # config
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        # super(HubertModel, self).__init__(config)
        super().__init__(config)
        self.config = config
        # print('config: ', config)
        self.probe_layer = self.config.probe_layer
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # replace Feature Projection for introcuing new feature



        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0: # true(lz)
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoderForIntervention(config) # 我加的

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_1st_stage(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None: # false
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward_1st_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_2nd_stage(self, input_values: Optional[torch.Tensor], layer_forward: Optional[int] = 0, attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = input_values
        # print("hidden_states.shape: ", hidden_states.shape)
        encoder_outputs = self.encoder.forward_2nd_stage(
            hidden_states, # [1, 299, 768]
            attention_mask=attention_mask, # none
            output_attentions=output_attentions, # false
            output_hidden_states=output_hidden_states, # true
            return_dict=return_dict,
            layer_forward = layer_forward,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )