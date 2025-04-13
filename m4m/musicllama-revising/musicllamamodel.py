import torch
from transformers import LlamaForCausalLM

import os
import torch.nn as nn
import torch
import numpy as np
import math

A_CONTENT = 128256
B_CONTENT = 128257
# SPECIAL_TOKENS = [128257, 128258]
# N_PREFIX = 50


def get_positional_encoding(max_seq_len, d_model):
    positional_encoding = torch.zeros((max_seq_len, d_model))
    position = torch.arange(0, max_seq_len).reshape(-1, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    return positional_encoding


class AudioRepTransformer(nn.Module):
    def __init__(self, feature_dim=768, output_dim=4096, d_model=512, nhead=4, num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=1024, dropout=0.2):
        super(AudioRepTransformer, self).__init__()
        self.input_linear = nn.Linear(feature_dim, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, output_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   batch_first=True,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                 num_layers=num_encoder_layers)
            # nn.Transformer(d_model, nhead, num_encoder_layers,
            #                               num_decoder_layers, dim_feedforward,
            #                               dropout, batch_first=True)
        self.pos_encoding = {
            "frame": get_positional_encoding(33, d_model // 4),
            "second": get_positional_encoding(601, d_model // 2),
            "song": get_positional_encoding(21, d_model // 4),
            "tgt": get_positional_encoding(21, d_model),
        }
        self.nhead = nhead

    def get_pos_encoding(self, pos_id):
        for key in self.pos_encoding:
            if not self.pos_encoding[key].device == pos_id.device:
                self.pos_encoding[key] = self.pos_encoding[key].to(pos_id.device)
        pos_id = pos_id.long()
        frame_encoding = self.pos_encoding["frame"][pos_id[:, :, 0]]
        second_encoding = self.pos_encoding["second"][pos_id[:, :, 1]]
        song_encoding = self.pos_encoding["song"][pos_id[:, :, 2]]
        return torch.cat([frame_encoding, second_encoding, song_encoding], -1)

    def forward(self, src, tgt_id, pos_id):
        x = self.input_linear(src) + self.get_pos_encoding(pos_id)
        #tgt = self.pos_encoding["tgt"][tgt_id.long()]
        memory = self.transformer(x, src_key_padding_mask=pos_id[:, :, 0] > 0)
        # memory_mask = (pos_id[:, None, :, -1] == tgt_id[:, :, None])
        # memory_mask[(pos_id[:, None, :, -1]*tgt_id[:, :, None]) == 0] = False
        # memory_mask = memory_mask[:, None, ...].repeat(1, self.nhead, 1, 1).flatten(0, 1)
        # output = self.transformer.decoder(tgt, memory=memory, tgt_key_padding_mask=tgt_id > 0,
        #                                   memory_key_padding_mask=pos_id[:, :, 0] > 0)
        y = self.out_linear(memory)
        return y


class MusicEncoder(nn.Module):
    def __init__(self, path, device, rvq_dim=2048, emb_dim=4096):
        super().__init__()
        embeddings = torch.from_numpy(np.load(path))
        self.emb = {"emb": embeddings.half()}
        #self.audio_rep_transformer = AudioRepTransformer()
        self.audio_rep_projector = nn.Linear(768, 4096, bias=False) # previous: 128
        print(self.emb["emb"].shape)

        self.device = device
        self.vocab_size = len(self.emb["emb"])

    def forward(self, input_ids, clap_rep=None, pos_id=None,
                tgt=None,
                inference=False,
                loss_mask=None, labels=None,
                hidden_states=None, logits=None, mode="embeddings"):
        # idx = input_ids == A_CONTENT
        idx = (input_ids == A_CONTENT) | (input_ids == B_CONTENT)

        if mode == "embeddings":
            if not self.emb["emb"].device == clap_rep.device:
                self.emb["emb"] = self.emb["emb"].to(clap_rep.device)
                self.audio_rep_projector = self.audio_rep_projector.to(clap_rep.device)
            if inference:
                self.emb["emb"] = self.emb["emb"].float()
            audio_feature = self.audio_rep_projector(clap_rep)
            mask = pos_id > 0
            inputs_embeds = self.emb["emb"][input_ids]
            inputs_embeds = inputs_embeds.float()
            inputs_embeds[idx] = audio_feature[mask]
            return inputs_embeds
        else:
            xlen = labels.shape[-1]
            if xlen > loss_mask.shape[-1]:
                xlen = loss_mask.shape[-1]
                labels = labels[:, :xlen]
                logits = logits[:, :xlen]
            
            print('labels', labels.shape)
            print('logits', logits.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_mask = loss_mask[..., 1:xlen].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss_mask = loss_mask.view(-1)
            shift_logits = shift_logits[loss_mask > 0]
            shift_labels = shift_labels[loss_mask > 0].to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            return loss

            # assert hidden_states is not None
            # hidden_states = hidden_states[idx]
            # predicted_feature = self.out_proj(hidden_states)
            # loss = torch.abs(predicted_feature - clap_rvq).mean()
            # return loss

    def save_weights(self, folder):
        model_path = os.path.join(folder, "music_encoder.pth")
        torch.save(self.state_dict(), model_path)


class MusicLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, embeded_token_path: str = "weights/embeded_tokens.npy"):
        super().__init__(config)
        self.music_encoder = MusicEncoder(embeded_token_path, device='cuda') # device
        print('init musicencoder')

    def forward(
            self,
            attention_mask=None,
            output_attentions=None,
            return_dict=None,
            labels=None,
            input_ids=None,
            clap_rep=None,
            pos_id=None,
            tgt=None,
            loss_mask=None,
            **kwargs
        ):
        print('forwarding')
        inputs_embeds = self.music_encoder(input_ids=input_ids,
                                            clap_rep=clap_rep, pos_id=pos_id,
                                            tgt=tgt,
                                            mode="embeddings")

        out = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = out["hidden_states"][-1].clone().detach().requires_grad_(True)
        logits = out["logits"].clone().detach().requires_grad_(True)

        print('cal loss')
        loss = self.music_encoder(input_ids=None,
                                    labels=labels,
                                    clap_rep=clap_rep,
                                    pos_id=pos_id,
                                    loss_mask=loss_mask,
                                    hidden_states=hidden_states,
                                    logits=logits,
                                    mode="loss")
        out["loss"] = loss
        # outputs = super().forward(*args, **kwargs)
        return out

    def generate(self, *args, **kwargs):
        print('generating')
        inputs_embeds = self.music_encoder(input_ids=kwargs["input_ids"],
                                            clap_rep=kwargs["clap_rep"],
                                            pos_id=kwargs["pos_id"],
                                            inference=True,
                                            mode="embeddings")
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "max_length": kwargs["max_length"]
        }
        outputs = self.base_model.generate(*args, **kwargs)
        return outputs