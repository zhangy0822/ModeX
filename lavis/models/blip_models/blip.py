"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
from packaging import version

import torch
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.vit import interpolate_pos_embed
from transformers import BertTokenizer
import transformers
from torch import nn
import math

class BlipBase(BaseModel):
    def __init__(self):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version < version.parse("4.27"), "BLIP models are not compatible with transformers>=4.27, run pip install transformers==4.25 to downgrade"
        
    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        if "visual_encoder_m.pos_embed" in self.state_dict().keys():
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
            )

        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    
class Adapter(nn.Module):
    def __init__(self, d_model=768, bottleneck=None, act_layer=nn.ReLU()):
        super().__init__()
        self.act = act_layer
        self.D_fc1 = nn.Linear(d_model, bottleneck)
        self.D_fc2 = nn.Linear(bottleneck, d_model)
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.D_fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.D_fc1.bias)
            nn.init.zeros_(self.D_fc2.weight)
            nn.init.zeros_(self.D_fc2.bias)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = x + xs
        return x


class MEC_adapter(nn.Module):
    def __init__(self, d_model=768, bottleneck=None, dropout=0.0, act_layer=nn.ReLU()):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.multi_up_proj = nn.Linear(self.down_size, self.n_embd)
        self.visual_up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.multi_up_proj.weight)
            nn.init.zeros_(self.multi_up_proj.bias)
            nn.init.zeros_(self.visual_up_proj.weight)
            nn.init.zeros_(self.visual_up_proj.bias)

        self.expert_weights = nn.Linear(self.n_embd, 3)
        # self.expert_weights=nn.Linear(self.n_embd,2)

        self.t = 10

    def forward(self, x, add_residual=True, mode='text', layer=0):

        residual = x
        weights = torch.softmax(self.expert_weights(x) / self.t, -1)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up_text = self.up_proj(down)
        # up = self.up_proj(down)
        up_visual = self.visual_up_proj(down)
        up_cross = self.multi_up_proj(down)

        up = up_text * weights[:, :, 0].unsqueeze(-1) + up_visual * weights[:, :, 1].unsqueeze(-1) + up_cross * weights[
                                                                                                                :, :,
                                                                                                                2].unsqueeze(
            -1)
        # up = up_text*weights[:,:,0].unsqueeze(-1) + up_visual*weights[:,:,1].unsqueeze(-1)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output



class Modex(nn.Module):
    def __init__(self,
                 d_model_visual=1024, 
                 d_model_textual=768, 
                 bottleneck=16,
                 dropout=0.0,
                 layer_index=0,  
                 ):
        
        super().__init__()

        self.n_embd_visual = d_model_visual
        self.n_embd_textual = d_model_textual
        self.down_size = bottleneck

        self.visual_msa=Adapter(d_model=self.n_embd_visual, bottleneck=self.down_size)
        self.visual_ffn=Adapter(d_model=self.n_embd_visual, bottleneck=self.down_size)
        
        self.layer_index=layer_index
        
            
        # self.textual_msa=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        self.cross_msa=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        # self.cross_ffn=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)

        # self.cross_msa=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        # self.cross_ffn=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        
        # self.textual_ffn=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        
        ## self.decoder_ffn=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        # self.decoder_cross_ffn=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        # self.decoder_msa=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        # self.decoder_cross_msa=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        
        
        
        
    def forward(self, x, mode='text', layer=0):
        # print(mode)
        if mode == 'text_msa':
            if hasattr(self, "textual_msa"):
                output = self.textual_msa(x)
            else:
                output = x
        elif mode == 'text_ffn':
            if hasattr(self, "textual_ffn"):
                output = self.textual_ffn(x)
            else:
                output = x
        elif mode == 'cross_ffn':
            if hasattr(self, "cross_ffn"):
                output = self.cross_ffn(x)
            else:
                output = x
        elif mode == 'visual_msa':
            if hasattr(self, "visual_msa"):
                output = self.visual_msa(x)
            else:
                output = x
        elif mode == 'visual_ffn':
            if hasattr(self, "visual_ffn"):
                output = self.visual_ffn(x)
            else:
                output = x  
        elif mode == 'cross_msa':
            if hasattr(self, "cross_msa"):
                output = self.cross_msa(x)
            else:
                output = x           
        elif mode == 'decoder_ffn':
            if hasattr(self, "decoder_ffn"):
                output=self.decoder_ffn(x)
            else:
                output = x 
        elif mode == 'decoder_cross_ffn':
            if hasattr(self, "decoder_cross_ffn"):
                output=self.decoder_cross_ffn(x)
            else:
                output = x 
        elif mode == 'decoder_msa':
            if hasattr(self, "decoder_msa"):
                output=self.decoder_msa(x)
            else: 
                output = x 
        elif mode == 'decoder_cross_msa':
            if hasattr(self, "decoder_cross_msa"):
                output=self.decoder_cross_msa(x)
            else: 
                output=x
        else:
            raise NotImplementedError

        return output