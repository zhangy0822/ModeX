"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import os
import time

import lavis.common.dist_utils as dist_utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.dist_utils import download_cached_file
from lavis.common.logger import MetricLogger
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.vit import interpolate_pos_embed
from transformers import BertTokenizer
from torch import nn
import math

class AlbefBase(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        return BertTokenizer.from_pretrained("bert-base-uncased")

    def load_from_pretrained(self, url_or_filename, rename_text_keys=True):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        if (
            "visual_encoder_m.pos_embed" in self.state_dict().keys()
            and "visual_encoder_m.pos_embed" in state_dict
        ):
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
            )

        if rename_text_keys:
            for key in list(state_dict.keys()):
                if "bert" in key:
                    new_key = key.replace("bert.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_output = model.text_encoder.forward_text(text_input)
        text_embed = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :])
        )
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    if hasattr(model.tokenizer, "enc_token_id"):
        text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat = model.visual_encoder.forward_features(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        encoder_output = image_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )
        output = model.text_encoder(
            text_ids[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):

        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx.cpu()].to(model.device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )
        output = model.text_encoder(
            text_ids[start + i].repeat(k_test, 1),
            attention_mask=text_atts[start + i].repeat(k_test, 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


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
        self.layer_index=layer_index
        if self.layer_index<=11:
            self.visual_msa=Adapter(d_model=self.n_embd_visual, bottleneck=self.down_size)
            self.visual_ffn=Adapter(d_model=self.n_embd_visual, bottleneck=self.down_size)
            
            self.layer_index=layer_index
            
                
            self.textual_msa=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
            
            
            if self.layer_index <= 5:
                self.textual_ffn=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
            else:
                self.cross_msa=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
                self.cross_ffn=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
        else:
            self.textual_msa=Adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
            self.cross_msa=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
            self.cross_ffn=MEC_adapter(d_model=self.n_embd_textual, bottleneck=self.down_size)
            

        
        
        
        
        
    def forward(self, x, mode='text', layer=0):
        # print(layer, mode)
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
            assert mode
            
        
        return output
    
    
class MEC_adapter(nn.Module):
    def __init__(self, d_model=768, bottleneck=None, dropout=0.0, act_layer=nn.ReLU()):
        super().__init__()
        self.n_embd=d_model
        self.down_size=bottleneck
        
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj=nn.Linear(self.down_size, self.n_embd)
        self.multi_up_proj=nn.Linear(self.down_size, self.n_embd)
        self.visual_up_proj=nn.Linear(self.down_size, self.n_embd)
            
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
        
        
        self.expert_weights=nn.Linear(self.n_embd,3)
        # self.expert_weights=nn.Linear(self.n_embd,2)
        
        self.t=10
        
    def forward(self, x, add_residual=True, mode='text', layer=0):
               
        residual = x 
        weights=torch.softmax(self.expert_weights(x)/self.t,-1)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up_text = self.up_proj(down)
        # up = self.up_proj(down)
        up_visual = self.visual_up_proj(down)
        up_cross = self.multi_up_proj(down)


        up = up_text*weights[:,:,0].unsqueeze(-1) + up_visual*weights[:,:,1].unsqueeze(-1) + up_cross*weights[:,:,2].unsqueeze(-1)
        # up = up_text*weights[:,:,0].unsqueeze(-1) + up_visual*weights[:,:,1].unsqueeze(-1)

        if add_residual:
            output = up + residual
        else:
            output = up
            
        return output
    
    
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