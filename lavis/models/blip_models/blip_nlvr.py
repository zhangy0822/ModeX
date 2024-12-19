"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
import torch
import torch.nn.functional as F
from lavis.common.dist_utils import download_cached_file
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path, is_url
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.blip_models.blip import BlipBase, Modex
from lavis.models.blip_models.blip_outputs import BlipIntermediateOutput, BlipOutput
from lavis.models.blip_models.nlvr_encoder import BertModel
from lavis.models.vit import VisionTransformerEncoder, interpolate_pos_embed
from torch import nn
from transformers import BertConfig

def set_peft_config(mode=None, cfg=None):
    peft_config = {}
    if mode == 'image':
        # set the current transfer type for visual modality
        use_lora, use_adapter, use_prompt, use_zia, use_adaptformer = False, False, False, False, False
        if use_lora:
            peft_config['transfer_type'], peft_config['use_lora'] = 'lora', True
            peft_config['lora_alpha'] = 32
            peft_config['lora_r'] = 16
            
        elif use_adapter:
            peft_config['transfer_type'], peft_config['use_adapter'] = 'adapter', True
            peft_config['reducation_factor'] = 1/48
            peft_config['act2fn'] = 'gelu'
        
        elif use_prompt:
            peft_config['transfer_type'], peft_config['use_prompt'] = 'prompt', True
            peft_config['num_token'] = 10
            peft_config['dropout'] = 0.0
            
        elif use_zia:
            peft_config['transfer_type'], peft_config['use_zia'] = 'zia', True
            peft_config['num_token'] = 10
            
        elif use_adaptformer:
            peft_config['transfer_type'], peft_config['use_adaptformer'] = 'adaptformer', True
            peft_config['reducation_factor'] = 1/48
        else:
            logging.info(f"employing fully finetune the model for visual modality")
        # end
        logging.info(f"Visual modality peft_config is: {peft_config}")
        return peft_config
    elif mode == 'text':
        # set the current transfer type for textual modality
        use_lora, use_adapter, use_prompt, use_zia, use_adaptformer = False, False, False, False, False
        
        if use_lora:
            peft_config['transfer_type'], peft_config['use_lora'] = 'lora', True
            peft_config['lora_alpha'] = 32
            peft_config['lora_r'] = 16
            
        elif use_adapter:
            peft_config['transfer_type'], peft_config['use_adapter'] = 'adapter', True
            peft_config['reducation_factor'] = 1/48
            peft_config['act2fn'] = 'gelu'
            
        elif use_prompt:
            peft_config['transfer_type'], peft_config['use_prompt'] = 'prompt', True
            peft_config['num_token'] = 10
            peft_config['dropout'] = 0.0
            
        elif use_zia:
            peft_config['transfer_type'], peft_config['use_zia'] = 'zia', True
            peft_config['num_token'] = 10
            
        elif use_adaptformer:
            peft_config['transfer_type'], peft_config['use_adaptformer'] = 'adaptformer', True
            peft_config['reducation_factor'] = 1/48
        else:
            logging.info(f"employing fully finetune the model for visual modality")
        # end        
        logging.info(f"Textual modality peft_config is: {peft_config}")
        return peft_config



@registry.register_model("blip_nlvr")
class BlipNLVR(BlipBase, MomentumDistilationMixin):
    """
    Class for BLIP NLVR model.

    Supported model types:
        - base: model with pre-trained BLIP weights, used as initialization for fine-tuning.
        - nlvr: finetuned model on NLVR2 dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_nlvr", "nlvr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "nlvr": "configs/models/blip_nlvr.yaml",
    }

    def __init__(self, image_encoder, text_encoder, num_classes, 
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        hidden_size = text_encoder.config.hidden_size
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

        
    def forward(self, samples, is_train=True):
        """
        Forward function for training and evaluation.

        Args:
            samples (dict): a dict of input samples, which contains the following keys:
                - image0 (torch.Tensor): input image 0, shape (batch_size, 3, H, W), default H=384, W=384.
                - image1 (torch.Tensor): input image 1, shape (batch_size, 3, H, W), default H=384, W=384.
                - text_input (list): list of strings, each string is a natural language sentence.
                - label (torch.LongTensor): ground truth label with shape (batch_size,).
            is_train (bool): whether the model is in training mode.
                If True, the model will return the loss;
                If False, the model will return the prediction.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_nlvr", "nlvr")
            >>> samples = {
            ...     "image0": torch.randn(2, 3, 384, 384),
            ...     "image1": torch.randn(2, 3, 384, 384),
            ...     "text_input": ["there is a ferret in tall grass", "there are lips in one of the images"],
            ...     "label": torch.tensor([0, 1]),
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
        """
        text = samples["text_input"]
        text = self.tokenizer(text, padding="longest", return_tensors="pt").to(
            self.device
        )
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        targets = samples["label"]

        image0 = samples["image0"]
        image1 = samples["image1"]
        images = torch.cat([image0, image1], dim=0)

        image_embeds = self.visual_encoder.forward_features(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
        # print(image0_embeds.shape, image0_embeds.dtype, image1_embeds.shape, image1_embeds.dtype, )
        encoder_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=[image0_embeds, image1_embeds],
            encoder_attention_mask=[
                image_atts[: image0_embeds.size(0)],
                image_atts[image0_embeds.size(0) :],
            ],
            return_dict=True,
        )

        prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])

        if is_train:
            loss = F.cross_entropy(prediction, targets)
            return BlipOutput(
                loss=loss,
                intermediate_output=BlipIntermediateOutput(
                    image_embeds=torch.stack([image0_embeds, image1_embeds], dim=0),
                    encoder_output=encoder_output,
                ),
            )
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        
        peft_config_image = set_peft_config(mode='image')
        peft_config_text = set_peft_config(mode='text')       
        
        use_modex = True
        if use_modex:
            # num_shared_layer = 12 
            assert peft_config_image == {} and peft_config_text == {}
            logging.info("We use the modex")
            modex = nn.ModuleList([Modex(d_model_visual=768, d_model_textual=768, bottleneck=16, layer_index=i) for i in range(12)]) 
        else:
            assert peft_config_image != {} or peft_config_text != {}
            modex=None
        
        image_encoder = VisionTransformerEncoder.from_config(cfg,peft_config=peft_config_image,modex=modex)

        # text encoder + multimodal encoder
        bert_config = BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
        text_encoder = BertModel(config=bert_config, add_pooling_layer=False, peft_config=peft_config_text,modex=modex)

        num_classes = cfg.get("num_classes", 3)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            num_classes=num_classes,
        )

        model.load_checkpoint_from_config(cfg)
        
        model.set_required_grad_setting(peft_config_image, peft_config_text, task = 'nlvr2', modex=modex)
        
        logging.info(f"=> model detail is {model}.")
        
        # output all parameters and their corresponding attributes of requires_grad
        for name, param in model.named_parameters():
            logging.info(f"Parameter: {name}, Requires gradient: {param.requires_grad}, Parameter dtype: {param.dtype}")
 
        # Generate model statistics
        model_info = {}
        model_info['n_params'] = sum(p.numel() for p in model.parameters())
        model_info['n_trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info['n_visual_encoder'] = sum(p.numel() for p in model.visual_encoder.parameters())
        model_info['n_textual_encoder'] = sum(p.numel() for p in model.text_encoder.parameters())
        model_info['n_cls_head'] = sum(p.numel() for p in model.cls_head.parameters())
        model_info['trainable_params_ratio'] = model_info['n_trainable_params'] * 100 / model_info['n_params'] 
        logging.info(model_info)


        return model

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

        for key in list(state_dict.keys()):
            if "crossattention.self." in key:
                new_key0 = key.replace("self", "self0")
                new_key1 = key.replace("self", "self1")
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]
            elif "crossattention.output.dense." in key:
                new_key0 = key.replace("dense", "dense0")
                new_key1 = key.replace("dense", "dense1")
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print("load checkpoint from %s" % url_or_filename)
        print(f"missing keys {msg.missing_keys}")
        return msg
