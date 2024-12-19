"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
from lavis.common.registry import registry
# from lavis.common.quantization import quant_model_bnb
from lavis.models.blip_models.blip import BlipBase, Modex
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn


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


@registry.register_model("blip_caption")
class BlipCaption(BlipBase):
    """
    BLIP captioning model.

    Supported model types:
        - base_coco: fine-tuned BLIP base model on COCO caption dataset (Karparthy split).
        - large_coco: fine-tuned BLIP large model on COCO caption dataset (Karparthy split).

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_caption", "base_coco")
        >>> model = load_model("blip_caption", "large_coco")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_coco": "configs/models/blip_caption_base_coco.yaml",
        "large_coco": "configs/models/blip_caption_large_coco.yaml",
    }

    def __init__(self, image_encoder, text_decoder, prompt=None, max_txt_len=40):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        self.max_txt_len = max_txt_len

    def forward_encoder(self, samples):
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        return image_embeds

    def forward_decoder(self, samples, image_embeds):
        # prepare inputs for forwarding decoder
        raw_text = samples["text_input"]
        text = self.tokenizer(
            raw_text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        decoder_output = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        return decoder_output, decoder_targets

    def forward(self, samples):
        r"""
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size.
        Returns:
            output (BlipOutput): A BlipOutput object containing the following
                attributes:
                - loss (torch.Tensor): A scalar tensor containing the total loss. For BlipCaption, this is the same as the LM loss.
                - loss_lm (torch.Tensor): A scalar tensor containing the LM loss.
                - intermediate_outputs (BlipIntermediateOutput): A BlipIntermediateOutput object containing intermediate outputs.
                  see :class:`lavis.models.blip_models.blip_outputs.BlipOutput` for more details.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> text_input = ["a large statue of a person spraying water from a fountain"]
        >>> samples = {"image": image, "text_input": text_input}
        >>> output = model(samples)
        >>> output.keys()
        odict_keys(['intermediate_output', 'loss', 'loss_lm'])
        >>> output.intermediate_output.image_embeds.shape
        torch.Size([1, 577, 768])
        >>> output.intermediate_output.decoder_labels.shape
        torch.Size([1, 13])
        ```"""

        image_embeds = self.forward_encoder(samples)
        decoder_output, decoder_targets = self.forward_decoder(samples, image_embeds)

        # return decoder_out
        return BlipOutput(
            loss=decoder_output.loss,
            loss_lm=decoder_output.loss,
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> samples = {"image": image}
        >>> captions = model.generate(samples)
        >>> captions
        ['a large statue of a person spraying water from a fountain']
        >>> captions = model.generate(samples, use_nucleus_sampling=True, num_captions=3)
        >>> captions # example output, results may vary due to randomness
        ['singapore showing the view of some building',
        'the singapore harbor in twilight, as the weather is going down',
        'the famous singapore fountain at sunset']
        """
        # prepare inputs for decoder generation.
        encoder_out = self.forward_encoder(samples)
        image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

        prompt = [self.prompt] * image_embeds.size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        # get decoded text
        decoder_out = self.text_decoder.generate_from_encoder(
            tokenized_prompt=prompt,
            visual_embeds=image_embeds,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        captions = [output[len(self.prompt) :] for output in outputs]

        return captions

    @classmethod
    def from_config(cls, cfg):
        
        peft_config_image = set_peft_config(mode='image')
        peft_config_text = set_peft_config(mode='text')
        
        use_modex = True
        if use_modex:
            # num_shared_layer = 0
            assert peft_config_image == {} and peft_config_text == {}

            logging.info("We use the modex")
            modex = nn.ModuleList([Modex(d_model_visual=768, d_model_textual=768, bottleneck=16, layer_index=i) for i in range(12)])            
        else:
            assert peft_config_image != {} or peft_config_text != {}
            modex=None    
        
        # vision encoder
        image_encoder = VisionTransformerEncoder.from_config(cfg,peft_config=peft_config_image,modex=modex)
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoder.from_config(cfg,peft_config=peft_config_text,modex=modex)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 40)

        model = cls(image_encoder, text_decoder, prompt=prompt, max_txt_len=max_txt_len)
        model.load_checkpoint_from_config(cfg)
        
        model.set_required_grad_setting(peft_config_image,peft_config_text,task='captioning',modex=modex)

        logging.info(f"=> model detail is {model}.")
        
         # output all parameters and their corresponding attributes of requires_grad
        for name, param in model.named_parameters():
            logging.info(f"Parameter: {name}, Requires gradient: {param.requires_grad}, Parameter dtype: {param.dtype}")


        # Generate model statistics
        model_info = {}
        model_info['n_params'] = sum(p.numel() for p in model.parameters())
        model_info['n_trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info['n_visual_encoder'] = sum(p.numel() for p in model.visual_encoder.parameters())
        model_info['n_textual_decoder'] = sum(p.numel() for p in model.text_decoder.parameters())
        model_info['trainable_params_ratio'] = model_info['n_trainable_params'] * 100 / model_info['n_params'] 
        logging.info(model_info)
        
        return model
    
    
def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

#         if isinstance(l, (nn.MultiheadAttention, Attention)):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)
