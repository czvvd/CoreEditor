# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""
import pdb
# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from tqdm import tqdm
import torch
from rich.console import Console
from torch import nn
from torchtyping import TensorType
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
from CoreEditor.models.unet import UNet3DConditionModel
from einops import rearrange, repeat
import numpy as np
import cv2
CONSOLE = Console(width=120)

try:
    from diffusers import (
        DDIMScheduler,
        StableDiffusionInstructPix2PixPipeline,
    )
    from transformers import logging


except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
logger = logging.get_logger(__name__)
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        sd_path = 'CheckPoint/SD/'
        tokenizer = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder").to(dtype=torch.float16).to(self.device)
        vae = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae").to(dtype=torch.float16).to(self.device)
        unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet_ip2p").to(dtype=torch.float16).to(self.device)
        feature_extractor = CLIPImageProcessor.from_pretrained(IP2P_SOURCE, subfolder="feature_extractor")
        self.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        self.scheduler.set_timesteps(100)

        pipe = StableDiffusionInstructPix2PixPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,scheduler=self.scheduler,feature_extractor=feature_extractor,requires_safety_checker=False,safety_checker=None).to(self.device)
        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()
        # pipe.enable_model_cpu_offload()
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            print("Using full precision")
            pipe.unet.float()
            pipe.vae.float()

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae

        CONSOLE.print("InstructPix2Pix loaded!")

    @torch.no_grad()
    def edit_image(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],  
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        video_length=20,
        **kwargs
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        # self.pipe.to(self.device)
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)
        device = self.device
        kwargs_tempo = kwargs.copy()
        kwargs_tempo['corr'] = None

        # select t, set multi-step diffusion
        T = torch.randint(min_step-1, max_step, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        saved_down_v1 = []
        saved_down_k1 = []
        saved_down_v2 = []
        saved_down_k2 = []
        saved_down_v3 = []
        saved_down_k3 = []
        saved_down_v4 = []
        saved_down_k4 = []
        saved_down_v5 = []
        saved_down_k5 = []
        saved_down_v6 = []
        saved_down_k6 = []
        saved_down_v7 = []
        saved_down_k7 = []
        saved_down_v8 = []
        saved_down_k8 = []
        saved_down_v9 = []
        saved_down_k9 = []

        saved_mid_v = []
        saved_mid_k = []

        saved_up_v1 = []
        saved_up_k1 = []
        saved_up_v2 = []
        saved_up_k2 = []
        saved_up_v3 = []
        saved_up_k3 = []
        saved_up_v4 = []
        saved_up_k4 = []
        saved_up_v5 = []
        saved_up_k5 = []
        saved_up_v6 = []
        saved_up_k6 = []
        saved_up_v7 = []
        saved_up_k7 = []
        saved_up_v8 = []
        saved_up_k8 = []
        saved_up_v9 = []
        saved_up_k9 = []

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            latents = rearrange(latents, '(b f) c h w -> b c f h w',f=video_length)
            image_cond_latents = self.prepare_image_latents(image_cond,video_length=video_length)

        # add noise
        noise = torch.randn_like(latents)
        
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])

        latents_origin = latents

        self.clean_features()

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,**kwargs_tempo).sample

            if t in self.scheduler.timesteps:
                saved_down_v1.append(self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.v.cpu())
                saved_down_k1.append(self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.k.cpu())
                saved_down_v2.append(self.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.v.cpu())
                saved_down_k2.append(self.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.k.cpu())
                # saved_down_v3.append(self.unet.down_blocks[0].attentions[2].transformer_blocks[0].attn1.v.cpu())
                # saved_down_k3.append(self.unet.down_blocks[0].attentions[2].transformer_blocks[0].attn1.k.cpu())
                saved_down_v4.append(self.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.v.cpu())
                saved_down_k4.append(self.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.k.cpu())
                saved_down_v5.append(self.unet.down_blocks[1].attentions[1].transformer_blocks[0].attn1.v.cpu())
                saved_down_k5.append(self.unet.down_blocks[1].attentions[1].transformer_blocks[0].attn1.k.cpu())
                # saved_down_v6.append(self.unet.down_blocks[1].attentions[2].transformer_blocks[0].attn1.v.cpu())
                # saved_down_k6.append(self.unet.down_blocks[1].attentions[2].transformer_blocks[0].attn1.k.cpu())
                saved_down_v7.append(self.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn1.v.cpu())
                saved_down_k7.append(self.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn1.k.cpu())
                saved_down_v8.append(self.unet.down_blocks[2].attentions[1].transformer_blocks[0].attn1.v.cpu())
                saved_down_k8.append(self.unet.down_blocks[2].attentions[1].transformer_blocks[0].attn1.k.cpu())
                # saved_down_v9.append(self.unet.down_blocks[2].attentions[2].transformer_blocks[0].attn1.v.cpu())
                # saved_down_k9.append(self.unet.down_blocks[2].attentions[2].transformer_blocks[0].attn1.k.cpu())

                saved_mid_v.append(self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.v.cpu())
                saved_mid_k.append(self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.k.cpu())

                saved_up_v1.append(self.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.v.cpu())
                saved_up_k1.append(self.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.k.cpu())
                saved_up_v2.append(self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.v.cpu())
                saved_up_k2.append(self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.k.cpu())
                saved_up_v3.append(self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.v.cpu())
                saved_up_k3.append(self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.k.cpu())
                saved_up_v4.append(self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.v.cpu())
                saved_up_k4.append(self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.k.cpu())
                saved_up_v5.append(self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.v.cpu())
                saved_up_k5.append(self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.k.cpu())
                saved_up_v6.append(self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.v.cpu())
                saved_up_k6.append(self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.k.cpu())
                saved_up_v7.append(self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.v.cpu())
                saved_up_k7.append(self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.k.cpu())
                saved_up_v8.append(self.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.v.cpu())
                saved_up_k8.append(self.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.k.cpu())
                saved_up_v9.append(self.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1.v.cpu())
                saved_up_k9.append(self.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1.k.cpu())

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            latents = rearrange(latents,'b c f h w-> (b f) c h w',f=video_length)
            decoded_img = self.latents_to_img(latents)
        video = decoded_img.permute(0,2,3,1)
        for i in tqdm(range(len(video))):
            edited_image = video[i]
            out_to_save = cv2.cvtColor(edited_image.cpu().numpy().astype(np.float32).clip(0.0, 1.0) * 255.0, cv2.COLOR_RGB2BGR)
            cv2.imwrite('edited_test/reference/' + str(i) + '.png', out_to_save)
        ref_idx = input(f"Input reference index: ").strip()
        ref_idx = int(ref_idx)
        latents = latents_origin

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                if i < kwargs["inject_step"] and kwargs['attn_fusion_rate'] != 1.0:
                    self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.inject_v = saved_down_v1[i].to(device)
                    self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.inject_k = saved_down_k1[i].to(device)
                    self.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.inject_v = saved_down_v2[i].to(device)
                    self.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.inject_k = saved_down_k2[i].to(device)
                    # self.unet.down_blocks[0].attentions[2].transformer_blocks[0].attn1.inject_v = saved_down_v3[i].to(device)
                    # self.unet.down_blocks[0].attentions[2].transformer_blocks[0].attn1.inject_k = saved_down_k3[i].to(device)
                    self.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_v = saved_down_v4[i].to(device)
                    self.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_k = saved_down_k4[i].to(device)
                    self.unet.down_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_v = saved_down_v5[i].to(device)
                    self.unet.down_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_k = saved_down_k5[i].to(device)
                    # self.unet.down_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_v = saved_down_v6[i].to(device)
                    # self.unet.down_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_k = saved_down_k6[i].to(device)
                    self.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_v = saved_down_v7[i].to(device)
                    self.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_k = saved_down_k7[i].to(device)
                    self.unet.down_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_v = saved_down_v8[i].to(device)
                    self.unet.down_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_k = saved_down_k8[i].to(device)
                    # self.unet.down_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_v = saved_down_v9[i].to(device)
                    # self.unet.down_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_k = saved_down_k9[i].to(device)

                    self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.inject_v = saved_mid_v[i].to(device)

                    self.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_v = saved_up_v1[i].to(device)
                    self.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_k = saved_up_k1[i].to(device)
                    self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_v = saved_up_v2[i].to(device)
                    self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_k = saved_up_k2[i].to(device)
                    self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_v = saved_up_v3[i].to(device)
                    self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_k = saved_up_k3[i].to(device)
                    self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_v = saved_up_v4[i].to(device)
                    self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_k = saved_up_k4[i].to(device)
                    self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_v = saved_up_v5[i].to(device)
                    self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_k = saved_up_k5[i].to(device)
                    self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_v = saved_up_v6[i].to(device)
                    self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_k = saved_up_k6[i].to(device)
                    self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_v = saved_up_v7[i].to(device)
                    self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_k = saved_up_k7[i].to(device)
                    self.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.inject_v = saved_up_v8[i].to(device)
                    self.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.inject_k = saved_up_k8[i].to(device)
                    self.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1.inject_v = saved_up_v9[i].to(device)
                    self.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1.inject_k = saved_up_k9[i].to(device)
                else:
                    self.clean_features()
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, **kwargs,ref_idx=ref_idx).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            latents = rearrange(latents, 'b c f h w-> (b f) c h w', f=video_length)
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    def clean_features(self):
        self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.inject_v = None
        self.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.inject_k = None
        self.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.inject_v = None
        self.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.inject_k = None
        # self.unet.down_blocks[0].attentions[2].transformer_blocks[0].attn1.inject_v = None
        # self.unet.down_blocks[0].attentions[2].transformer_blocks[0].attn1.inject_k = None
        self.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_v = None
        self.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_k = None
        self.unet.down_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_v = None
        self.unet.down_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_k = None
        # self.unet.down_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_v = None
        # self.unet.down_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_k = None
        self.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_v = None
        self.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_k = None
        self.unet.down_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_v = None
        self.unet.down_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_k = None
        # self.unet.down_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_v = None
        # self.unet.down_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_k = None

        self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.inject_v = None

        self.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[2].attentions[0].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[2].attentions[1].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[2].attentions[2].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.inject_k = None
        self.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1.inject_v = None
        self.unet.up_blocks[3].attentions[2].transformer_blocks[0].attn1.inject_k = None

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def latents_to_img(self, latents: TensorType["BS", 4, "H", "W"]) -> TensorType["BS", 3, "H", "W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: TensorType["BS", 3, "H", "W"],video_length) -> TensorType["BS", 4, "H", "W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()
        image_latents = rearrange(image_latents, '(b f) c h w -> b c f h w', f=video_length)

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
