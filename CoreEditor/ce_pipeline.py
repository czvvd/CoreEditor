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

"""CoreEditor Pipeline and trainer"""

import os
import pdb
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type, List
from rich.progress import Console
from copy import deepcopy
import numpy as np 
from PIL import Image
import mediapy as media
from CoreEditor.lang_sam import LangSAM

import torch, random
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText
from CoreEditor.ce_datamanager import (
    CoreEditorDataManagerConfig,
)
from diffusers.models.attention_processor import AttnProcessor
from CoreEditor import utils
from CoreEditor.utils import generate_coords_from_directions
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps

from CoreEditor.models.pipeline_controlvideo import ControlVideoPipeline
from CoreEditor.models.unet import UNet3DConditionModel
from CoreEditor.models.controlnet import ControlNetModel3D
from CoreEditor.ip2p import InstructPix2Pix

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL


import cv2
import torch.nn.functional as F
from tqdm import tqdm


CONSOLE = Console(width=120)
@dataclass
class CoreEditorPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: CoreEditorPipeline)
    """target class to instantiate"""
    datamanager: CoreEditorDataManagerConfig = CoreEditorDataManagerConfig()
    """specifies the datamanager config"""
    render_rate: int = 1500
    """how many gauss steps for gauss training"""
    render_rate_edited: int = 500
    """how many gauss steps for gauss training"""
    edit_prompt: str = ""
    """Positive Prompt"""
    reverse_prompt: str = ""
    """DDIM Inversion Prompt"""
    negative_prompt: str = ""
    """DDIM Inversion Prompt"""
    added_prompt: str = ""
    """DDIM Inversion Prompt"""
    langsam_obj: str = ""
    """The object to be edited"""
    guidance_scale: float = 5
    """Classifier Free Guidance"""
    num_inference_steps: int = 20
    """Inference steps"""
    chunk_size: int = 5

    mask_factor: float = 1.0
    """Mask Scale"""
    edited_save_path: str = "edited_test/Ours/"
    edited_save_name: str = "Ours"
    """Save Path for edited images"""

    inject_step: int = 0

    ip2p: bool = False

    feature_threshold: float = 2.47

    diff_t: int = 261

    attn_fusion_rate: float = 0.5


class CoreEditorPipeline(VanillaPipeline):
    """CoreEditor pipeline"""

    config: CoreEditorPipelineConfig

    def __init__(
        self,
        config: CoreEditorPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.test_mode = test_mode
        self.langsam = LangSAM()

        self.edit_prompt = self.config.edit_prompt
        self.reverse_prompt = self.config.reverse_prompt
        # self.pipe_device = 'cuda:1'

        if self.config.ip2p:
            self.pipe = InstructPix2Pix(self.device, ip2p_use_full_precision=False)
            # load base text embedding using classifier free guidance
            self.text_embedding = self.pipe.pipe._encode_prompt(
                self.edit_prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=""
            )
            self.added_noise_schedule = [999, 200, 200, 21]
            # self.added_noise_schedule = [999, 999, 999, 999]
        else:
            sd_path = 'CheckPoint/SD/'
            ddim_scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
            ddim_inverser = DDIMInverseScheduler.from_pretrained(sd_path, subfolder="scheduler")
            tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16).to(
                self.device)
            vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16).to(self.device)
            unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16).to(
                self.device)
            controlnet = ControlNetModel3D.from_pretrained_2d('CheckPoint/ControlNet/').to(dtype=torch.float16).to(
                self.device)
            self.pipe = ControlVideoPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                controlnet=controlnet, scheduler=ddim_scheduler,inverse_scheduler=ddim_inverser
            ).to(self.device)
            self.pipe.enable_vae_slicing()
            self.pipe.enable_xformers_memory_efficient_attention()
            self.positive_prompt = self.edit_prompt + ', ' + self.config.added_prompt
            self.positive_reverse_prompt = self.reverse_prompt + ', ' + "low quality"

        random.seed(61)
        self.num_inference_steps = self.config.num_inference_steps
        self.guidance_scale = self.config.guidance_scale
        self.chunk_size = self.config.chunk_size

    def render_reverse(self):
        '''Render rgb, depth and reverse rgb images back to latents'''
        self.mask = torch.zeros(len(self.datamanager.cameras), 512, 512, 1)

        rendered_rgb = []
        rendered_depth = []
        disparity = []
        mask_npy = []

        for cam_idx in range(len(self.datamanager.cameras)):

            CONSOLE.print(f"Rendering view {cam_idx}", style="bold yellow")
            current_cam = self.datamanager.cameras[cam_idx].to(self.device)
            if current_cam.metadata is None:
                current_cam.metadata = {}
            current_cam.metadata["cam_idx"] = cam_idx

            rendered_image = self._model.get_outputs_for_camera(current_cam)


            rendered_rgb.append(rendered_image['rgb'].to(torch.float16).unsqueeze(0)) # [512 512 3] 0-1
            rendered_depth.append(rendered_image['depth'].to(torch.float16).unsqueeze(0)) # [512 512 1]
            disparity.append(self.depth2disparity_torch((rendered_image['depth'].to(torch.float16))[:, :, 0][None]))

            if self.config.langsam_obj != "":
                langsam_obj = self.config.langsam_obj
                langsam_rgb_pil = Image.fromarray(((rendered_image['rgb'].to(torch.float16)).cpu().numpy() * 255).astype(np.uint8))
                masks, _, _, _ = self.langsam.predict(langsam_rgb_pil, langsam_obj)
                try:
                    mask_npy.append(masks.clone().cpu().numpy()[0] * 1)
                except Exception as e:
                    masks = torch.full((1,self.datamanager.cameras[0].height.item(),self.datamanager.cameras[0].width.item()), False, dtype=torch.bool)
                    mask_npy.append(masks.clone().cpu().numpy()[0] * 1)

        rendered_rgb = torch.cat(rendered_rgb,0)
        rendered_depth = torch.cat(rendered_depth,0)

        # rendered_rgb_down = F.interpolate(rendered_rgb.permute(0,3,1,2),size=(400,600),mode='bilinear',align_corners=False).permute(0,2,3,1)
        # for i in range(len(disparity)):
        #     disparity[i] = F.interpolate(disparity[i],size=(400,600),mode='bilinear',align_corners=False)

        with torch.no_grad():
            latent = self.image2latent(rendered_rgb.to(self.device))

        for cam_idx in range(len(self.datamanager.cameras)):
            if self.config.langsam_obj != "":
                self.update_datasets(cam_idx, rendered_rgb[cam_idx].cpu(), rendered_depth[cam_idx], disparity[cam_idx], latent[cam_idx], mask_npy[cam_idx])
            else:
                self.update_datasets(cam_idx, rendered_rgb[cam_idx].cpu(), rendered_depth[cam_idx], disparity[cam_idx],  latent[cam_idx], None)
        self.get_correspondence_all()

    def edit_images_ip2p(self):
        '''Edit images with Video Model'''
        # Set up ControlNet and AttnAlign

        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        mask_images = [self.datamanager.train_data[i]["mask_image"] for i in range(len(self.datamanager.train_data)) if 'mask_image' in self.datamanager.train_data[i].keys()]

        # get original image from dataset
        original_image = [self.datamanager.train_data[i]["original"].unsqueeze(0).to(self.device) for i in range(len(self.datamanager.train_data))]
        original_image = torch.cat(original_image,0).permute(0,3,1,2)

        # generate current index in datamanger
        rendered_image = [self.datamanager.train_data[i]["unedited_image"].unsqueeze(0).to(self.device) for i in range(len(self.datamanager.train_data))]
        rendered_image = torch.cat(rendered_image,0).permute(0,3,1,2)

        edited_images = self.pipe.edit_image(
            self.text_embedding.to(self.device).half(),
            rendered_image.to(self.device).half(),
            original_image.to(self.device).half(),
            guidance_scale=self.config.guidance_scale,
            image_guidance_scale=1.5,
            diffusion_steps=20,
            lower_bound=1.0,
            upper_bound=1.0,
            video_length=len(self.datamanager.train_data),
            corr = self.corr, mask_wrap = self.mask_wrap,
            attn_fusion_rate=self.config.attn_fusion_rate, is_inverse=False,inject_step=self.config.inject_step
        )

        edited_images = edited_images.permute(0,2,3,1).contiguous().to(torch.float32)
        for i in tqdm(range(len(self.datamanager.train_data))):
            edited_image = edited_images[i]
            self.datamanager.train_data[i]["edited_image"] = edited_image
            out_to_save = cv2.cvtColor(edited_image.cpu().numpy().clip(0.0, 1.0) * 255.0, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.config.edited_save_path + str(i) + self.config.edited_save_name + '.png', out_to_save)

            out_to_save = cv2.cvtColor(self.datamanager.train_data[i]["original"].cpu().numpy().clip(0.0, 1.0) * 255.0, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.config.edited_save_path + str(i) + 'aUnedited.png', out_to_save)

            self.datamanager.train_data[i]["image"] = edited_image
            if mask_images != []:
                mask = torch.from_numpy(mask_images[i])
                bg_mask = 1 - mask

                unedited_image = self.datamanager.train_data[i]["unedited_image"].to(edited_image.device)
                bg_cntrl_edited_image = edited_image.permute(2,0,1) * mask[None].to(edited_image.device) + unedited_image.permute(2,0,1) * bg_mask[None].to(edited_image.device)
                self.datamanager.train_data[i]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)
        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")

    def edit_images_control(self):
        '''Edit images with Video Model'''
        # Set up ControlNet and AttnAlign

        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")

        init_latent = [self.datamanager.train_data[i]["z_0_image"].unsqueeze(0) for i in range(len(self.datamanager.train_data))]
        disparity = [self.datamanager.train_data[i]["disparity"].to(self.device) for i in range(len(self.datamanager.train_data))]
        init_latent = torch.cat(init_latent,0).permute(1,0,2,3).unsqueeze(0)

        mask_images = [self.datamanager.train_data[i]["mask_image"] for i in range(len(self.datamanager.train_data)) if 'mask_image' in self.datamanager.train_data[i].keys()]
        # pdb.set_trace()
        with torch.no_grad():
            edited_video = self.pipe(prompt=self.positive_prompt, video_length=len(self.datamanager.train_data), frames=disparity,
                               num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale,
                               width=512, height=512,latents=init_latent, negative_prompt=self.config.negative_prompt, corr=self.corr, mask_wrap=self.mask_wrap,
                               inject_step=self.config.inject_step, feature_threshold=self.config.feature_threshold, attn_fusion_rate=self.config.attn_fusion_rate, diff_t=self.config.diff_t
                               )
        edited_video = edited_video.squeeze().permute(1,2,3,0)


        for i in tqdm(range(len(self.datamanager.train_data))):
            edited_image = edited_video[i]
            # edited_image = F.interpolate(edited_image.permute(2,0,1).unsqueeze(0),size=(self.datamanager.cameras[0].height.item(),self.datamanager.cameras[0].width.item()),mode='bilinear',align_corners=False).squeeze(0).permute(1,2,0)
            self.datamanager.train_data[i]["image"] = edited_image.to(torch.float32)
            out_to_save = cv2.cvtColor(edited_image.cpu().numpy().clip(0.0, 1.0) * 255.0, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.config.edited_save_path + str(i) + self.config.edited_save_name + '.png', out_to_save)

            out_to_save = cv2.cvtColor(self.datamanager.train_data[i]["original"].cpu().numpy().clip(0.0, 1.0) * 255.0,
                                       cv2.COLOR_RGB2BGR)
            
            # cv2.imwrite(self.config.edited_save_path + str(i) + 'aUnedited.png', out_to_save)
            if mask_images != []:
                mask = torch.from_numpy(mask_images[i]).to(edited_image.device)
                bg_mask = 1 - mask

                unedited_image = self.datamanager.train_data[i]["unedited_image"]
                bg_cntrl_edited_image = edited_image.permute(2,0,1) * mask[None] + unedited_image.permute(2,0,1) * bg_mask[None]
                self.datamanager.train_data[i]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)

                out_to_save = cv2.cvtColor(bg_cntrl_edited_image.permute(1,2,0).to(torch.float32).cpu().numpy().clip(0.0, 1.0) * 255.0, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.config.edited_save_path + str(i) + self.config.edited_save_name + '.png', out_to_save)

                # out_to_save = cv2.cvtColor(self.datamanager.train_data[i]["original"].cpu().numpy().clip(0.0, 1.0) * 255.0,
                #                            cv2.COLOR_RGB2BGR)
                # cv2.imwrite(self.config.edited_save_path + str(i) + 'aUnedited.png', out_to_save)
        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")

    def edit_images(self):
        if self.config.ip2p:
            self.edit_images_ip2p()
        else:
            self.edit_images_control()

    def get_correspondence_all(self):
        height = self.datamanager.cameras[0].height.item()/8
        width = self.datamanager.cameras[0].width.item()/8
        heights = [height,31,16,8]
        widths = [width ,47,24,64]
        # heights = [6,12,24,48]
        corr_all_all = []
        mask_all_all = []
        for heightsss_factor in heights:
            corr_all = []
            mask_all = []
            for i in range(len(self.datamanager.train_data)):
                corr_cur = []
                mask_cur = []
                for j in range(len(self.datamanager.train_data)):
                    correspondence, mask = self.cal_correspondence(j, i,thres=0.01*(self.datamanager.cameras[0].height.item()/height)/self.config.mask_factor,height = height,width=width)
                    # if correspondence.shape[1] == 15:
                    #     correspondence = F.interpolate(correspondence.permute(0,3,1,2),size=(16,24),mode='bilinear',align_corners=False).permute(0,2,3,1)
                    #     mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),size=(16,24),mode='bilinear',align_corners=False).squeeze().bool()
                    # if correspondence.shape[1] == 7:
                    #     correspondence = F.interpolate(correspondence.permute(0,3,1,2),size=(8,12),mode='bilinear',align_corners=False).permute(0,2,3,1)
                    #     mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),size=(8,12),mode='bilinear',align_corners=False).squeeze().bool()
                    corr_cur.append(correspondence)
                    mask_cur.append(mask.unsqueeze(0))
                corr_cur = torch.cat(corr_cur, 0)
                mask_cur = torch.cat(mask_cur, 0)
                corr_all.append(corr_cur.unsqueeze(0))
                mask_all.append(mask_cur.unsqueeze(0))
            corr_all = torch.cat(corr_all,0)
            mask_all = torch.cat(mask_all,0)
            corr_all_all.append(corr_all.to(self.device))
            mask_all_all.append(mask_all.to(self.device))
            height = height / 2.0
            width = width / 2.0
        self.corr = corr_all_all
        self.mask_wrap = mask_all_all

        return

    @torch.no_grad()
    def cal_warpped_img(self, ref_index, current_index, ref_image, thres=0.01, mode="bilinear"):
        cur_mask = self.mask[current_index].to(self.device)

        current_camera = self.datamanager.cameras[current_index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1))
        curr_coords = current_camera.get_image_coords(index=0).squeeze()[:, :, [1, 0]]

        # ref_camera_transforms = self.model.camera_optimizer(ref_index.unsqueeze(dim=0))
        ref_camera = self.datamanager.cameras[ref_index].to(self.device)
        ref_ray_bundle = ref_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1))
        depth_current = torch.from_numpy(self.datamanager.train_data[current_index]["depth_image"]).squeeze(0)[:, :, None, None].to(
            self.device)

        point = current_ray_bundle.origins + current_ray_bundle.directions * depth_current
        new_direction = point - ref_ray_bundle.origins

        coords_cur_ref = generate_coords_from_directions(ref_camera, torch.tensor(list(range(1))).unsqueeze(-1),
                                                         new_direction, camera_opt_to_camera=None)

        depth_ref = torch.from_numpy(self.datamanager.train_data[ref_index]["depth_image"]).squeeze(0)[:, :, None, None].to(self.device)
        point = ref_ray_bundle.origins + ref_ray_bundle.directions * depth_ref
        new_direction = point - current_ray_bundle.origins

        coords_ref_cur = generate_coords_from_directions(current_camera, torch.tensor(list(range(1))).unheightsqueeze(-1),
                                                         new_direction, camera_opt_to_camera=None)


        valid_mask = self.get_reprojection_error(coords_cur_ref, coords_ref_cur, curr_coords, thres=thres)
        new_img = self.warp_image(ref_image.to(self.device), coords_cur_ref, valid_mask, mode=mode)
        mask = valid_mask * (1 - cur_mask.squeeze())  # *rgb_mask

        # blended_img = new_img * mask[..., None] + cur_img.to(self.device) * (1 - mask)[..., None]
        cur_img = torch.zeros(512, 512, 3).to(self.device)
        blended_img = new_img * mask[..., None] + cur_img * (1 - mask)[..., None]
        mask = (valid_mask + (cur_mask.squeeze())).clamp(0, 1)

        return blended_img, mask

    @torch.no_grad()
    def cal_correspondence(self, ref_index, current_index, thres=0.2, height=64, width=64):
        current_camera = self.datamanager.cameras[current_index].to(self.device)
        current_camera_tempo = current_camera
        current_camera_tempo.rescale_output_resolution(scaling_factor=height/self.datamanager.cameras[0].height.item())
        current_ray_bundle = current_camera_tempo.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1))
        curr_coords = current_camera_tempo.get_image_coords(index=0).squeeze()[:, :, [1, 0]]

        # ref_camera_transforms = self.model.camera_optimizer(ref_index.unsqueeze(dim=0))
        ref_camera = self.datamanager.cameras[ref_index].to(self.device)
        ref_camera_tempo = ref_camera
        ref_camera_tempo.rescale_output_resolution(scaling_factor=height/self.datamanager.cameras[0].height.item())
        ref_ray_bundle = ref_camera_tempo.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1))
        depth_current = torch.from_numpy(self.datamanager.train_data[current_index]["depth_image"]).squeeze(0)[:, :,
                        None, None].to(self.device)
        depth_current_tempo = F.interpolate(depth_current.permute(2,3,0,1),size=(int(height),int(width)),mode='bilinear',align_corners=False).permute(2,3,0,1)
        point = current_ray_bundle.origins + current_ray_bundle.directions * depth_current_tempo
        new_direction = point - ref_ray_bundle.origins

        coords_cur_ref = generate_coords_from_directions(ref_camera_tempo, torch.tensor(list(range(1))).unsqueeze(-1),
                                                         new_direction, camera_opt_to_camera=None)

        depth_ref = torch.from_numpy(self.datamanager.train_data[ref_index]["depth_image"]).squeeze(0)[:, :, None,
                    None].to(self.device)
        depth_ref_tempo = F.interpolate(depth_ref.permute(2, 3, 0, 1), size=(int(height), int(width)), mode='bilinear',
                                            align_corners=False).permute(2, 3, 0, 1)
        point = ref_ray_bundle.origins + ref_ray_bundle.directions * depth_ref_tempo
        new_direction = point - current_ray_bundle.origins

        coords_ref_cur = generate_coords_from_directions(current_camera_tempo, torch.tensor(list(range(1))).unsqueeze(-1),
                                                         new_direction, camera_opt_to_camera=None)
        if ref_index==current_index:
            is_self = True
        else:
            is_self = False

        valid_mask = self.get_reprojection_error(coords_cur_ref, coords_ref_cur, curr_coords, thres=thres,is_self=is_self)

        src_grid = coords_cur_ref.squeeze().unsqueeze(0)  # 1, H,W,2
        src_grid[:, 0] = src_grid[:, 0] / ((height) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((width) / 2) - 1  # scale to -1~1
        src_grid = src_grid.float()  # 1, N, 1,2


        return src_grid, valid_mask

    def warp_image(self, image, src_grid, mask, mode="bilinear"):
        image = image.permute(2, 0, 1).unsqueeze(0)  # 1, 3 ,H, W

        H, W = image.shape[2:]
        src_grid = src_grid.squeeze().unsqueeze(0)  # 1, H,W,2
        src_grid[:, 0] = src_grid[:, 0] / ((W) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H) / 2) - 1  # scale to -1~1
        src_grid = src_grid.float()  # 1, N, 1,2

        wraped_rgb = F.grid_sample(image, src_grid, mode=mode, padding_mode='zeros', align_corners=False).squeeze().permute(1, 2, 0).float()  # N,3
        new_image = torch.zeros(H, W, 3, device=image.device)

        new_image[mask[..., None].expand(*mask.shape, 3)] = wraped_rgb[mask[..., None].expand(*mask.shape, 3)]
        return new_image

    def get_reprojection_error(self, forward_grid, backward_grid, curr_grid, thres=1, is_self=False):

        forward_grid = forward_grid.squeeze()  # H x W x 2
        backward_grid = backward_grid.squeeze()
        H, W = forward_grid.shape[:2]
        # print(forward_grid)
        forward_grid[..., 0] = forward_grid[..., 0] / (W) * 2 - 1
        forward_grid[..., 1] = forward_grid[..., 1] / (H) * 2 - 1
        W_bound = 1 - 1 / W
        H_bound = 1 - 1 / H
        forward_mask = (forward_grid[..., 0] > -W_bound) & (forward_grid[..., 0] < W_bound) & (
                    forward_grid[..., 1] > -H_bound) & (forward_grid[..., 1] < H_bound)


        backward_grid[..., 0] = backward_grid[..., 0] / (W) * 2 - 1
        backward_grid[..., 1] = backward_grid[..., 1] / (H) * 2 - 1

        backward_mask = (backward_grid[..., 0] > -W_bound) & (backward_grid[..., 0] < W_bound) & (
                    backward_grid[..., 1] > -H_bound) & (backward_grid[..., 1] < H_bound)
        # print(backward_mask.shape)
        backward_grid[~backward_mask, :] = 10

        curr_grid[..., 0] = curr_grid[..., 0] / (W) * 2 - 1
        curr_grid[..., 1] = curr_grid[..., 1] / (H) * 2 - 1

        re_proj_grid = F.grid_sample(backward_grid[None, ...].permute(0, 3, 1, 2), forward_grid[None, ...],
                                     mode="nearest", padding_mode="zeros").squeeze().permute(1, 2, 0)

        error = torch.norm(re_proj_grid - curr_grid.cuda(), dim=-1)

        if is_self:
            return torch.full(forward_mask.size(),True,dtype=torch.bool).to(forward_mask.device)
        # valid_mask = (error < thres) & forward_mask
        else:
            # return forward_mask
            return (error < thres) & forward_mask
    
    def depth2disparity_torch(self, depth):
        """
        Args: depth torch tensor
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity) # 0.00233~1
        disparity_map = torch.concatenate([disparity_map, disparity_map, disparity_map], dim=0)
        return disparity_map[None]

    def update_datasets(self, cam_idx, unedited_image, depth, disparity, latent, mask):
        """Save mid results"""
        self.datamanager.train_data[cam_idx]["unedited_image"] = unedited_image 
        self.datamanager.train_data[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        self.datamanager.train_data[cam_idx]["disparity"] = disparity.cpu()
        if latent is not None:
            self.datamanager.train_data[cam_idx]["z_0_image"] = latent.cpu()
        if mask is not None:
            self.datamanager.train_data[cam_idx]["mask_image"] = mask

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if step>30000 and step % self.config.render_rate_edited == 0:
        # if step == 30500:
            self.datamanager.update_dateset()
            self.render_reverse()
            if self.config.ip2p:
                self.pipe.num_train_timesteps =self.added_noise_schedule[min(len(self.added_noise_schedule)-1, (step-30000)//self.config.render_rate_edited)]
            self.edit_images()
        ray_bundle, batch = self.datamanager.next_train(step) # camera, data
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @torch.no_grad()
    def image2latent(self, image):
        """Encode images to latents"""
        image = image * 2 - 1
        image = image.permute(0, 3, 1, 2)  # torch.Size([1, 3, 512, 512]) -1~1
        if self.config.ip2p:
            latents = self.pipe.pipe.vae.encode(image)['latent_dist'].mean
        else:
            latents = self.pipe.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents
    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError




