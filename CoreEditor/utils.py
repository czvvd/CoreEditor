import pdb

import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
import glob
from diffusers.utils import USE_PEFT_BACKEND
from enum import Enum, auto
from torchtyping import TensorType
import nerfstudio.utils.poses as pose_utils


class CameraType(Enum):
    """Supported camera types."""

    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()

def read_depth2disparity(depth_dir):
    depth_paths = sorted(glob.glob(depth_dir + '/*.npy'))
    disparity_list = []
    for depth_path in depth_paths:
        depth = np.load(depth_path) # [512,512,1] 
        
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        # disparity_map = disparity_map.astype(np.uint8)[:,:,0]
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=2)
        disparity_list.append(disparity_map[None]) 

    detected_maps = np.concatenate(disparity_list, axis=0)
    
    control = torch.from_numpy(detected_maps.copy()).float()
    return rearrange(control, 'f h w c -> f c h w')

def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask):
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    key_ref_cross = key_ref_cross[:, ref_frame_index]
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")
    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")

    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)
    attention_probs = attn.get_attention_scores(query, key_ref_cross, attention_mask)
    hidden_states_ref_cross = torch.bmm(attention_probs, value_ref_cross) 
    return hidden_states_ref_cross

class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):

        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        query = attn.head_to_batch_dim(query)
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size
            ref0_frame_index = [0] * video_length
            ref1_frame_index = [1] * video_length
            ref2_frame_index = [2] * video_length
            ref3_frame_index = [3] * video_length
            
            hidden_states_ref0 = compute_attn(attn, query, key, value, video_length, ref0_frame_index, attention_mask)
            hidden_states_ref1 = compute_attn(attn, query, key, value, video_length, ref1_frame_index, attention_mask)
            hidden_states_ref2 = compute_attn(attn, query, key, value, video_length, ref2_frame_index, attention_mask)

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, ref3_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, ref3_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states_ref3 = torch.bmm(attention_probs, value)
        
        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * torch.mean(torch.stack([hidden_states_ref0, hidden_states_ref1, hidden_states_ref2, hidden_states_ref3]), dim=0) if not is_cross_attention else hidden_states_ref3 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def radial_and_tangential_distort(coords, distortion_params):
    """
    Applies radial and tangential distortion to 2D coordinates.

    Args:
        coords: The undistorted coordinates (a torch.Tensor).
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns:
        The distorted coordinates (a torch.Tensor).
    """

    k1, k2, k3, k4, p1, p2 = distortion_params.squeeze()

    x = coords[..., 0]
    y = coords[..., 1]

    r2 = x ** 2 + y ** 2
    radial_distortion = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4

    x_distorted = x * (1 + radial_distortion) + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_distorted = y * (1 + radial_distortion) + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    return torch.stack([x_distorted, y_distorted], dim=-1)


def generate_coords_from_directions(
        ref_camera,
        camera_indices: TensorType["num_rays":..., "num_cameras_batch_dims"],

        directions,
        camera_opt_to_camera
):
    true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]
    num_rays_shape = camera_indices.shape[:-1]
    cam_types = torch.unique(ref_camera.camera_type, sorted=False)

    # print(self.camera_to_worlds.shape)
    c2w = ref_camera.camera_to_worlds
    # print(c2w.shape)
    assert c2w.shape == num_rays_shape + (3, 4)

    if camera_opt_to_camera is not None:
        c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
    rotation = c2w[..., :3, :3]  # (..., 3, 3)

    assert rotation.shape == num_rays_shape + (3, 3)

    directions = torch.matmul(directions, rotation)

    if CameraType.PERSPECTIVE.value in cam_types:
        directions = directions / (directions[..., 2] / (-1))[..., None]
        coords = torch.zeros(*directions.shape[:-1], 2, device=directions.device)
        coords[..., 0] = directions[..., 0]
        coords[..., 1] = directions[..., 1]
    elif CameraType.EQUIRECTANGULAR.value in cam_types:
        raise NotImplementedError(f"Camera type not implemented:{cam_types}")
    elif CameraType.FISHEYE.value in cam_types:
        coords = torch.zeros(*directions.shape[:-1], 2, device=directions.device)
        directions /= torch.linalg.vector_norm(directions, dim=-1, keepdims=True)
        theta = torch.acos(-directions[..., 2])

        sin_theta = torch.sin(theta)
        coords[..., 0] = directions[..., 0] * theta / sin_theta
        coords[..., 1] = directions[..., 1] * theta / sin_theta

        # print(coords)
    else:
        # print(CameraType.EQUIRECTANGULAR.value)
        raise NotImplementedError(f"Camera type not implemented:{cam_types}")

    distortion_params = None
    distortion_params_delta = None
    if ref_camera.distortion_params is not None:
        distortion_params = ref_camera.distortion_params.unsqueeze(0)
        if distortion_params_delta is not None:
            distortion_params = distortion_params + distortion_params_delta
    elif distortion_params_delta is not None:
        distortion_params = distortion_params_delta

    # Do not apply distortion for equirectangular images
    if distortion_params is not None and CameraType.EQUIRECTANGULAR.value not in cam_types:
        coords = radial_and_tangential_distort(
            coords.reshape(1, -1, 2),
            distortion_params,
        ).reshape(coords.shape)

    fx, fy = ref_camera.fx[true_indices].squeeze(-1), ref_camera.fy[true_indices].squeeze(-1)  # (num_rays,)
    cx, cy = ref_camera.cx[true_indices].squeeze(-1), ref_camera.cy[true_indices].squeeze(-1)  # (num_rays,)

    new_coords = torch.zeros_like(coords)
    new_coords[..., 0] = coords[..., 0] * fx + cx
    new_coords[..., 1] = -coords[..., 1] * fy + cy

    return new_coords
