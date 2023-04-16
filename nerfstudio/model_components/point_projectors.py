import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle

from typing import Any, Dict, List, Optional, Tuple, Type, Union

class PointProjectors(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ray_bundle: RayBundle,
                depth: TensorType[..., 1],
                additional_inputs: Dict[str, TensorType]) -> Tuple[TensorType[..., 2], TensorType[..., 3]]:
        world_positions = ray_bundle.origins + ray_bundle.directions * depth
        ref_cameras = additional_inputs["ref_cameras"]
        ref_cameras_c2w = ref_cameras.camera_to_worlds
        fx, fy = ref_cameras.fx, ref_cameras.fy
        cx, cy = ref_cameras.cx, ref_cameras.cy
        R, t = ref_cameras_c2w[:, :3], ref_cameras_c2w[:, 3]
        ref_positions = (R.T @ world_positions.T - R.T @ t).T
        x_coords = fx * ref_positions[:, 0] / (-ref_positions[:, 2]) + cx
        y_coords = fy * ref_positions[:, 1] / (-ref_positions[:, 2]) + cy
        ref_pixels = torch.concatenate((x_coords, y_coords), dim=1)
        return ref_pixels, world_positions
