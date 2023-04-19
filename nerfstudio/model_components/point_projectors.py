import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle

from typing import Any, Dict, List, Optional, Tuple, Type, Union

class PointProjectors(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                world_positions: TensorType[..., 3],
                cameras: Cameras) -> Tuple[TensorType[..., 2], TensorType[..., 3]]:

        # World space -> Reference cameras space
        cameras = cameras.to("cuda:0")
        cameras_c2w = cameras.camera_to_worlds # (num_rays, 3, 4)
        fx, fy = cameras.fx, cameras.fy
        cx, cy = cameras.cx, cameras.cy
        R, t = cameras_c2w[..., :3], cameras_c2w[..., 3]
        positions = (R.permute(0,2,1) @ world_positions.unsqueeze(2) - R.permute(0,2,1) @ t.unsqueeze(2)).squeeze()

        # Reference cameras space -> Reference pixels
        x_coords = fx * positions[:, [0]] / (-positions[:, [2]]) + cx
        y_coords = fy * positions[:, [1]] / (-positions[:, [2]]) + cy
        ref_pixels = torch.concatenate((x_coords, y_coords), dim=1)
        return ref_pixels, -positions[:, [2]]
