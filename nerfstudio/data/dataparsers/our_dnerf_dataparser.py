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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from typing_extensions import Literal
from nerfstudio.cameras import camera_utils

@dataclass
class OurDNeRFDataParserConfig(DataParserConfig):
    """D-NeRF dataset parser config"""

    _target: Type = field(default_factory=lambda: DNeRF)
    """target class to instantiate"""
    data: Path = Path("data/davis/dytrain")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "none"
    """The method to use for orientation."""
    center_poses: bool = False
    """Whether to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""


@dataclass
class DNeRF(DataParser):
    """DNeRF Dataset"""

    config: OurDNeRFDataParserConfig

    def __init__(self, config: OurDNeRFDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms.json")
        img_num = len(meta["frames"])
        img_train_num = int(self.config.train_split_percentage * img_num)
        img_val_num = img_num - img_train_num
        image_filenames = []
        poses = []

        if (split=="train"):
            for frame in meta["frames"][:img_train_num]:
                fname = self.data / Path(frame["file_path"].rstrip(".jpg") + ".png")
                image_filenames.append(fname)
                poses.append(np.array(frame["transform_matrix"]))
            poses = np.array(poses).astype(np.float32)
            times = torch.arange(0, img_train_num, dtype=torch.float32) / (img_train_num-1)

        elif (split=="val"):
            for frame in meta["frames"][img_train_num:]:
                fname = self.data / Path(frame["file_path"].rstrip(".jpg") + ".png")
                image_filenames.append(fname)
                poses.append(np.array(frame["transform_matrix"]))
            poses = np.array(poses).astype(np.float32)
            times = torch.arange(0, img_val_num, dtype=torch.float32) / (img_val_num-1)

        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        fx = meta["fl_x"]
        fy = meta["fl_y"]
        cx = image_width / 2.0
        cy = image_height / 2.0

        orientation_method = self.config.orientation_method
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, _ = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))
        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        # in x,y,z order
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            times=times,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
        )

        return dataparser_outputs
