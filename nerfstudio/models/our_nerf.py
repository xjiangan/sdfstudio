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

"""
Implementation of test-nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.volsdf import VolSDFModel, VolSDFModelConfig


@dataclass
class OurNeRFModelConfig(VolSDFModelConfig):
    """Our NeRF Model Config"""

    _target: Type = field(default_factory=lambda: OurNeRFModel)
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    optical_flow_loss_mult: float = 0.0
    """Optical flow loss multiplier."""
    disparity_loss_mult: float = 0.0
    """Disparity loss multiplier."""

class OurNeRFModel(VolSDFModel):
    """OurNeRF model

    Args:
        config: TestNeRFModel configuration to instantiate model
    """

    config: OurNeRFModelConfig


