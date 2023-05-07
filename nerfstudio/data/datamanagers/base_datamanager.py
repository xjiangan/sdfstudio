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
Datamanager.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from rich.progress import Console
from scipy.interpolate import RegularGridInterpolator
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.preprocessor.data_preprocessor import (
    DepthOmniData,
    NormalOmniData,
    OpticalFlowRAFT,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.heritage_dataparser import HeritageDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.mipnerf360_dataparser import Mipnerf360DataParserConfig
from nerfstudio.data.dataparsers.monosdf_dataparser import MonoSDFDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.our_dataparser import OurDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.record3d_dataparser import Record3DDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import GeneralizedDataset, InputDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PixelSampler
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.misc import IterableWrapper

CONSOLE = Console(width=120)

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[  # Omit prefixes of flags in subcommands.
    tyro.extras.subcommand_type_from_defaults(
        {
            "nerfstudio-data": NerfstudioDataParserConfig(),
            "mipnerf360-data": Mipnerf360DataParserConfig(),
            "blender-data": BlenderDataParserConfig(),
            "friends-data": FriendsDataParserConfig(),
            "instant-ngp-data": InstantNGPDataParserConfig(),
            "nuscenes-data": NuScenesDataParserConfig(),
            "record3d-data": Record3DDataParserConfig(),
            "dnerf-data": DNeRFDataParserConfig(),
            "phototourism-data": PhototourismDataParserConfig(),
            "monosdf-data": MonoSDFDataParserConfig(),
            "sdfstudio-data": SDFStudioDataParserConfig(),
            "heritage-data": HeritageDataParserConfig(),
            "our-data": OurDataParserConfig(),
        },
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )
]
"""Union over possible dataparser types, annotated with metadata for tyro. This is the
same as the vanilla union, but results in shorter subcommand names."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:
        1. A Raybundle: This will contain the rays we are sampling, with latents and
            conditionals attached (everything needed at inference)
        2. A "batch" of auxilury information: This will contain the mask, the ground truth
            pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""
        raise NotImplementedError

    @abstractmethod
    def next_train(self, step: int) -> Tuple:
        """Returns the next batch of data from the train data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple:
        """Returns the next batch of data from the eval data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple:
        """Returns the next eval image."""
        raise NotImplementedError

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """


class VanillaDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: VanillaDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser = self.config.dataparser.setup()

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 2,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.train_camera_optimizer,  # should be shared between train and eval.
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            image_indices=self.config.eval_image_indices,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            if isinstance(batch["image"], BasicImages):  # If this is a generalized dataset, we need to get image tensor
                batch["image"] = batch["image"].images[0]
                camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1], 1))
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups


@dataclass
class FlexibleDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: FlexibleDataManager)
    """Target class to instantiate."""
    train_num_images_to_sample_from: int = 1
    """Number of images to sample during training iteration."""


class FlexibleDataManager(VanillaDataManager):
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        additional_output = {}
        if "src_imgs" in image_batch.keys():
            ray_indices = ray_indices.to(image_batch["src_idxs"].device)
            assert (ray_indices[:, 0] == image_batch["image_idx"]).all()
            additional_output["uv"] = ray_indices[:, 1:]
            additional_output["src_idxs"] = image_batch["src_idxs"][0]
            additional_output["src_imgs"] = image_batch["src_imgs"][0]
            additional_output["src_cameras"] = self.train_dataset._dataparser_outputs.cameras[
                image_batch["src_idxs"][0].to("cpu")
            ]
        return ray_bundle, batch, additional_output


@dataclass
class OurDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: OurDataManager)
    """Target class to instantiate."""
    # train_num_images_to_sample_from: int = 1
    # """Number of images to sample during training iteration."""


class OurDataManager(VanillaDataManager):
    def setup_train(self):
        super().setup_train()
        print("Preprocessing training data...")
        frames = torch.permute(
            torch.stack([self.train_dataset.get_image(i) for i in range(len(self.train_dataset))]), (0, 3, 1, 2)
        )
        cache_folder = os.path.join(self.dataparser.config.data.absolute(), "train_cache")
        self.normals_train = NormalOmniData(frames, os.path.join(cache_folder, "normals"))
        self.depths_train = DepthOmniData(frames, os.path.join(cache_folder, "depths"))
        self.optical_flows_train = OpticalFlowRAFT(frames, os.path.join(cache_folder, "optical_flows"))

    def setup_eval(self):
        super().setup_eval()
        print("Preprocessing evaluation data...")
        frames = torch.permute(
            torch.stack([self.eval_dataset.get_image(i) for i in range(len(self.eval_dataset))]), (0, 3, 1, 2)
        )
        cache_folder = os.path.join(self.dataparser.config.data.absolute(), "eval_cache")
        self.normals_eval = NormalOmniData(frames, os.path.join(cache_folder, "normals"))
        self.depths_eval = DepthOmniData(frames, os.path.join(cache_folder, "depths"))
        self.optical_flows_eval = OpticalFlowRAFT(frames, os.path.join(cache_folder, "optical_flows"))

    def next_train(self, step: int) -> Tuple[RayBundle, RayBundle, Dict, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        src_ray_indices = batch["indices"]
        src_ray_bundle = self.train_ray_generator(src_ray_indices)


        # Get source cameras
        src_camera_indices = src_ray_bundle.camera_indices.to('cpu').squeeze() # size: (num_rays, )
        src_cameras = self.train_dataset.cameras[src_camera_indices]

        # Get height and width of a frame
        image_height, image_width = src_cameras.image_height[0], src_cameras.image_width[0]  # size: (1, )

        # Get indices of pixels on source cameras
        src_pixels = src_ray_bundle.coords[:, [1, 0]]  # size: (num_rays, 2), order: (x_coords, y_coords)

        # Load normal, depth, and optical_flow
        src_camera_ids = np.unique(src_camera_indices)
        src_pixels_np = src_pixels.to("cpu").numpy()
        normal = np.empty((src_camera_indices.shape[0], 3))
        depth = np.empty((src_camera_indices.shape[0]))
        optical_flow = np.empty((src_camera_indices.shape[0], 2))
        for c in src_camera_ids:
            curr_entries = np.where(src_camera_indices == c)
            curr_src_pixels = src_pixels_np[curr_entries]
            normal[curr_entries] = self.normals_train.at(c, curr_src_pixels[:, 0], curr_src_pixels[:, 1])
            depth[curr_entries] = self.depths_train.at(c, curr_src_pixels[:, 0], curr_src_pixels[:, 1])
            if c < src_camera_ids.max():
                optical_flow[curr_entries] = self.optical_flows_train.at(
                    c, curr_src_pixels[:, 0], curr_src_pixels[:, 1])
        normal = torch.from_numpy(normal)
        depth = torch.from_numpy(depth)
        optical_flow = torch.from_numpy(optical_flow)

        # ### Verification 1 (debugging only): heatmap of depth + normal + optical flow (x&y)
        # for _tested_camera in range(src_camera_ids.max()-1):
        #     _src_camera_ids = np.ones((1,), dtype=int) * _tested_camera
        #     _src_camera_indices = np.ones((image_height * image_width,), dtype=int) * _tested_camera
        #     _src_pixels_np = np.indices((image_height, image_width)).transpose((1, 2, 0)).reshape((-1, 2)).astype(np.float32)
        #     _src_pixels_np = _src_pixels_np[:, [1, 0]]
        #     _normal = np.empty((_src_pixels_np.shape[0], 3))
        #     _depth = np.empty((_src_pixels_np.shape[0]))
        #     _optical_flow = np.empty((_src_pixels_np.shape[0], 2))
        #     for _c in _src_camera_ids:
        #         _curr_entries = np.where(_src_camera_indices == _c)
        #         _curr_src_pixels = _src_pixels_np[_curr_entries]
        #         _normal[_curr_entries] = self.normals_train.at(_c, _curr_src_pixels[:, 0], _curr_src_pixels[:, 1])
        #         _depth[_curr_entries] = self.depths_train.at(_c, _curr_src_pixels[:, 0], _curr_src_pixels[:, 1])
        #         if _c <= _src_camera_ids.max():
        #             _optical_flow[_curr_entries] = self.optical_flows_train.at(
        #                 _c, _curr_src_pixels[:, 0], _curr_src_pixels[:, 1])
        #     _depth_mat = _depth.reshape((image_height, image_width))
        #     _normal_mat = _normal.reshape((image_height, image_width, 3))
        #     _optical_flow_x_mat = _optical_flow[:, 0].reshape((image_height, image_width))
        #     _optical_flow_y_mat = _optical_flow[:, 1].reshape((image_height, image_width))
        #     if _tested_camera % 10 == 0:
        #         fig, axs = plt.subplots(2, 2)
        #         axs[0,0].imshow(_depth_mat)
        #         axs[0,0].set_title('depth')
        #         axs[0,1].imshow(_normal_mat)
        #         axs[0,1].set_title('normal')
        #         axs[1,0].imshow(_optical_flow_x_mat)
        #         axs[1,0].set_title('optical flow x')
        #         axs[1,1].imshow(_optical_flow_y_mat)
        #         axs[1,1].set_title('optical flow y')
        #         fig.subplots_adjust(wspace=0.2, hspace=0.5)
        #         fig.suptitle('camera {}'.format(_tested_camera))
        #         plt.show()

        # ### Verification 2 (debugging only): warp from src to ref using optical flow
        # _tested_camera = 0 # [0, src_camera_ids.max()-1]
        # _tested_src_camera_idx = torch.where(image_batch["image_idx"] == _tested_camera)[0]
        # _tested_ref_camera_idx = torch.where(image_batch["image_idx"] == _tested_camera+1)[0]
        # _src_img = image_batch["image"][_tested_src_camera_idx.to('cpu').squeeze()].numpy()
        # _ref_img = image_batch["image"][_tested_ref_camera_idx.to('cpu').squeeze()].numpy()
        # _src_camera_ids = np.ones((1,), dtype=int) * _tested_camera
        # _src_camera_indices = np.ones((image_height * image_width,), dtype=int) * _tested_camera
        # _src_pixels_np = np.indices((image_height, image_width)).transpose((1, 2, 0)).reshape((-1, 2)).astype(np.float32)
        # _src_pixels_np = _src_pixels_np[:, [1, 0]]
        # _normal = np.empty((_src_pixels_np.shape[0], 3))
        # _depth = np.empty((_src_pixels_np.shape[0]))
        # _optical_flow = np.empty((_src_pixels_np.shape[0], 2))
        # for _c in _src_camera_ids:
        #     _curr_entries = np.where(_src_camera_indices == _c)
        #     _curr_src_pixels = _src_pixels_np[_curr_entries]
        #     _normal[_curr_entries] = self.normals_train.at(_c, _curr_src_pixels[:, 0], _curr_src_pixels[:, 1])
        #     _depth[_curr_entries] = self.depths_train.at(_c, _curr_src_pixels[:, 0], _curr_src_pixels[:, 1])
        #     if _c <= _src_camera_ids.max():
        #         _optical_flow[_curr_entries] = self.optical_flows_train.at(
        #             _c, _curr_src_pixels[:, 0], _curr_src_pixels[:, 1])
        # _warped_pixels_np = _src_pixels_np + _optical_flow
        # _interp = RegularGridInterpolator((np.arange(image_height), np.arange(image_width)), _ref_img)
        # _warped_X = _warped_pixels_np[:, 0].reshape((image_height, image_width)).clip(0, image_width-1)
        # _warped_Y = _warped_pixels_np[:, 1].reshape((image_height, image_width)).clip(0, image_height-1)
        # _warped_img = _interp((_warped_Y, _warped_X))
        # plt.imshow(_src_img)
        # plt.title("source image")
        # plt.show()
        # plt.imshow(_ref_img)
        # plt.title("reference image")
        # plt.show()
        # plt.imshow(_warped_img)
        # plt.title("warped image")
        # plt.show()

        # Get indices of source rays whose reference cameras are not in training dataloader
        reprojection_mask_indices = src_ray_bundle.camera_indices.eq(src_ray_bundle.camera_indices.to('cpu').max()) \
            .long().to('cpu').squeeze().nonzero().squeeze() # size: (<num_rays, )

        # Get reference cameras
        ref_camera_indices = src_camera_indices + 1 # size: (num_rays, )
        ref_camera_indices[reprojection_mask_indices] = 0
        ref_cameras = self.train_dataset.cameras[ref_camera_indices]

        # Get indices of reference rays
        ref_ray_indices = torch.zeros(src_ray_indices.shape).long() # size: (num_rays, 3)
        ref_ray_indices[:, 0] = ref_camera_indices # camera indices
        ref_ray_indices[:, 1] = src_ray_indices[:, 1] + optical_flow[:, 1] # row indices, y_coords
        ref_ray_indices[:, 2] = src_ray_indices[:, 2] + optical_flow[:, 0] # col indices, x_coords

        # Get indices of reference rays whose resultant pixels are out of the border
        mask_row = torch.logical_or(ref_ray_indices[:, 1] < 0, ref_ray_indices[:, 1] > image_height - 1)
        mask_col = torch.logical_or(ref_ray_indices[:, 2] < 0, ref_ray_indices[:, 2] > image_width - 1)
        disparity_mask_indices = torch.logical_or(mask_row, mask_col).long().to('cpu').\
            squeeze().nonzero().squeeze() # size: (<num_rays, )
        ref_ray_indices[disparity_mask_indices] = 0

        # Generate ray bundle from reference cameras
        ref_ray_bundle = self.train_ray_generator(ref_ray_indices)

        # Store results
        batch["normal"] = normal
        batch["depth"] = depth
        batch["optical_flow"] = optical_flow
        additional_output = {}
        additional_output["src_pixels"] = src_pixels
        additional_output["src_cameras"] = src_cameras
        additional_output["ref_cameras"] = ref_cameras
        additional_output["reprojection_mask_indices"] = reprojection_mask_indices
        additional_output["disparity_mask_indices"] = disparity_mask_indices

        return src_ray_bundle, ref_ray_bundle, batch, additional_output
