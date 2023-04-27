import os
import pickle
import sys
from itertools import product
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F

# from pips import saverloader
# from pips.nets.pips import Pips
from scipy.interpolate import RegularGridInterpolator
from torchvision import transforms
from torchvision.io import read_image, read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm, trange

from nerfstudio.data.datamanagers.preprocessor.omnidata.omnidata_tools.torch.modules.midas.dpt_depth import (
    DPTDepthModel,
)

# sys.path.append(path.join(path.dirname(__file__), "pips"))


class DataPreprocessorBase:
    def __init__(self, frames, cache_folder=None):
        # self.base_folder = video_folder_path
        # self.cache_folder = path.join(video_folder_path, self.__class__.__name__)
        # if (
        #     not path.exists(self.cache_folder)
        #     or len([f for f in os.listdir(self.cache_folder) if f.endswith(".npy")]) == 0
        # ):
        #     if not path.exists(self.cache_folder):
        #         os.mkdir(self.cache_folder)
        #     if isinstance(f_name, (str, bytes)):
        #         frames = self.get_frames_vid(f_name)
        #     else:
        #         frames = self.get_frames_img(f_name)
        #     self.init_meta(frames)
        #     self.data = self.preprocess(frames)
        #     np.save(path.join(self.cache_folder, "data.npy"), self.data)
        # else:
        #     self.load_meta()
        #     self.data = np.load(path.join(self.cache_folder, "data.npy"))
        self.data = None
        if cache_folder is not None:
            self.cache_folder = cache_folder
            if not path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)
            if path.exists(path.join(self.cache_folder, "data.npy")):
                self.data = np.load(path.join(self.cache_folder, "data.npy"))
                meta_data = pickle.load(open(path.join(self.cache_folder, "meta.pkl"), "rb"))
                self.load_meta(meta_data)

        if self.data is None:
            self.data = self.preprocess(frames)
            meta = self.init_meta(frames)
            if cache_folder is not None:
                pickle.dump(meta, open(path.join(self.cache_folder, "meta.pkl"), "wb"))
                np.save(path.join(self.cache_folder, "data.npy"), self.data)

    # def get_frames_vid(self, f_name):
    #     frames, _, _ = read_video(path.join(self.base_folder, f_name), output_format="TCHW")
    #     return frames

    # def get_frames_img(self, f_name_generator):
    #     frames = []
    #     for f_name in f_name_generator:
    #         frames.append(read_image(path.join(self.base_folder, f_name)))
    #     return torch.stack(frames)

    def init_meta(self, frames):
        raise RuntimeError("This is an abstract class!!")

    def load_meta(self, meta_data):
        raise RuntimeError("This is an abstract class!!")

    def preprocess(self, frames):
        raise RuntimeError("This is an abstract class!!")

    def at(self, frame_number, loc_x, loc_y):
        """
        Args:
            frame_number (int)
            loc_x (1d list or numpy array): coordinates along image width
            loc_y (1d list or numpy array): coordinates along image height
        """
        raise RuntimeError("This is an abstract class!!")


class OpticalFlowRAFT(DataPreprocessorBase):
    batch_sz = 4
    model_input_H = 520
    model_input_W = 960

    def preprocess(self, frames):
        print("Computing optical flow...")
        weights = Raft_Large_Weights.DEFAULT
        _transforms = weights.transforms()
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to("cuda")
        model = model.eval()
        results = []
        for batch_start in trange(0, frames.shape[0] - 1, self.batch_sz):
            batch_end = min(frames.shape[0] - 1, batch_start + self.batch_sz)
            batch1 = frames[batch_start:batch_end]
            batch2 = frames[batch_start + 1 : batch_end + 1]
            batch1 = F.resize(batch1, size=[self.model_input_H, self.model_input_W], antialias=False)
            batch2 = F.resize(batch2, size=[self.model_input_H, self.model_input_W], antialias=False)
            batch1, batch2 = _transforms(batch1, batch2)
            batch1 = batch1.to("cuda")
            batch2 = batch2.to("cuda")
            with torch.no_grad():
                flows = model(batch1, batch2)[-1].cpu().numpy()
            flows = np.transpose(flows, [0, 2, 3, 1])
            results.append(flows)
        results = np.concatenate(results, axis=0)
        img_h, img_w = frames.shape[2:]
        results *= np.array([img_w / self.model_input_W, img_h / self.model_input_H])[None, None, None, :]
        return results

    def init_meta(self, frames):
        self.n_frames, _, self.img_h, self.img_w = frames.shape
        tmp = np.linspace(0, self.img_h, self.model_input_H + 1)
        self.ys = (tmp[1:] + tmp[:-1]) / 2
        tmp = np.linspace(0, self.img_w, self.model_input_W + 1)
        self.xs = (tmp[1:] + tmp[:-1]) / 2
        self.flows = self.data
        meta_data = {"n_frames": self.n_frames, "ys": self.ys, "xs": self.xs}
        return meta_data

    def load_meta(self, meta_data):
        self.flows = self.data
        self.n_frames = meta_data["n_frames"]
        self.ys = meta_data["ys"]
        self.xs = meta_data["xs"]

    def at(self, frame_number, loc_x, loc_y):
        assert 0 <= frame_number < self.n_frames
        flow = self.flows[frame_number]
        results = RegularGridInterpolator((self.ys, self.xs), flow, bounds_error=False, fill_value=0)((loc_y, loc_x))
        # results *= np.array([self.img_w / self.model_input_W, self.img_h / self.model_input_H])[None, :]
        return results


class NormalOmniData(DataPreprocessorBase):
    model_input_sz = 384

    def init_meta(self, frames):
        self.n_frames, _, self.img_h, self.img_w = frames.shape
        self.normals = self.data
        self.model_input_W = self.normals.shape[2]
        self.model_input_H = self.normals.shape[1]
        tmp = np.linspace(0, self.img_h, self.model_input_H + 1)
        self.ys = (tmp[1:] + tmp[:-1]) / 2
        tmp = np.linspace(0, self.img_w, self.model_input_W + 1)
        self.xs = (tmp[1:] + tmp[:-1]) / 2
        meta_data = {"n_frames": self.n_frames, "ys": self.ys, "xs": self.xs}
        return meta_data

    def load_meta(self, meta_data):
        self.normals = self.data
        self.n_frames = meta_data["n_frames"]
        self.ys = meta_data["ys"]
        self.xs = meta_data["xs"]

    def align_and_concat(self, frame_left, frame_right, true_width):
        overlapped_region_width = self.model_input_sz * 2 - true_width
        overlap_left_region = frame_left[:, :, -overlapped_region_width:]
        overlap_right_region = frame_right[:, :, :overlapped_region_width]
        left_non_overlap = frame_left[:, :, :-overlapped_region_width]
        right_non_overlap = frame_right[:, :, overlapped_region_width:]
        new_overlap = (overlap_left_region + overlap_right_region) / 2
        result = np.concatenate([left_non_overlap, new_overlap, right_non_overlap], axis=2)
        norms = np.linalg.norm(result, axis=0)
        return result / norms[None, :, :]

    def preprocess(self, frames):
        print("Computing normals...")
        model_path = path.join(path.dirname(__file__), "models/omnidata/omnidata_dpt_normal_v2.ckpt")
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)
        checkpoint = torch.load(model_path, map_location=lambda storage, _: storage.cuda())
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to("cuda")
        model.eval()
        model_img_sz = []

        def get_sz(img):
            model_img_sz.extend(img.shape[1:])
            return img

        trans_totensor = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(self.model_input_sz),
                transforms.Lambda(get_sz),
                transforms.FiveCrop(self.model_input_sz),
                transforms.Lambda(lambda crops: torch.stack([crop for crop in crops[:2]])),
            ]
        )

        results = []
        for i in trange(len(frames)):
            frame = frames[i]
            with torch.no_grad():
                img_tensors = trans_totensor(frame).to("cuda")
                model_output = model(img_tensors).clamp(0, 1).cpu().numpy()
            final_output = self.align_and_concat(model_output[0], model_output[1], model_img_sz[1]).transpose(1, 2, 0)
            results.append(final_output)
        return np.stack(results, axis=0)

    def at(self, frame_number, loc_x, loc_y):
        assert 0 <= frame_number < self.n_frames
        normal = self.normals[frame_number]
        results = RegularGridInterpolator((self.ys, self.xs), normal, bounds_error=False, fill_value=0)((loc_y, loc_x))
        return results


class DepthOmniData(DataPreprocessorBase):
    model_input_sz = 384

    def init_meta(self, frames):
        self.n_frames, _, self.img_h, self.img_w = frames.shape
        self.depths = self.data
        self.model_input_W = self.depths.shape[2]
        self.model_input_H = self.depths.shape[1]
        tmp = np.linspace(0, self.img_h, self.model_input_H + 1)
        self.ys = (tmp[1:] + tmp[:-1]) / 2
        tmp = np.linspace(0, self.img_w, self.model_input_W + 1)
        self.xs = (tmp[1:] + tmp[:-1]) / 2
        meta_data = {"n_frames": self.n_frames, "ys": self.ys, "xs": self.xs}
        return meta_data

    def load_meta(self, meta_data):
        self.depths = self.data
        self.n_frames = meta_data["n_frames"]
        self.ys = meta_data["ys"]
        self.xs = meta_data["xs"]

    def align_and_concat(self, frame_left, frame_right, true_width):
        overlapped_region_width = self.model_input_sz * 2 - true_width
        overlap_left_region = frame_left[:, -overlapped_region_width:]
        overlap_right_region = frame_right[:, :overlapped_region_width]
        left_flatten = overlap_left_region.reshape(-1)
        right_flatten = overlap_right_region.reshape(-1)
        # plt.scatter(left_flatten, right_flatten)
        # plt.plot([left_flatten.min(), left_flatten.max()], [left_flatten.min(), left_flatten.max()], color="red")

        A = np.stack([left_flatten, np.ones_like(left_flatten)], axis=1)
        w = np.linalg.lstsq(A, right_flatten, rcond=None)[0]
        # plt.plot(
        #     [left_flatten.min(), left_flatten.max()],
        #     [w[0] * left_flatten.min() + w[1], w[0] * left_flatten.max() + w[1]],
        #     color="green",
        # )
        # plt.show()
        new_left = w[0] * frame_left + w[1]
        left_non_overlap = new_left[:, :-overlapped_region_width]
        right_non_overlap = frame_right[:, overlapped_region_width:]
        left_overlap = new_left[:, -overlapped_region_width:]
        right_overlap = frame_right[:, :overlapped_region_width]
        new_overlap = (left_overlap + right_overlap) / 2
        result = np.concatenate([left_non_overlap, new_overlap, right_non_overlap], axis=1)
        return result

    def preprocess(self, frames):
        print("Computing depths...")
        model_path = path.join(path.dirname(__file__), "models/omnidata/omnidata_dpt_depth_v2.ckpt")
        model = DPTDepthModel(backbone="vitb_rn50_384")
        checkpoint = torch.load(model_path, map_location=lambda storage, _: storage.cuda())
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to("cuda")
        model.eval()
        model_img_sz = []

        def get_sz(img):
            model_img_sz.extend(img.shape[1:])
            return img

        trans_totensor = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(self.model_input_sz),
                transforms.Lambda(get_sz),
                transforms.FiveCrop(self.model_input_sz),
                transforms.Lambda(lambda crops: torch.stack([crop for crop in crops[:2]])),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
        results = []
        for i in trange(len(frames)):
            frame = frames[i]
            with torch.no_grad():
                img_tensors = trans_totensor(frame).to("cuda")
                model_output = 1 - model(img_tensors).clamp(min=0.0, max=1.0).cpu().numpy()
            final_output = self.align_and_concat(model_output[0], model_output[1], model_img_sz[1])
            # plt.imshow(final_output)
            # plt.show()
            results.append(final_output)
        return np.stack(results, axis=0)

    def at(self, frame_number, loc_x, loc_y):
        assert 0 <= frame_number < self.n_frames
        depth = self.depths[frame_number]
        results = RegularGridInterpolator((self.ys, self.xs), depth, bounds_error=False, fill_value=0)((loc_y, loc_x))
        return results


# class OptitalFlowPIPS(DataPreprocessorBase):
#     def __init__(self, video_folder_path: str, f_name="video.mp4"):
#         self.model_input_H = 360
#         self.model_input_W = 640
#         super().__init__(video_folder_path, f_name)
#         self.flows = self.load_cached()

#     def load_cached(self):
#         self.n_frames = len([f for f in os.listdir(self.cache_folder) if f.endswith(".npy")])
#         return np.stack([np.load(path.join(self.cache_folder, f"{i}.npy")) for i in range(self.n_frames)])

#     def init_meta(self, frames):
#         self.n_frames, _, self.img_h, self.img_w = frames.shape
#         np.savetxt(
#             path.join(self.cache_folder, "meta.txt"), np.asarray([self.n_frames, self.img_h, self.img_w]), fmt="%d"
#         )

#     def load_meta(self):
#         self.n_frames, self.img_h, self.img_w = np.loadtxt(path.join(self.cache_folder, "meta.txt")).astype(int)

#     def infer(self, model, frame_a, frame_b, _batch_stride):
#         frame_a = F.resize(frame_a, [self.model_input_H, self.model_input_W])
#         frame_b = F.resize(frame_b, [self.model_input_H, self.model_input_W])
#         rgbs = torch.stack([frame_a, frame_b]).unsqueeze(0)
#         result = np.empty([self.model_input_H, self.model_input_W, 2])
#         for p in range(10):
#             batch_stride = _batch_stride * 2**p
#             try:
#                 batch_coords = np.mgrid[0 : self.model_input_H : batch_stride, 0 : self.model_input_W : batch_stride]
#                 batch_coords = batch_coords.reshape(2, -1)
#                 for i, j in product(range(batch_stride), repeat=2):
#                     curr_coords = batch_coords + np.array([i, j])[:, None]
#                     curr_coords = curr_coords[
#                         None,
#                         :,
#                         np.logical_and(curr_coords[0] < self.model_input_H, curr_coords[1] < self.model_input_W),
#                     ]
#                     xy = torch.from_numpy(np.transpose(curr_coords[:, [1, 0], :], [0, 2, 1]) + 0.5).float().cuda()
#                     preds, _, _, _ = model(xy, rgbs, iters=6)
#                     trajs_e = preds[-1]
#                     result[curr_coords[0, 0], curr_coords[0, 1]] = (trajs_e - xy).cpu().numpy()
#                 return result, batch_stride
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"| WARNING: ran out of memory, retrying with batch stride {batch_stride}")
#                     torch.cuda.empty_cache()
#                 else:
#                     raise e

#     def preprocess(self, frames):
#         model = Pips(stride=4).cuda()
#         model_ckpt_path = os.path.join(os.path.dirname(__file__), "models/pips")
#         saverloader.load(model_ckpt_path, model)
#         model.eval()
#         batch_stride = 1

#         for i in range(self.n_frames - 1):
#             frame_a = frames[i].cuda()
#             frame_b = frames[i + 1].cuda()
#             with torch.no_grad():
#                 result, batch_stride = self.infer(model, frame_a, frame_b, batch_stride)
#                 np.save(path.join(self.cache_folder, f"{i}.npy"), result)


if __name__ == "__main__":
    root_dir = "C:\\Users\\yanks\\Documents\\ETH\\3DV\\sdfstudio\\data\\davis\\train\\images"
    num_frames = len([f for f in os.listdir(root_dir) if f.endswith(".jpg")])
    frames = []
    for i in range(num_frames):
        img = plt.imread(path.join(root_dir, f"frame_{i+1:05d}.jpg"))
        frames.append(img)
    frames = np.stack(frames, axis=0)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    optical_flow = NormalOmniData(frames)
    print(optical_flow.at(10, [0, 10, 100], [0, 10, 100]))
