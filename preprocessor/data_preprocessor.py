import os
import sys
from itertools import product
from os import path

import numpy as np
import torch
import torchvision.transforms.functional as F

# from pips import saverloader
# from pips.nets.pips import Pips
from scipy.interpolate import RegularGridInterpolator
from torchvision.io import read_image, read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm, trange

# sys.path.append(path.join(path.dirname(__file__), "pips"))



class DataPreprocessorBase:
    def __init__(self, video_folder_path: str, f_name):
        self.base_folder = video_folder_path
        self.cache_folder = path.join(video_folder_path, self.__class__.__name__)
        if (
            not path.exists(self.cache_folder)
            or len([f for f in os.listdir(self.cache_folder) if f.endswith(".npy")]) == 0
        ):
            if not path.exists(self.cache_folder):
                os.mkdir(self.cache_folder)
            if isinstance(f_name, (str, bytes)):
                frames = self.get_frames_vid(f_name)
            else:
                frames = self.get_frames_img(f_name)
            self.init_meta(frames)
            self.preprocess(frames)
        else:
            self.load_meta()

    def get_frames_vid(self, f_name):
        frames, _, _ = read_video(path.join(self.base_folder, f_name), output_format="TCHW")
        return frames

    def get_frames_img(self, f_name_generator):
        frames = []
        for f_name in f_name_generator:
            frames.append(read_image(path.join(self.base_folder, f_name)))
        return torch.stack(frames)

    def init_meta(self, frames):
        raise RuntimeError("This is an abstract class!!")

    def load_meta(self):
        raise RuntimeError("This is an abstract class!!")

    def preprocess(self, frames):
        raise RuntimeError("This is an abstract class!!")

    def at(self, frame_number, loc_x, loc_y):
        raise RuntimeError("This is an abstract class!!")


class OpticalFlowRAFT(DataPreprocessorBase):
    def __init__(self, video_folder_path: str, f_name="video.mp4", batch_sz=4):
        self.batch_sz = batch_sz
        self.model_input_H = 520
        self.model_input_W = 960
        super().__init__(video_folder_path, f_name)
        self.flows = self.load_cached()

    def load_cached(self):
        self.n_frames = len([f for f in os.listdir(self.cache_folder) if f.endswith(".npy")])
        return np.stack([np.load(path.join(self.cache_folder, f"{i}.npy")) for i in range(self.n_frames)])

    def preprocess(self, frames):
        weights = Raft_Large_Weights.DEFAULT
        transforms = weights.transforms()
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to("cuda")
        model = model.eval()

        for batch_start in trange(0, self.n_frames - 1, self.batch_sz):
            batch_end = min(self.n_frames - 1, batch_start + self.batch_sz)
            batch1 = frames[batch_start:batch_end]
            batch2 = frames[batch_start + 1 : batch_end + 1]
            batch1 = F.resize(batch1, size=[self.model_input_H, self.model_input_W], antialias=False)
            batch2 = F.resize(batch2, size=[self.model_input_H, self.model_input_W], antialias=False)
            batch1, batch2 = transforms(batch1, batch2)
            batch1 = batch1.to("cuda")
            batch2 = batch2.to("cuda")
            with torch.no_grad():
                flows = model(batch1, batch2)[-1].cpu().numpy()
            flows = np.transpose(flows, [0, 2, 3, 1])
            for ii, f in enumerate(flows):
                np.save(path.join(self.cache_folder, f"{ii + batch_start}.npy"), f)

    def generate_grid(self):
        tmp = np.linspace(0, self.img_h, self.model_input_H + 1)
        self.ys = (tmp[1:] + tmp[:-1]) / 2
        tmp = np.linspace(0, self.img_w, self.model_input_W + 1)
        self.xs = (tmp[1:] + tmp[:-1]) / 2

    def init_meta(self, frames):
        self.n_frames, _, self.img_h, self.img_w = frames.shape
        np.savetxt(
            path.join(self.cache_folder, "meta.txt"), np.asarray([self.n_frames, self.img_h, self.img_w]), fmt="%d"
        )
        self.generate_grid()

    def load_meta(self):
        self.n_frames, self.img_h, self.img_w = np.loadtxt(path.join(self.cache_folder, "meta.txt")).astype(int)
        self.generate_grid()

    def at(self, frame_number, loc_x, loc_y):
        assert 0 <= frame_number < self.n_frames
        flow = self.flows[frame_number]
        return RegularGridInterpolator((self.ys, self.xs), flow, bounds_error=False, fill_value=0)((loc_x, loc_y))


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
    optical_flow = OpticalFlowRAFT("preprocessor\\test\\blackswan", (f"{i:05d}.jpg" for i in range(50)))
    print(optical_flow.at(10, [0], [0]))
