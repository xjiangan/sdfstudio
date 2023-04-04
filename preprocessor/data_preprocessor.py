import os
from os import path

import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy.interpolate import RegularGridInterpolator
from torchvision.io import read_image, read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm, trange


class DataPreprocessorBase:
    def __init__(self, video_folder_path: str, f_name):
        self.base_folder = video_folder_path
        self.cache_folder = path.join(video_folder_path, self.__class__.__name__)
        if not path.exists(self.cache_folder) or len(os.listdir(self.cache_folder)) == 0:
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
            batch1 = F.resize(batch1, size=[520, 960], antialias=False)
            batch2 = F.resize(batch2, size=[520, 960], antialias=False)
            batch1, batch2 = transforms(batch1, batch2)
            batch1 = batch1.to("cuda")
            batch2 = batch2.to("cuda")
            with torch.no_grad():
                flows = model(batch1, batch2)[-1].cpu().numpy()
            flows = np.transpose(flows, [0, 2, 3, 1])
            for ii, f in enumerate(flows):
                np.save(path.join(self.cache_folder, f"{ii + batch_start}.npy"), f)

    def generate_grid(self):
        tmp = np.linspace(0, self.img_h, 520 + 1)
        self.ys = (tmp[1:] + tmp[:-1]) / 2
        tmp = np.linspace(0, self.img_w, 960 + 1)
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


class OpticalFlowPIPS(DataPreprocessorBase):
    pass


if __name__ == "__main__":
    optical_flow = OpticalFlowRAFT("preprocessor\\test\\blackswan", (f"{i:05d}.jpg" for i in range(50)), 8)
    print(optical_flow.at(10, [0], [0]))
