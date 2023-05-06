import argparse
import glob
import os.path
import pdb
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from data.transforms import get_transform
from modules.midas.dpt_depth import DPTDepthModel
from modules.unet import UNet
from PIL import Image
from torchvision import transforms


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        return sample['image']


parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--task', dest='task', help="normal or depth")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = './pretrained_models/'

trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# get target task and model
if args.task == 'normal':
    image_size = 384
    
    ## Version 1 model
    # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
    # model = UNet(in_channels=3, out_channels=3)
    # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

    # if 'state_dict' in checkpoint:
    #     state_dict = {}
    #     for k, v in checkpoint['state_dict'].items():
    #         state_dict[k.replace('model.', '')] = v
    # else:
    #     state_dict = checkpoint
    
    
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)

    trans_totensor = transforms.Compose([Resize(
                                            384,
                                            384,
                                            resize_target=None,
                                            keep_aspect_ratio=True,
                                            ensure_multiple_of=32,
                                            resize_method="minimal",
                                            image_interpolation_method=cv2.INTER_CUBIC,
                                        ),
                                        get_transform('rgb', image_size=None)])

elif args.task == 'depth':
    image_size = 384
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
                                transforms.CenterCrop(512)])


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')

        print(f'Reading input {img_path} ...')
        # img = Image.open(img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
# trans_totensor(img)
        img_tensor = trans_totensor({"image": img})[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        # trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0,1)
            output = 1 - output
#             output = standardize_depth_map(output)
            plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
            
        else:
            trans_topil(output[0]).save(save_path)
            
        print(f'Writing output {save_path} ...')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()