# Code for prepare the dataset. Enhanced from the original Monodepth2 codebase.
# Created: 2023-10-5
# Origin used for paper: https://arxiv.org/abs/2404.09831
# Hope you can cite our paper if you use the code for your research.
from __future__ import absolute_import, division, print_function

import os.path

from PIL import Image
import random

import torch.utils.data as data
from torchvision import transforms
import options as g
from torchvision import transforms as T
from utils import *
from torchvision.utils import save_image



def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloader    """

    def __init__(self, opts, filenames, is_train=False, img_ext='.png'):
        super(MonoDataset, self).__init__()
        self.opts = opts
        self.filenames = filenames
        folder_name = {'train': []}
        folder_name['train'] = ['aug1/data', 'aug2/data'] if opts.weather == 'robust' else g.weatherList
        folder_name['train'] = ['rgb/data'] if opts.weather == 'clear' else folder_name['train']
        self.folder_name = folder_name
        self.interp = Image.ANTIALIAS
        self.frame_ids = opts.novel_frame_ids + [0]
        self.is_train = is_train
        self.use_crop = not opts.no_crop if is_train else False
        self.img_ext = img_ext
        self.current_mode = None
        self.loader = pil_loader
        self.to_tensor = T.ToTensor()
        self.target_scales = opts.scales
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.resize = {}
        for i in self.target_scales:
            s = 2 ** i
            self.resize[i] = T.Resize((self.opts.height // s, self.opts.width // s), interpolation=self.interp)


    def preprocess(self, inputs, color_aug):
        """ We create color_aug objects ahead of time and apply the same enhancement to all the images in the project. This ensures that all images fed into the pose network receive the same enhancement. """
        # Adjust the color image to the desired scale and expand it as needed
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k or "color_weather" in k:
                n, im, i = k
                last_scale = (n, im, -1)
                target_scale = 0
                # Minimize additional inputs to reduce video memory
                inputs[(n, im, target_scale)] = self.resize[target_scale](inputs[last_scale])
                last_scale = (n, im, target_scale)
                if n == "color" and im == 0:
                    for s in self.target_scales:
                        if s == -1 or s == 0:
                            continue
                        else:
                            inputs[(n, im, s)] = self.resize[s](inputs[last_scale])
                            last_scale = (n, im, s)
        for k in list(inputs):
            f = inputs[k]
            if "color" in k or "color_weather" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if im == 0 and i == 0 and "color_weather" in k:
                    inputs[("color_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        if not self.is_train:#test part code
            inputs = {'save_mode': True}
            ld_mode = self.current_mode
            if "test" in self.opts.eval_split or "eigen" in self.opts.eval_split:
                info = self.filenames[index].split()
                assert len(info) == 3, "The length of info must be equal to 3"
                get_image = self.get_robust_color(ld_mode, info, info[2], False) if self.opts.weather == "robust" else self.get_color(ld_mode, info, info[2], False)
            elif "cadc" in self.opts.eval_split:
                get_image = self.get_color(os.path.join(self.opts.data_path, self.filenames[index]), False)
            elif "dense" in self.opts.eval_split:
                get_image = self.get_color(os.path.join(self.opts.data_path, 'snowy', self.filenames[index].replace(',', '_') + '.png'), False)
            elif "stereo" in self.opts.eval_split:
                filename = self.index_to_name(ld_mode, index)
                get_image, inputs["name"] = self.get_color(ld_mode, filename, False)

            inputs["org_image"] = self.to_tensor(get_image) if "rgb" in self.opts.vis_what else 1
            inputs[("color", 0, 0)] = self.to_tensor(self.resize[0](get_image))

            if self.opts.debug >= 2:
                print(ld_mode, "test", index)
            return inputs

        inputs = {'save_mode': False}
        do_flip = self.is_train and random.random() > 0.5
        do_color_aug = self.is_train and random.random() > 0.5 if self.opts.weather != 'robust' else False

        length = self.folder_name['train'].__len__()
        ranges = self.opts.mixRate
        assert len(ranges) == length, "The length of mixRate must be equal to the length of folder_name['train']"
        r = random.random()
        for j in range(len(ranges)):
            if r < ranges[j]:
                aug_folder = self.folder_name['train'][j]
                break
        if self.opts.debug >= 2:
            print(aug_folder, self.is_train, index)
        frame = self.filenames[index].split()
        base_folder = 'rgb/data' if self.opts.org_pjct else self.aug_folder

        side = frame[2]
        for i in self.frame_ids:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]  # 全部图片的翻转都是独立的，所以不存在不一致的情况
                inputs[("color", i, -1)] = self.get_color(base_folder, frame, other_side, do_flip)
            else:
                frame_copy = frame.copy()
                frame_copy[1] = str(int(frame_copy[1]) + int(i))
                inputs[("color", i, -1)] = self.get_color(base_folder, frame_copy, side, do_flip)
                if i == 0:
                    inputs[("color_weather", i, -1)] = self.get_color(aug_folder, frame, side, do_flip)

        for scale in self.target_scales:
            K = self.K.copy()
            K[0, :] *= self.opts.width // (2 ** scale)
            K[1, :] *= self.opts.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_ids:
            if ("color_aug", i, -1) in inputs:
                del inputs[("color_aug", i, -1)]
            if ("color", i, -1) in inputs:
                del inputs[("color", i, -1)]
            for j in (self.target_scales + [-1]):
                if ("color_weather", i, j) in inputs:
                    del inputs[("color_weather", i, j)]

        if "s" in self.frame_ids:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def specify_data(self, ld_mode):
        self.current_mode = ld_mode

