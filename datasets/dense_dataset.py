import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class DenseDataset(MonoDataset):
    RAW_HEIGHT = 1024
    RAW_WIDTH = 1920

    def __init__(self, *args, **kwargs):
        super(DenseDataset, self).__init__(*args, **kwargs)

    def get_color(self, path, do_flip):
        color = self.loader(path)
        w, h = color.size
        color = color.crop((20, 50, w - 20, h - 50))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
