# Some utils code made by myself to aid the training process. Include:
'''
1. Multi-gpu training code;
2. Visualization the pred depth, abs_rel, etc.;
3. Tensorboard aggregation
4.
'''
# Copyright Jiyuan Wang 2024. Patent Pending. All rights reserved.
# Created: 2023-10-5
# Origin used for paper: https://arxiv.org/abs/2404.09831
# We delete some personal information and mark with #change. Hope you can cite our paper if you use the code for your research.
import numpy as np
from PIL import Image
from PIL import ImageChops
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from layers import depth_to_disp, disp_to_depth
from tqdm import trange
import cv2
from gpu_chek.gpu_checker import get_gpu_info
from utils import *
import options as g
from datetime import datetime
import torch
import subprocess
import concurrent.futures
import torch.nn as nn
import torch.nn.functional as F
from options import MonodepthOptions
from tqdm import tqdm, trange

MIN_DEPTH = g.MIN_DEPTH
MAX_DEPTH = g.MAX_DEPTH
new_width = g.defalut_width
new_height = g.defalut_height
verbose = False  # Enter True to print the other information
save_tmp = False  # Enter True to save the middle result


def ini():
    print("please check the super parameters in my_utils.py")
    print("MIN_DEPTH:", MIN_DEPTH, ";MAX_DEPTH:", MAX_DEPTH, ";new_width:", new_width, ";new_height:", new_height,
        ";verbose:", verbose, ";save_tmp:", save_tmp)
    print("dataset:", dataset)
    options = MonodepthOptions()
    return options.parse()


def crop_image(re_img, nw=new_width, nh=new_height):
    if isinstance(re_img, Image.Image):
        width, height = re_img.size
    else:
        height, width = re_img.shape[:2]
    left = (width - nw) / 2
    top = (height - nh) / 2
    right = (width + nw) / 2
    bottom = (height + nh) / 2
    crop_im = re_img.crop((left, top, right, bottom)) if isinstance(re_img, Image.Image) else re_img[int(top):int(bottom), int(left):int(right)]
    return crop_im



def createUniqueName(args):
    tmp = "test" if args.debug else ""
    T = datetime.now().strftime('%d-%H-%M')
    name = [tmp, args.model_name, args.weather, T]
    name = '_'.join(name)
    return name

def modify_rate(rates, weather):
    if weather == 'robust':
        rates = [0.5, 1]
        return rates
    elif weather != 'all':
        rates = [1]
        return rates

    need2done, record, newRate = False, 0, []
    for rate in rates:
        if rate > 1:
            need2done, record = True, int(rate)
            break
    if not need2done:
        newRate = rates.append(1)
    else:
        for i in range(record):
            newRate.append((i + 1) / record)
    return newRate

def setup_for_distributed(is_master):
    """This function disables printing when not in master process    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class doNothing():
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def crop(basepath, outpath):
    basepath = input('please input the image path:') if basepath is None else basepath
    outpath = input('please input the output path:') if outpath is None else outpath
    os.makedirs(outpath, exist_ok=True)
    dir_list = os.listdir(basepath)
    print("length of dir_list:", len(dir_list))
    for i in trange(len(dir_list)):
        if os.path.exists(os.path.join(outpath, dir_list[i])):
            if verbose:
                print("file exist")
            continue
        path = os.path.join(basepath, dir_list[i])
        if verbose:
            print("path:", path)
        img = Image.open(path).convert('RGB')
        crop_im = crop_image(img)
        crop_im.save(os.path.join(outpath, dir_list[i]))

def VisualizeDepth(path=None, depth=None, out_name='depth', process='None', writer=None, rcd=None, debug=0, nwidth=new_width, nheight=new_height, finish=True, color_map='inferno', save_as='disp'):
    '''
    For visualization, the program call template is as follows:

    if Visualization:
        # Output and check the result
        val_depth(writer, None, pred_depth, 'val_pred_depth', process='None', epoch=epoch)
        raise NotImplementedError

    - `writer`: You can pass a `writer`.
    - The first argument after the writer should be `None`.
    - The second argument can be `pred_depth` or `gt_depth`, representing the image to be visualized.
    - The third argument is the output image name without any path or suffix, and it will be automatically saved in the `val` folder.
    - The fourth argument specifies the processing method: `resize`, `crop`, or `none`.
    - Lastly, after specifying the `writer`, you can also pass `rcd`, which indicates the recording location, such as the `epoch`.
    '''
    if process == 'resize':
        p_depth = np.array(Image.fromarray(depth).resize((nwidth, nheight)))
    elif process == 'crop':
        p_depth = crop_image(depth, nwidth, nheight)
    elif process == 'clip':
        p_depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
    else:
        p_depth = depth

    if save_as == 'disp':
        disp = depth_to_disp(p_depth, min_depth=0.1, max_depth=100)
        disp_color = VisualizeMap(disp, map_type=color_map, validImg=np.logical_and(disp > 0, disp < 1), whiteBG=False)
    else:
        disp_color = VisualizeMap(p_depth, map_type='inferno_r', vmin=0.1, vmax=80)
    # save or return
    if finish:
        if writer is not None:
            disp_color = np.transpose(disp_color, (2, 0, 1))
            if "gt" in out_name:
                writer.add_image(out_name, disp_color, global_step=0)
            else:
                writer.add_image(out_name, disp_color, global_step=rcd)
        else:
            out_name = out_name.replace("/", "_")
            plt.imsave('./val/{}'.format(out_name + ".png"), disp_color)
            plt.clf()
            plt.close()
    else:
        return disp_color


def VisualizeMap(showImg, map_type='inferno', validImg=None, whiteBG=False, vmin=None, vmax=None, doTrans=False):
    '''
    :param showImg: The image to be converted to a color map
    :param map_type: color map type
    :param validImg: The bool matrix, shape, is the same as showImg, and is used to filter values in showImg
    '''
    if showImg.ndim == 3:
        showImg = showImg.squeeze(0)
    if vmin is None and vmax is None:
        if validImg is None:
            vmin = showImg.min()
            vmax = showImg.max()
        else:
            vmin = showImg[validImg].min()
            vmax = showImg[validImg].max()
    normImg = ((showImg - vmin) / (vmax - vmin)) * validImg if validImg is not None else (showImg - vmin) / (vmax - vmin)
    if whiteBG:
        normImg[~validImg] = np.nan
    normImg = normImg.cpu() if torch.is_tensor(normImg) else normImg
    showImg_color = plt.get_cmap(map_type)(normImg)

    showImg_color = (showImg_color * 255).astype(np.uint8)
    if doTrans:
        showImg_color = np.transpose(showImg_color, (2, 0, 1))
    return showImg_color



def VisualizeAll(opt, pred_depth, gt_depth, mask, mode, i, Record):
    color = {}
    if 'rgb' in opt.vis_what:
        color['rgb'] = Record["image"][i].cpu().numpy().transpose(1, 2, 0)
    if 'depth' in opt.vis_what:
        color['depth'] = VisualizeDepth(None, pred_depth, process='resize', finish=False, save_as='depth')
    if 'gt' in opt.vis_what:
        color['gt'] = VisualizeMap(gt_depth, 'inferno_r', validImg=np.logical_and(gt_depth > 0.1, gt_depth < 80), whiteBG=True, vmin=0.1, vmax=80)
    if 'abs' in opt.vis_what:
        error_map = np.multiply(np.abs(gt_depth - pred_depth) / (gt_depth + 1e-6), mask)
        color['abs'] = VisualizeMap(error_map, 'inferno_r', mask, vmin=0, vmax=0.2, whiteBG=True)
    if 'feat' in opt.vis_what:
        color['after_img'] = Record["after-img"][i].cpu().numpy().transpose(1, 2, 0).clip(0, 1)

        before = Record["before-feat"][i].cpu().numpy()
        after = Record["after-feat"][i].cpu().numpy()
        mask = np.ones_like(before, dtype=bool)
        mask[0, :] = False
        mask[:, 0] = False
        mask[:, -2:] = False
        color['before_feat'] = VisualizeMap(before, 'inferno', mask)
        color['after_feat'] = VisualizeMap(after, 'inferno', mask)
    length, into = len(color), 1
    if len(opt.vis_what) == 1 and opt.vis_what[0] != 'feat':
        plt.imsave('./val/{}'.format(mode.replace("/", "_") + str(i) + ".png"), color[opt.vis_what[0]])
        return
    plt.figure(figsize=(60, 60))
    for key in color:
        plt.subplot(length, 1, into)
        plt.imshow(color[key])
        plt.title(key)
        plt.axis('off')
        into += 1
    plt.tight_layout()
    plt.savefig('./val/{}'.format(mode.replace("/", "_") + str(i) + ".png"))
    plt.clf()
    plt.close()


def move_folder(basepath=None, path=None, mode=None):
    import shutil
    if mode is None:
        raise ValueError("mode is None")
    if path is None:
        raise ValueError("path is None")
    src = os.path.join(basepath.split('visualize')[0], path, mode)
    dst = os.path.join(basepath, path)
    shutil.copytree(src, dst)


def easy_visualize(basepath=None, mode=None):
    '''
    We use this function to aggregate all of the experiments' log together and have a comparison
    '''
    basepath = modify_opt(path=basepath)
    list = os.listdir(basepath)
    list = [i for i in list if i != 'visualize' and i != 'offical']
    basepath = os.path.join(basepath, 'visualize')
    remove_logfolder(basepath, True)
    os.mkdir(basepath)
    for path in tqdm(list):
        move_folder(basepath, path, mode)


def meshgrid(x, y, indexing):
    grid_y, grid_x = torch.meshgrid(y, x)
    return (grid_x, grid_y)

#change
def modify_opt(opt=None, path=None):
    """It is used to detect the Server (I finished this work in 3 server and have to change the opt or some path each by each to fit the server), modify the opt parameter, and can also be used to modify the path; you can modify it by yourself"""
    if opt is not None:
        return opt
    else:
        return path

def remove_logfolder(log_path, overwrite=False):
    """Used to delete the log folder with the same name"""
    import shutil
    if os.path.exists(log_path) and overwrite:
        shutil.rmtree(log_path)
        print("Has Removed old log files at:  ", log_path)

class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None, d_map=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = torch.abs(target - pred)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()
        part1 = -F.threshold(-diff, -delta, 0.)  # 筛选出diff>0.2*max(diff)的部分置零,其余为diff
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)  # 筛选出diff<delta的部分置零，其余为diff^2+delta^2
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        return diff

def calculate_parameters(opts, encoder, decoder):
    if not opts.caulculate_computation:
        return
    import sys
    from thop import profile
    input = torch.rand((12, 3, 192, 640), device='cuda')
    x0 = torch.rand((6, 1, 192, 640), device='cuda')
    f1 = torch.randn((12, 64, 96, 320), device='cuda')
    f2 = torch.randn((12, 128, 48, 160), device='cuda')
    f3 = torch.randn((12, 216, 24, 80), device='cuda')
    f4 = torch.randn((12, 288, 12, 40), device='cuda')
    f5 = torch.randn((12, 288, 6, 20), device='cuda')
    feat = [f1, f2, f3, f4, f5]

    encoder.eval()
    flops, params = profile(encoder, inputs=(input,))
    print("FLOPs: ", flops / 1e9, "G", "Params: ", params / 1e6, "M")
    params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("Params: ", params / 1e6, "M")

    decoder.eval()
    flops, params = profile(decoder, inputs=(feat, "contrast", x0, input))
    print("FLOPs: ", flops / 1e9, "G", "Params: ", params / 1e6, "M")
    params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("Params: ", params / 1e6, "M")

    sys.exit()

#change
def check_val(opts, splits):
    #check whether the val dataset is correct
    import datasets
    from torch.utils.data import DataLoader
    wait2check = ['mix_snow', 'mix_rain', 'fog', 'snowgan', 'raingan']
    val_filenames = readlines("Your Project Path Here/splits/{}/val_files.txt".format(splits))
    val_dataset = datasets.KITTIRAWDataset(opts, val_filenames, is_train=False)
    val_loader = DataLoader(val_dataset, 12, False, num_workers=12, pin_memory=True, drop_last=False)
    load_val_mode = g.weatherList
    for ld_mode in load_val_mode:
        print("load mode: ", ld_mode)
        val_dataset.specify_data(ld_mode)
        for i, batch in enumerate(val_loader):
            print(i, ",", end="")
            if i % 50 == 1:
                print()
        print()

#chaneg
if __name__ == '__main__':
    opts = ini()
    if opts.easy_vis:
        easy_visualize(modify_opt(path='Your project path here/log_err'), 'val')

    else:
        check_val(opts, 'eigen_full')

