# Code for evaluating the monocular depth estimation model, enhanced from the original Monodepth2 codebase.
# Enhanced details:
# The code can be used to evaluate the model on the KITTI, DrivingStereo(sunny, cloudy, rainy, foggy), CADC datasets, KITTI-C datasets and Dense Dataset. You can easily add your model at the func: create_dataset
# The code supports visulization of the feature maps and the input images. You can easily add your model at the func: inference; Like the paper Figure4.
# The code use cuda to further speed up the evaluation process.
# Author: Jiyuan Wang
# Created: 2023-10-05
# Origin used for paper: https://arxiv.org/abs/2310.05556v2
# Hope you can cite our paper if you use the code for your research.
from __future__ import absolute_import, division, print_function
import os, sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
import time
from options import *

options = MonodepthOptions()
opts = options.parse()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.cuda_devices)
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm, trange
from my_utils import *
import datasets
import networks

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "./splits")
STEREO_SCALE_FACTOR = 5.4
hard_map = [357, 358, 306, 307, 224, 42, 249, 310, 421, 422]
vis_select = 159
batch_size = 6


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_errors_torch(gt, pred):
    """Computation of error metrics between predicted and ground truth depths in cuda
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity_torch(l_disp, r_disp):
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = torch.meshgrid(torch.linspace(0, 1, w, device=l_disp.device), torch.linspace(0, 1, h, device=l_disp.device))
    l = torch.transpose(l, 0, 1)
    l_mask = (1.0 - torch.clamp(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = torch.flip(l_mask, dims=[2])
    return m_disp  # r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return m_disp  # r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def create_dataset(opt):
    if 'stereo' not in opt.eval_split:
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        filenames = filenames[:40] if opt.debug >= 1 else filenames
    else:
        opt.data_path = opt.data_path.replace("kitti", "drivingstereo")
        filenames = readlines(os.path.join(splits_dir, opt.eval_split[:6], stereo_map[opt.eval_split] + '.txt'))
        filenames = filenames[:10] if opt.debug >= 1 else filenames
        dataset = datasets.DrivingStereoDataset(opt, filenames, is_train=False)
        dataset.specify_data(stereo_map[opt.eval_split])


    if opt.eval_split == 'cadc':
        opt.data_path = opt.data_path.replace("kitti", "cadcd")
        dataset = datasets.CADCDataset(opt, filenames, is_train=False)
    elif opt.eval_split == 'dense':
        opt.data_path = opt.data_path.replace("kitti", "dense")
        dataset = datasets.DenseDataset(opt, filenames, is_train=False)
    elif 'eigen' in opt.eval_split:
        dataset = datasets.KITTIRAWDataset(opt, filenames, is_train=False, )
        dataset.specify_data(class_map[0][0])
    opt.num_workers = 0 if opt.debug >= 1 else opt.batch_size + 4
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    return dataset, dataloader, opt


def create_model(opt):
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location='cuda:0')
    encoder = networks.mpvit_small()
    encoder.num_ch_enc = [64, 128, 216, 288, 288]
    depth_decoder = networks.HR_DepthDecoder(opt)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))

    encoder.cuda()
    encoder.eval()

    depth_decoder.cuda()
    depth_decoder.eval()
    return encoder, depth_decoder


def inference(opt, dataset, dataloader, encoder, depth_decoder, pbar=None):
    before = time.time()
    record = {} if opt.vis_mode else None
    for vis in opt.vis_what:
        if vis == "feat":
            record["before-feat"] = torch.zeros((len(dataset), opt.height, opt.width))
            record["after-feat"] = torch.zeros((len(dataset), opt.height, opt.width))
            record["after-img"] = torch.zeros((len(dataset), 3, opt.height, opt.width))
        elif vis == "rgb":
            record["image"] = torch.zeros((len(dataset), 3, opt.height, opt.width))

    if opt.test_with_torch:
        pred_disps = torch.zeros((len(dataset), opt.height, opt.width), device='cuda')
    else:
        pred_disps = []
    start_idx, names = 0, []
    # Start inference
    with torch.no_grad():
        for data in dataloader:
            # region data preparation
            input_color = data[("color", 0, 0)]
            end_idx = start_idx + input_color.shape[0]
            input_color = input_color.cuda()
            if 'stereo' in opt.eval_split:
                names.append(data["name"])
            if opt.post_process:
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            # endregion

            # region forward
            output = depth_decoder(encoder(input_color), type="test" if opt.use_diffusion else 'teacher') if not opt.extra_condition else depth_decoder(encoder(input_color), "test", rgb=input_color)
            pred_disp, _ = disp_to_depth(output['disp', 0], 0.1, 80)
            pred_disp = pred_disp[:, 0]

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity_torch(pred_disp[:N], torch.flip(pred_disp[N:], dims=[2])) if opt.test_with_torch else batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
            for vis in opt.vis_what:
                if vis == "feat":
                    record["before-feat"][start_idx:end_idx] = output['beforeCNN'].squeeze(1)
                    record["after-feat"][start_idx:end_idx] = output['afterCNN'][:, 0, :, :]
                    record["after-img"][start_idx:end_idx] = output['afterCNN'][:, 1:, :, :]
                elif vis == "rgb":
                    record["image"][start_idx:end_idx] = input_color
            if opt.test_with_torch:
                pred_disps[start_idx:end_idx] = pred_disp
            else:
                pred_disps.append(pred_disp)
            start_idx = end_idx
            pbar.update(1)
            # endregion

    if opt.test_with_torch:
        pred_disps = pred_disps.cpu().numpy()
    pred_disps = np.concatenate(pred_disps) if not opt.test_with_torch else pred_disps
    return pred_disps, names, record


from multiprocessing import Pool


def evaluate(opt, pred_disps, names, gt_depths, mode=None, train_mode=False, train_opt=None, record=None):
    errors = []
    ratios = []
    with Pool(16) as pool:
        for i in range(pred_disps.shape[0]):
            # region Load single GT
            if 'stereo' in opt.eval_split:
                depth_path = os.path.join(opt.data_path, names[i // batch_size][i % batch_size]).replace("left-image", "depth-map")
                if opt.debug >= 3:
                    print(depth_path)
                depth_png = np.array(Image.open(depth_path), dtype=int)
                # make sure we have a proper 16bit depth map here not 8bit!
                assert (np.max(depth_png) > 255)
                gt_depth = depth_png.astype(np.float32) / 256  # gt_depth = gt_depth[250:800, :]
            elif opt.eval_split == 'cadc':
                gt_depth = gt_depths[i][234:774, 0:1280]
            elif opt.eval_split == 'dense':
                gt_depth = gt_depths[i][50:(1024 - 50), 20:(1920 - 20)]
            else:
                gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            # endregion
            gt_depth_org = gt_depth
            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp
            pred_depth_org = pred_depth

            if "test" in opt.eval_split:
                continue
            if "eigen" in opt.eval_split:
                gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
                gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(gt_depth.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                mask = gt_depth > 0
                #for other datase, we all just use gt_depth > 0 as the mask

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            if not opt.disable_median_scaling or opt.net_type == 'vit':
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            if train_mode:
                if i == 10:
                    if train_opt['epoch'] == 0 and mode == 'rgb/data':
                        VisualizeDepth(None, gt_depth_org, 'val_gt_depth', process='None', writer=train_opt['writer'])
                    VisualizeDepth(None, pred_depth_org * ratio, 'val_pred_depth/{}-{}'.format(mode, i), process='None', writer=train_opt['writer'], rcd=train_opt['epoch'], save_as='depth')

            if ((i in hard_map and opt.vis_mode == 1) or opt.vis_mode == 2) and not train_mode:
                pool.apply_async(VisualizeAll, (opt, pred_depth_org * ratio, gt_depth_org, mask, mode, i, record))

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            err = compute_errors(gt_depth, pred_depth)
            # print_errors(np.array(err), str(i) + mode, type='markdown')#You can open it to show each image's error
            errors.append(err)
        pool.close()
        pool.join()
    return errors, ratios


def print_errors(errors, name, type='latex'):
    if type == 'latex':
        print(("{:>20}").format(name), end='')
        print(("&{:10.3f}" * 7).format(*errors.tolist()) + "\\\\")
    elif type == 'markdown':
        print(("|{:>20}").format(name), end='')
        print(("|{:10.3f}" * 7).format(*errors.tolist()) + "|")


def print_title(name):
    print(("{:>20}").format(name), end='')
    print(("&{:>10}" * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\\\\")


def print_ratio(ratio_all):
    for ratios in ratio_all:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))


def prepare_gt_depths(opt=None, train_mode=False):
    if not train_mode:
        gt_depths = np.load(modify_opt(path=os.path.join(opt.data_path, "gt_depths.npy")), allow_pickle=True)
        print("-> Evaluating")
        return gt_depths
    elif train_mode:
        print("Loading ground truth depths...", end=' ')
        val_gt_path = modify_opt(path="Put you validation gt path here, if you don't want to test the depth with gt, you can also use the monodepth2 way with ph loss")
        val_gt_depths = np.load(val_gt_path, allow_pickle=True)
        return val_gt_depths


def evaluate_all(opt):
    """Evaluates a pretrained model using a specified test set"""
    # region preparation
    opt = modify_opt(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_devices)
    opt.device = torch.device("cuda")
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))

    dataset, dataloader, opt = create_dataset(opt)
    encoder, depth_decoder = create_model(opt)

    gt_depths = prepare_gt_depths(opt)
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))
    # endregion

    start = time.time()
    print_title('condition')
    error_all, ratio_all = [], []
    if opt.test_with_weather:
        #test for weatherKITTI
        load_val_mode = g.weatherList
        for mode in load_val_mode:
            dataset.specify_data(mode)
            pbar = tqdm(dataloader, desc=mode)
            pred_disps, names, visual = inference(opt, dataset, dataloader, encoder, depth_decoder, pbar=pbar)
            errors, ratios = evaluate(opt, pred_disps, names, gt_depths, mode, record=visual)
            mean_errors = np.array(errors).mean(0)
            pbar.close()
            print_errors(mean_errors, mode)
            error_all.append(mean_errors)
            ratio_all.append(ratios)
    elif opt.test_with_robust:
        #test for KITTI-C
        blur, blur_intensity = g.blurList, ["1", "2", "3", "4", "5"]
        for mode in blur:
            for intensity in blur_intensity:
                dataset.specify_data(mode + "/" + intensity)
                pbar = tqdm(dataloader, desc=mode + intensity)
                pred_disps, names, visual = inference(opt, dataset, dataloader, encoder, depth_decoder, pbar=pbar)
                errors, ratios = evaluate(opt, pred_disps, names, gt_depths, mode, record=visual)
                mean_errors = np.array(errors).mean(0)
                pbar.close()
                print_errors(mean_errors, mode + intensity)
                error_all.append(mean_errors)
                ratio_all.append(ratios)
    else:
        #test for other real adverse dataset
        pbar = tqdm(dataloader, desc="eval")
        calculate_parameters(opt, encoder, depth_decoder)
        pred_disps, names, visual = inference(opt, dataset, dataloader, encoder, depth_decoder, pbar=pbar)
        errors, ratios = evaluate(opt, pred_disps, names, gt_depths, opt.eval_split, record=visual)
        pbar.close()
        mean_errors = np.array(errors).mean(0)
        print_errors(mean_errors, opt.eval_split)

    mean_errors = np.array(error_all).mean(0)
    print_errors(mean_errors, 'average')
    print_ratio(ratio_all)
    print("Time:", time.time() - start)


if __name__ == "__main__":
    if opts.test_with_robust:
        opts.weather = 'robust'
    evaluate_all(opts)
