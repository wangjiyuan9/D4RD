# CopyRight wangjiyuan
# This file is used to store the basic global variables and options for the project.
from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
MIN_DEPTH = 1e-3
MAX_DEPTH = 80
defalut_height = 384
defalut_width = 1280

weatherList = ['rgb/data', 'raingan/data', 'fog/150m', 'snowgan/data', 'mix_rain/50mm', 'mix_snow/data', 'fog/75m']
blurList = ["brightness", "color_quant", "contrast", "dark", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise", "iso_noise", "jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"]
load_map = {'rgb/data': '晴', 'rain/50mm': '雨线', 'mix_rain/50mm': '雨', 'raingan/data': '雨境', 'image_02rain200': '极雨', 'fog/150m': '雾', 'fog/75m': '大雾', 'image_02fog50': '浓雾', 'average': '平均', 'variance': '方差', 'snowgan/data': '雪境', 'mix_snow/data': '雪', 'stereor': '真雨', 'stereof': '真雾', 'dense': '真雪', 'stereos': '真晴', 'stereoc': '真云'}
index_map = {0: 'abs_rel', 1: 'sq_rel', 2: 'rmse', 3: 'rmse_log', 4: 'a1', 5: 'a2', 6: 'a3', }
class_map = {0: ['rgb/data'], 1: ['rain/50mm', 'fog/150m', 'snowgan/data'], 2: ['mix_rain/50mm', 'fog/75m', 'mix_snow/data'], None: []}
test_map = ['stereor', 'stereof', 'stereos', 'stereoc', 'dense']  # 'cadc']
stereo_map = {'stereor': 'rainy', 'stereof': 'foggy', 'stereos': 'sunny', 'stereoc': 'cloudy'}
defalut_mixloss = True
defalut_denseaspp = True
defalut_planeres = True
defalut_flip = True
defalut_save = True


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Planedepth options")
        # my options
        self.parser.add_argument("--use_multi_gpu", "-umg", action="store_true", help="if set, use multi gpu for training")
        self.parser.add_argument("--org_pjct", help="if set 1,Predict the dep-org_pjct in the rain, and use the original image for photometric re-projection estimation", action="store_true", default=True)
        self.parser.add_argument("--mixRate", nargs="+", help='mix train mode, set to >=1 to use average mix stratge', type=float, default=[7])
        self.parser.add_argument("--vis_mode", type=int, help="the mode you want to vis,0 for not vis, 1 for seleced vis, 2 for all vis", default=0)
        self.parser.add_argument("--vis_what", '--vw', nargs="+", type=str, help="the folder you want to vis", default=["depth"])
        self.parser.add_argument("--weather", type=str, help="the weather you want to train", choices=["all", "clear", "rain", "fog", "snow", "robust"], default="all")
        self.parser.add_argument("--debug", help="0:no debug,1:output per epoch,2:output every image,3 highest debug,out put all", type=float, default=0)
        self.parser.add_argument("--scales", nargs="+", type=int, help="scales used in the loss", default=[0, 1, 2, 3])
        self.parser.add_argument("--loss", type=str, help="loss function", default="l1")
        self.parser.add_argument("--do_save", help="if set,save the model", action="store_true", default=defalut_save)
        self.parser.add_argument("--test_with_torch", '--twt', action="store_true", help="if set,use torch to test", default=True)
        self.parser.add_argument("--test_with_weather", '-tww', action="store_true", help="if set,use weather kitti to test")
        self.parser.add_argument("--local_rank", type=int, default=0)
        self.parser.add_argument("--cuda_devices", "--cuda", type=int, default=0)
        self.parser.add_argument("--use_diffusion", "--ud", action="store_true", help="if set,use diffusion")
        self.parser.add_argument("--use_teacher", "--ut", action="store_true", help="if set,use teacher model result to distill the student model result")
        self.parser.add_argument("--inference_steps", '-is', type=int, help="inference steps", default=20)
        self.parser.add_argument("--num_train_timesteps", '-ntt', type=int, help="num_train_timesteps", default=1000)
        self.parser.add_argument("--ddim_weight", type=float, help="the weight of diffusion loss, in paper, is L_{nis}, so it's 1", default=1)
        self.parser.add_argument("--no_ph", '-nph', action="store_true", help="if set,use no ph loss when training, also no l1 loss, because no inference when training")
        self.parser.add_argument("--teacher_weight", type=float, help="the weight of teacher distillation loss, in paper, is L_{dis}, so it's 1", default=1)
        self.parser.add_argument("--finetune", '-ft', action="store_true", help="if set,finetune the org robustdepth model,need to set the load_weight_model, but without the start epoch(0 is ok!)")
        self.parser.add_argument("--contrast_mode", '--cm', type=str, choices=["contrast", "trinity", "distill", "None"], default="None", help="for paper's 3 learning mothods")
        self.parser.add_argument("--condition_weight", type=float, default=0.5,help="the weight of condition loss, in paper, is \theta in equation 19")
        self.parser.add_argument("--dfs_after_sigmod", '--das', action="store_true", help="if set,use sigmod after diffusion, in paper is Sec 3.2 Outlier depth removal")
        self.parser.add_argument("--multi_scale", '-ms', action="store_true", help="if set,use multi scale")
        self.parser.add_argument("--extra_condition", '--ec', action="store_true", help="if set, use rgb and depth as condition, in paper is Sec 3.2 Feature-image joint condition")
        self.parser.add_argument('--teacher_loss', '--tl', action="store_true", help="if set, use distillation loss, in paper is L_{dis}")
        self.parser.add_argument('--delta_weight', type=float, help="delta_weight", default=0.01)
        self.parser.add_argument('--use_CNN', '-uC', action="store_true", help="if set, use robust CNN net to modify the condition, in paper, is Sec 3.4 Image level")
        self.parser.add_argument("--enhance_teacher", '--et', action="store_true", help="if set, use mask to enhance the teacher gt, in paper, is Sec 3.2 Pseudo-depth knowledge distillation")
        self.parser.add_argument("--note", type=str, help="add some note about this run", default="")
        self.parser.add_argument("--stage", type=int, help="the stage of the training, in paper, is Sec 3.5 Two stage training", default=0)
        self.parser.add_argument("--caulculate_computation", '-cc', action="store_true", help="if set, cauculate the computation of the model")

        # litemono
        self.parser.add_argument("--model", type=str, help="which model to load", default="lite-mono")
        self.parser.add_argument("--weight_decay", type=float, help="weight decay in AdamW", default=1e-2)
        self.parser.add_argument("--drop_path", type=float, help="drop path rate", default=0.2)
        self.parser.add_argument("--lr", nargs="+", type=float, help="learning rates of DepthNet and PoseNet. Initial learning rate, ""minimum learning rate, " "First cycle step size.", default=[0.0001, 5e-6, 31, 0.0001, 1e-5, 31])
        self.parser.add_argument("--mypretrain", type=str, help="if set, use my pretrained encoder", default="You project path/ckpt/lite-mono-pretrain.pth")

        # PATHS
        self.parser.add_argument("--data_path", type=str, help="path to the training data", default="Your data path")
        self.parser.add_argument("--log_dir", type=str, help="log directory", default="./log_err")

        # region TRAINING options
        self.parser.add_argument("--model_name", type=str, help="the name of the folder to save the model in", default="mdp")
        self.parser.add_argument("--split",
            type=str,
            help="which training split to use",
            default="eigen_zhou")
        self.parser.add_argument("--num_layers",
            type=int,
            help="number of resnet layers",
            default=50,
            choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
            type=str,
            help="dataset to train on",
            default="kitti",
            choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--height",
            type=int,
            help="input image height",
            default=192)
        self.parser.add_argument("--width",
            type=int,
            help="input image width",
            default=640)
        self.parser.add_argument("--smooth_weight",
            type=float,
            help="disparity smoothness weight",
            default=0.001)
        self.parser.add_argument("--novel_frame_ids", nargs="+", type=int, help="frames to load", default=[])
        self.parser.add_argument("--net_type", type=str, help="train which network", default="vit", )
        self.parser.add_argument("--use_ssim",
            help="if set, use ssim in the loss",
            action="store_true")
        # endregion

        # region OPTIMIZATION options
        self.parser.add_argument("--batch_size",
            type=int,
            help="batch size",
            default=8)
        self.parser.add_argument("--learning_rate",
            type=float,
            help="learning rate",
            default=1e-4)
        self.parser.add_argument("--beta_1",
            type=float,
            help="beta1 of Adam",
            default=0.5)
        self.parser.add_argument("--beta_2",
            type=float,
            help="beta2 of Adam",
            default=0.999)
        self.parser.add_argument("--start_epoch", "-se",
            type=int,
            help="epoch at start",
            default=0)
        self.parser.add_argument("--num_epochs",
            type=int,
            help="number of epochs",
            default=50)
        self.parser.add_argument('--milestones',
            default=[30, 40], nargs='*',
            help='epochs at which learning rate is divided by 2')
        self.parser.add_argument("--scheduler_step_size",
            nargs="+",
            type=int,
            help="step size of the scheduler",
            default=[20, 25, 29])
        self.parser.add_argument("--num_workers",
            type=int,
            help="number of dataloader workers",
            default=12)
        # endregion

        # LOADING options
        self.parser.add_argument("--load_weights_folder", "-lwf", type=str, help="name of model to load")
        self.parser.add_argument("--teacher_weights_folder", "-twf", type=str, help="path of teacher model to load")
        self.parser.add_argument("--log_frequency",
            type=int,
            help="number of batches between each tensorboard log",
            default=500)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
            help="if set evaluates in stereo mode",
            action="store_true")
        self.parser.add_argument("--eval_mono",
            help="if set evaluates in mono mode",
            action="store_true")
        self.parser.add_argument("--disable_median_scaling",
            help="if set disables median scaling in evaluation",
            action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
            help="if set multiplies predictions by this number",
            type=float,
            default=1)
        self.parser.add_argument("--eval_split",
            type=str,
            default="eigen_raw",
            help="which split to run eval on")
        self.parser.add_argument("--save_depth",
            help="if set saves depth predictions to npy",
            action="store_true")
        self.parser.add_argument("--post_process",
            help="if set will perform the flipping post processing from the original monodepth paper",
            action="store_true")
        self.parser.add_argument("--save_strategy",
            choices=["overwrite", "append"],
            default="overwrite",
            help="set to append if you want to continue save models"
        )
        # endregion

        # region visualization options
        self.parser.add_argument("--easy_vis", help="if set,merge the validation result tp one folder, with the need to compare", action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
