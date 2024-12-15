# coding=utf-8
# The full version of the trainer.py for Diffusion for Robust Depth(D4RD) based on MonoViT, Litemono. We additionally provide the ablation study part code: contrast mode(in paper table5, trinity,distill, etc.) KITTI-C training(in paper table3). The code is based on the original Monodepth2 codebase.
# Author: Jiyuan Wang
# Created: 2024-12-10
# Origin used for paper: https://arxiv.org/abs/2404.09831
# Hope you can cite our paper if you use the code for your research.
from __future__ import absolute_import, division, print_function
import os
import copy
import random
import time
import datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from tensorboardX import SummaryWriter
import json
import networks
from layers import *
from my_utils import *
import options as g
from Evaluate import *
from tqdm import tqdm, trange
import ssl
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

os.environ['MASTER_ADDR'] = 'localhost'
ssl._create_default_https_context = ssl._create_unverified_context


def init_seeds(seed=0, faster=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = faster


class Trainer:
    def __init__(self, options):
        # region pre-processing
        pid = os.getpid()
        print('pid: ', pid)
        self.opts = modify_opt(options)

        if self.opts.use_multi_gpu:
            dist.init_process_group(backend='nccl')
            self.local_rank = self.opts.local_rank
            self.opts.batch_size = self.opts.batch_size // torch.cuda.device_count()
            torch.cuda.set_device(self.local_rank)
            print('distributed init: rank {}'.format(self.opts.local_rank), flush=True)
            setup_for_distributed(self.opts.local_rank == 0)
            init_seeds(0 + self.opts.local_rank)
        else:
            init_seeds(0)

        self.device = torch.device("cuda")
        self.opts.device = self.device

        self.opts.scales = [0, 1, 2] if self.opts.net_type == "lite" else [0, 1, 2, 3]
        self.opts.flip_right = False
        self.num_scales = len(self.opts.scales)
        self.opts.novel_frame_ids = [-1, 1]
        self.num_pose_frames = 2
        self.opts.num_workers = self.opts.batch_size + 4
        self.opts.split = "eigen_zhou"
        print("Using SSIM loss", end=',')
        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.opts.model_name = createUniqueName(self.opts)
        self.log_path = os.path.join(self.opts.log_dir, self.opts.model_name)
        self.save_folder = os.path.join(self.log_path, "models", "weights_{}")

        assert self.opts.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opts.width % 32 == 0, "'width' must be a multiple of 32"

        if self.opts.flip_right:
            self.opts.batch_size = self.opts.batch_size // 2
        # For D4RD, will be [1/7,1/7,1/7,1/7,1/7,1/7,1/7] for 7 weather
        self.opts.mixRate = modify_rate(self.opts.mixRate, self.opts.weather)
        print("Mixing rate is: ", self.opts.mixRate)

        self.parameters_to_train, self.parameters_to_train_pose = [], []
        self.target_sides = ["r"]
        # endregion
        # region build network
        self.create_models()
        if options.use_teacher:
            twf = self.opts.teacher_weights_folder
            self.opts.teacher_weights_folder = "./ckpt/baseline" if twf is None else twf
            self.create_models('teacher')

        if self.opts.use_multi_gpu:
            for model_name, model in self.models.items():
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if self.opts.debug >= 1:
                    print("=>DistributedDataParallel for ", model_name, type(model))
                self.models[model_name] = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        if self.opts.net_type == "vit":
            self.params = [{"params": self.parameters_to_train, "lr": 1e-4}, {"params": list(self.models["encoder"].parameters()), "lr": 5e-5}]
            self.model_optimizer = optim.AdamW(self.params)
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, 0.9)
        else:
            self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opts.lr[0], weight_decay=self.opts.weight_decay)
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opts.lr[3], weight_decay=self.opts.weight_decay)
            #This is for the Litemono Scheduler
            self.model_lr_scheduler = ChainedScheduler(
                self.model_optimizer,
                T_0=int(self.opts.lr[2]),
                T_mul=1,
                eta_min=self.opts.lr[1],
                last_epoch=-1,
                max_lr=self.opts.lr[0],
                warmup_steps=0,
                gamma=0.9
            )
            self.model_pose_lr_scheduler = ChainedScheduler(
                self.model_pose_optimizer,
                T_0=int(self.opts.lr[5]),
                T_mul=1,
                eta_min=self.opts.lr[4],
                last_epoch=-1,
                max_lr=self.opts.lr[3],
                warmup_steps=0,
                gamma=0.9
            )
        self.model_prefix, self.models_to_load = self.prepare_model()
        if self.opts.load_weights_folder is not None or self.opts.teacher_weights_folder is not None or self.opts.finetune or self.opts.robust_weights_folder is not None:
            self.load_model()
        if self.opts.mypretrain is not None:
            self.load_pretrain()
        print("Training model named:  \033[91m", self.opts.model_name, "\033[0m")
        print("Models and tensorboard events files are saved to:  ", self.opts.log_dir)
        print("Training is using:  ", self.device)
        # endregion

        # region 建立数据集
        datasets_dict = {"kitti": datasets.KITTIRAWDataset}
        self.dataset = datasets_dict[self.opts.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "./splits", self.opts.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        if self.opts.debug > 0:
            train_filenames = train_filenames[:100] if self.opts.debug >= 1 else train_filenames[:1000]
            val_filenames = val_filenames[:40] if self.opts.debug >= 0.5 else val_filenames
            self.opts.num_workers = 0 if self.opts.debug >= 2 else self.opts.num_workers
            self.opts.num_epochs = min(10, self.opts.num_epochs) if (self.opts.debug >= 1 and self.opts.start_epoch == 0) else self.opts.num_epochs

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // (self.opts.batch_size * torch.cuda.device_count()) * (self.opts.num_epochs - self.opts.start_epoch)

        def worker_init(worker_id):
            worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_dataset = self.dataset(self.opts, train_filenames, is_train=True)
        if self.opts.use_multi_gpu:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            self.train_loader = DataLoader(self.train_dataset, self.opts.batch_size, False, num_workers=self.opts.num_workers, sampler=self.train_sampler, pin_memory=True, drop_last=True,
                worker_init_fn=worker_init, collate_fn=rmnone_collate)
        else:
            self.train_loader = DataLoader(self.train_dataset, self.opts.batch_size, False, num_workers=self.opts.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init,
                collate_fn=rmnone_collate)
        # only train use multi gpu;add the driving_stereo dataset, cadc dataset for test
        self.val_dataset = self.dataset(self.opts, val_filenames, is_train=False)
        self.val_loader = DataLoader(self.val_dataset, 20, False, num_workers=12, pin_memory=True, drop_last=False)
        # endregion
        # region 辅助函数
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opts.scales:
            h = int(self.opts.height // (2 ** scale))
            w = int(self.opts.width // (2 ** scale))
            self.backproject_depth[scale] = BackprojectDepth(self.opts.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opts.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        self.best_abs = 10.0
        self.berhu = BerhuLoss()
        if not self.opts.use_multi_gpu or dist.get_rank() == 0:
            self.create_summary_writer()

        self.val_gt_depths = prepare_gt_depths(self.opts, train_mode=True)
        print("√")  # endregion

    # 预创建函数
    def create_summary_writer(self):
        print("Using split:\n  ", self.opts.split)
        print("There are {:d} training items , {:d} validation items \n".format(len(self.train_dataset), len(self.val_dataset)))
        remove_logfolder(self.log_path, self.opts.save_strategy == "overwrite")
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_opts()

    def create_models(self, type='student'):
        '''
        Build the Student and Teacher models for training, support MonoViT, LiteMono.
        '''
        pre_map = {'teacher': 'teacher_', 'student': '', 'robust': 'robust_'}
        if type == 'student':
            print("=>Building network:")
            self.models = {}

        net_type = "vit" if type == "teacher" else self.opts.net_type
        pre = pre_map[type]
        print("==>build " + pre + net_type + " net")
        if net_type == "vit":
            self.models[pre + "encoder"] = networks.mpvit_small()
            self.models[pre + "encoder"].num_ch_enc = [64, 128, 216, 288, 288]

            self.models[pre + "depth"] = networks.HR_DepthDecoder(self.opts)
        elif net_type == "lite":
            self.models[pre + "encoder"] = networks.LiteMono(model=self.opts.model, drop_path_rate=self.opts.drop_path, width=self.opts.width, height=self.opts.height)
            self.models[pre + "depth"] = networks.DepthDecoderLite(self.models[pre + "encoder"].num_ch_enc, self.opts.scales, opts=self.opts)
        self.models[pre + "encoder"].to(self.device)
        self.models[pre + "depth"].to(self.device)

        if type == 'student':
            self.parameters_to_train += list(self.models["depth"].parameters())
            if net_type != "vit":
                self.parameters_to_train += list(self.models["encoder"].parameters())

            self.models["pose_encoder"] = networks.ResnetEncoder(18, True, num_input_images=self.num_pose_frames)
            self.models["pose_encoder"].to(self.device)

            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
            self.models["pose"].to(self.device)
            if net_type == "vit":
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                self.parameters_to_train += list(self.models["pose"].parameters())
            else:
                self.parameters_to_train_pose += list(self.models["pose_encoder"].parameters())
                self.parameters_to_train_pose += list(self.models["pose"].parameters())

    def set_train(self):
        self.models['encoder'].train()
        self.models['depth'].train()

    def set_eval(self):
        self.models['encoder'].eval()
        self.models['depth'].eval()

    def set_default(self):
        if self.opts.use_teacher:
            self.models['teacher_encoder'].eval()
            self.models['teacher_depth'].eval()
        self.models['pose_encoder'].train()
        self.models['pose'].train()

    def train(self):
        self.epoch, self.step = 0, 0
        self.start_time = time.time()
        for self.epoch in range(self.opts.start_epoch):
            self.model_lr_scheduler.step()
            self.model_pose_lr_scheduler.step()
        if self.opts.start_epoch != 0 and (not self.opts.use_multi_gpu or dist.get_rank() == 0):
            self.val()
            print("Each validation use time: ", sec_to_hm_str(time.time() - self.start_time))
            self.start_time = time.time()

        if self.opts.net_type == "vit":
            depth_lr = self.model_optimizer.param_groups[1]['lr']
            pose_lr = self.model_optimizer.param_groups[0]['lr']
            print(f'\nStarting from epoch {self.epoch} and current learning rate for depth is {depth_lr} and pose lr is {pose_lr}')

        print("==>Training started...")
        for self.epoch in range(self.opts.start_epoch, self.opts.num_epochs):
            self.run_epoch()
            if not self.opts.use_multi_gpu or dist.get_rank() == 0:
                if self.opts.do_save and self.epoch > 10:
                    self.save_model(str(self.epoch))
                else:
                    self.save_model("last")
        print("Training finished after {} epochs,".format(self.epoch))

    def run_epoch(self):
        if self.opts.use_multi_gpu:
            self.train_sampler.set_epoch(self.epoch)  # At the beginning of each epoch, the random seed of the data loader is reset to ensure that the order of data read by each process is different
        all_batches = len(self.train_loader)
        belogged_loss, record_loss = {}, {}
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            if inputs is None:
                self.model_optimizer.zero_grad()
                self.model_optimizer.step()
                self.step += 1
                continue
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            if self.opts.net_type == "lite":
                self.model_pose_optimizer.zero_grad()
            losses["loss/total_loss"].backward()
            self.model_optimizer.step()
            if self.opts.net_type == "lite":
                self.model_pose_optimizer.step()
            duration = time.time() - before_op_time

            # record loss
            for key, ipt in losses.items():
                if key.split('/')[1] not in record_loss:
                    record_loss[key.split('/')[1]] = 0
                record_loss[key.split('/')[1]] += ipt.item()
            # log with steps
            early_phase = batch_idx % 100 == 0 and self.step < self.opts.log_frequency
            late_phase = self.step % self.opts.log_frequency == 0

            if early_phase or late_phase:
                for key, ipt in losses.items():
                    belogged_loss[key.split('/')[1]] = ipt.item()
                self.log_time(batch_idx, duration, belogged_loss)
            self.step += 1

            # log with epoch
            if batch_idx == (all_batches - 2) and (not self.opts.use_multi_gpu or dist.get_rank() == 0):
                for key, ipt in record_loss.items():
                    self.writers["val"].add_scalar(key, ipt / (batch_idx + 1), self.epoch)
                self.log_img("train", inputs, outputs)
            if not self.opts.use_multi_gpu:
                del inputs, outputs, losses

        if not self.opts.use_multi_gpu or dist.get_rank() == 0:
            #we only do the validation on the main process
            with torch.no_grad():
                self.val()
        if self.opts.use_multi_gpu:
            dist.barrier()
        self.model_lr_scheduler.step()
        self.model_pose_lr_scheduler.step()

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        with torch.no_grad():
            if self.opts.stage == 1:
                self.opts.use_diffusion = False
            outputs_teacher = self.models["teacher_depth"](self.models["teacher_encoder"](inputs[("color", 0, 0)]), type="teacher", rgb=inputs[("color", 0, 0)]) if self.opts.use_teacher else None
            if self.opts.stage == 1:
                self.opts.use_diffusion = True

        inputs_color = torch.cat([inputs[("color_aug", 0, 0)], inputs[("color", 0, 0)]], dim=0) if self.opts.contrast_mode == "trinity" or self.opts.contrast_mode == "contrast" else inputs[("color_aug", 0, 0)]
        feat = self.models["encoder"](inputs_color)
        if self.opts.dfs_after_sigmod and self.opts.extra_condition:
            outputs = self.models["depth"](feat, type="student" if self.opts.contrast_mode == "None" or self.opts.contrast_mode == "distill" else "contrast", x0=outputs_teacher["x0"], rgb=inputs_color)
        else:
            outputs = self.models["depth"](feat, "student", outputs_teacher["x0"]) if self.opts.use_teacher else self.models["depth"](feat)

        outputs.update(self.predict_poses(inputs))
        if self.opts.use_teacher:
            outputs["teacher_disp", 0], outputs["teacher_condition"] = outputs_teacher["disp", 0], outputs_teacher["condition"]
        if self.opts.no_ph:# only use diffusion loss
            losses = {}
            losses["loss/ddim_loss"] = outputs["ddim_loss"]
            losses["loss/total_loss"] = losses["loss/ddim_loss"]
        else:
            self.pred_novel_images(inputs, outputs)
            losses = self.compute_losses(inputs, outputs)
        if self.opts.debug >= 2:
            print('\033[91m' + "loss:" + str(losses["loss/total_loss"].item())[:5] + 'ddim_loss:' + str(losses["loss/ddim_loss"].item())[:5] + '\033[0m')
        if self.opts.vis_mode:
            VisualizeDepth(None, outputs["depth", 0][0].cpu().detach().numpy(), "org", None)
            raise NotImplementedError("vis_mode!")
        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        frameIDs = self.opts.novel_frame_ids + [0]
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color", f_i, 0] for f_i in frameIDs}
            for f_i in frameIDs:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    outputs[("pose_feats", 0, f_i)] = pose_inputs

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch"""
        verbose = self.opts.debug
        if verbose == 1:
            print("Train √  Begin validation...")
        writer = self.writers["val"]
        cv2.setNumThreads(0)
        self.set_eval()
        test_encoder = self.models["encoder"] if not self.opts.use_multi_gpu else self.models["encoder"].module
        test_depth = self.models["depth"] if not self.opts.use_multi_gpu else self.models["depth"].module
        if verbose > 0:
            print_title(self.opts.model_name[:18])

        #######kitti########
        load_val_mode = g.weatherList
        load_val_mode = ['rgb/data'] if self.opts.weather == 'clear' else load_val_mode
        error_all = []
        for ld_mode in load_val_mode:
            self.val_dataset.specify_data(ld_mode)
            pbar = tqdm(self.val_loader, desc=ld_mode) if verbose > 0 else doNothing()
            pred_disps, _, _ = inference(self.opts, self.val_dataset, self.val_loader, test_encoder, test_depth, pbar)
            errors, ratios = evaluate(self.opts, pred_disps, None, self.val_gt_depths, ld_mode, train_mode=True, train_opt={'epoch': self.epoch, 'writer': writer})
            mean_errors = np.array(errors).mean(0)
            error_all.append(mean_errors)

            for ind, error in enumerate(mean_errors):
                writer.add_scalar('{}/{}'.format(g.load_map[ld_mode], g.index_map[ind]), error, self.epoch)
            if verbose > 0:
                print_errors(mean_errors, ld_mode if self.opts.weather == "robust" else g.load_map[ld_mode])
            pbar.close()
        # region Average, Variance of Error
        mean_errors = np.array(error_all).mean(0)
        var_errors = np.array(error_all).var(0)
        current_abs = mean_errors[0]
        if current_abs < self.best_abs:
            self.best_abs = current_abs
            self.best_epoch = self.epoch
            self.save_model('best')
        for ind, error in enumerate(mean_errors):
            writer.add_scalar('{}/{}'.format(g.load_map["average"], g.index_map[ind]), error, self.epoch)
            writer.add_scalar('{}/{}'.format(g.load_map["variance"], g.index_map[ind]), var_errors[ind], self.epoch)
        if verbose > 0:
            print_errors(mean_errors, g.load_map["average"])
            print_errors(var_errors, g.load_map["variance"])
            print("KITTI time:", time.time() - self.start_time)
        # endregion
        self.set_train()

    def pred_novel_images(self, inputs, outputs):
        """为小批量生成扭曲（重新投影）的彩色图像。生成的图像将保存到“输出”字典中"""
        if self.opts.use_diffusion and not self.opts.multi_scale:
            self.opts.scales = [0]
        pre_map = ['']
        pre_map = pre_map if not self.opts.enhance_teacher else ['', 'teacher_']
        source_scale = 0

        for pre in pre_map:
            for scale in self.opts.scales:
                disp = outputs[(pre + "disp", scale)]
                disp = F.interpolate(disp, [self.opts.height, self.opts.width], mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp, 0.1, 100)
                for i, frame_id in enumerate(self.opts.novel_frame_ids):
                    if frame_id == "s":
                        T = inputs["stereo_T"]
                    else:
                        T = outputs[("cam_T_cam", 0, frame_id)]
                    cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                    outputs[("sample", frame_id, scale)] = pix_coords
                    outputs[(pre + "color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)], outputs[("sample", frame_id, scale)], padding_mode="border",
                        align_corners=True)  #use clear image to reprojection

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss


    def compute_supervised_loss(self, pred, target, valid_pixels=None, loss_type=None):
        """ Calculate the supervision loss (L_{dis} in paper). - valid_pixels Mask of a valid depth-cueping pixel (i.e., non-zero depth value)"""
        if valid_pixels is None:
            valid_pixels = torch.ones(target.shape, device=self.device)
        if loss_type is None:
            loss_type = self.opts.loss
        if loss_type == 'log':
            loss = torch.log(torch.abs(target - pred) + 1) * valid_pixels
        elif loss_type == 'l1':
            loss = F.smooth_l1_loss(pred, target, reduction='none') * valid_pixels
        elif loss_type == 'berhu':
            loss = self.berhu(pred, target) * valid_pixels
        elif loss_type == 'kldiv':
            average = target.log()
            loss = F.kl_div(average, pred, reduction='none') * valid_pixels
        loss = loss.sum() / (valid_pixels.sum() + 1e-7)
        return loss

    def compute_mask(self, inputs, outputs):
        clear_rep, _ = self.compute_repro_irepo(inputs, outputs, 0, pre='teacher_')
        mask = (clear_rep < (1.5 / (self.epoch + 1))).float()
        return mask

    def compute_repro_irepo(self, inputs, outputs, scale, pre=''):
        #we Refactor the Reprojection part code into a separate function
        reprojection_losses, identity_reprojection_losses = [], []
        target = inputs[("color", 0, 0)]
        for frame_id in self.opts.novel_frame_ids:
            pred = outputs[(pre + "color", frame_id, scale)]  # This is the image after warp, the size is the same, this is the latitude of the same size as the image
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))
        reprojection_losses = torch.cat(reprojection_losses, 1)  # The dimension after cat is [batch_2, 2, size], and before it is [batch_2, 1, size]
        reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
        identity_reprojection_loss = None
        if pre == '':
            for frame_id in self.opts.novel_frame_ids:
                pred = inputs[("color", frame_id, 0)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001
        return reprojection_loss, identity_reprojection_loss

    def compute_losses(self, inputs, outputs, mode='train'):
        losses = {}
        total_loss, condition_smooth_loss = 0, 0
        if self.opts.net_type != 'plane' and self.opts.use_diffusion:
            mask = self.compute_mask(inputs, outputs) if self.opts.enhance_teacher else None
            if self.opts.use_diffusion:
                losses['loss/ddim_loss'] = outputs['ddim_loss'] * self.opts.ddim_weight
                total_loss += losses['loss/ddim_loss']
            if self.opts.teacher_loss:
                losses['loss/teacher_loss'] = self.compute_supervised_loss(outputs['teacher_disp', 0], outputs['disp', 0], mask) * self.opts.teacher_weight
                total_loss += losses['loss/teacher_loss']
            if self.opts.contrast_mode == "trinity" and self.opts.stage == 2:
                cst_loss = self.compute_supervised_loss(outputs['condition'], outputs["contrast_condition"], loss_type='l1') + 0.5 * self.compute_supervised_loss(outputs['condition'],
                    outputs["teacher_condition"], loss_type='l1') + 0.5 * self.compute_supervised_loss(outputs['contrast_condition'], outputs["teacher_condition"], loss_type='l1')
                delta_loss = outputs['delta_loss'] * self.opts.delta_weight if self.opts.use_CNN else 0
                losses['loss/cst_loss'] = delta_loss + cst_loss * self.opts.condition_weight
                condition_smooth_loss = get_smooth_loss(outputs['condition'], inputs[("color", 0, 0)])
                total_loss += losses['loss/cst_loss']
            elif self.opts.contrast_mode == "contrast" and self.opts.stage == 2:
                cst_loss = self.compute_supervised_loss(outputs['contrast_condition'], outputs["condition"], loss_type='l1') * self.opts.condition_weight
                delta_loss = outputs['delta_loss'] * self.opts.delta_weight if self.opts.use_CNN else 0
                losses['loss/cst_loss'] = delta_loss + cst_loss
                total_loss += losses['loss/cst_loss']
            elif self.opts.contrast_mode == "distill" and self.opts.stage == 2:
                cst_loss = self.compute_supervised_loss(outputs['teacher_condition'], outputs["condition"], loss_type='l1') * self.opts.condition_weight
                losses['loss/cst_loss'] = cst_loss
                total_loss += losses['loss/cst_loss']

        scale = 0  # todo Note that this should be changed when there is more scale in the future
        disp = outputs[("disp", scale)]
        color = inputs[("color", 0, scale)]
        reprojection_loss, identity_reprojection_loss = self.compute_repro_irepo(inputs, outputs, scale)
        to_optimise, idxs = torch.min(torch.cat((identity_reprojection_loss, reprojection_loss), dim=1), dim=1)
        losses["loss/ph_loss"] = to_optimise.mean()
        total_loss += losses["loss/ph_loss"]

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        losses["loss/adjust_loss"] = self.opts.smooth_weight * (smooth_loss  + condition_smooth_loss) / (2 ** scale)
        total_loss += losses["loss/adjust_loss"]
        losses["loss/total_loss"] = total_loss
        return losses


    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal """
        samples_per_sec = self.opts.batch_size * torch.cuda.device_count() / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} |".format(self.epoch, batch_idx, samples_per_sec)
        for key in losses:
            print_string += " {}:{:.5f} |".format(key, losses[key])
        print_string += " time elapsed: {} | time left: {}".format(sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left))

        print(print_string)

    def log_img(self, mode, inputs, outputs):
        """Write an event to the tensorboard events file"""
        writer = self.writers[mode]
        j, scale = 0, 0
        for frame_id in (self.opts.novel_frame_ids + [0]):
            writer.add_image("color/{}_{}".format(frame_id, 0), inputs[("color", frame_id, 0)][j].data, self.epoch)
            try:
                writer.add_image("color_pred/{}_{}".format(frame_id, 0), outputs[("color", frame_id, 0)][j].data, self.epoch)
            except KeyError:
                pass
            if frame_id == 0:
                writer.add_image("color_weather/{}_{}".format(frame_id, 1), inputs[("color_aug", frame_id, 0)][j].data, self.epoch)

        scale = 0
        if not self.opts.no_ph:
            try:
                writer.add_image("disp_{}/{}".format(scale, j), VisualizeMap(outputs["disp", 0][j].data, doTrans=True), self.epoch)
                writer.add_image("disp_teacher/{}".format(j), VisualizeMap(outputs["teacher_disp", 0][j].data, doTrans=True), self.epoch)
                writer.add_image("disp_robust/{}".format(j), VisualizeMap(outputs["robust_disp", 0][j].data, doTrans=True), self.epoch)
            except KeyError:
                pass
        if self.opts.use_diffusion:
            try:
                writer.add_image("condition/{}".format(j), VisualizeMap(outputs["condition"][j].data[0], doTrans=True), self.epoch)
                writer.add_image("contrast_condition/{}".format(j), VisualizeMap(outputs["contrast_condition"][j].data[0], doTrans=True), self.epoch)
                writer.add_image("teacher_condition/{}".format(j), VisualizeMap(outputs["teacher_condition"][j].data[0], doTrans=True), self.epoch)
            except KeyError:
                pass
            except TypeError:
                pass


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with  """
        models_dir = self.log_path
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.opts.device = str(self.device)
        to_save = self.opts.__dict__.copy()
        with open(os.path.join(models_dir, 'opts.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, folder_name):
        """Save model weights to disk   """
        save_folder = os.path.join(self.log_path, folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            if 'teacher' in model_name or 'robust' in model_name:
                continue
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            if self.opts.use_multi_gpu:
                to_save = model.module.state_dict()
            else:
                to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opts.height
                to_save['width'] = self.opts.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def prepare_model(self):
        prefix = []
        models_to_load = {}
        if self.opts.teacher_weights_folder is not None:
            self.opts.teacher_weights_folder = os.path.expanduser(self.opts.teacher_weights_folder)
            assert os.path.isdir(self.opts.teacher_weights_folder), "Cannot find folder {}".format(self.opts.teacher_weights_folder)
            prefix.append('teacher_')
            models_to_load['teacher_'] = ['encoder', 'depth']
            print("==>loading teacher model from folder {}".format(self.opts.teacher_weights_folder))

        if self.opts.load_weights_folder is not None:
            self.opts.load_weights_folder = os.path.expanduser(self.opts.load_weights_folder)
            assert os.path.isdir(self.opts.load_weights_folder), "Cannot find folder {}".format(self.opts.load_weights_folder)
            prefix.append('')
            models_to_load[''] = ['encoder', 'depth', 'pose_encoder', 'pose']
            print("==>loading model from folder {}".format(self.opts.load_weights_folder))

        if self.opts.robust_weights_folder is not None:
            self.opts.robust_weights_folder = os.path.expanduser(self.opts.robust_weights_folder)
            assert os.path.isdir(self.opts.robust_weights_folder), "Cannot find folder {}".format(self.opts.robust_weights_folder)
            prefix.append('robust_')
            models_to_load['robust_'] = ['encoder', 'depth']
            print("==>loading model from folder {}".format(self.opts.robust_weights_folder))
        return prefix, models_to_load

    def load_pretrain(self):
        #This is just for lite-mono baseline.
        self.opts.mypretrain = os.path.expanduser(self.opts.mypretrain)
        path = self.opts.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self, ):
        """Load model(s) from disk """
        prefix, models_to_load = self.model_prefix, self.models_to_load
        for pre in prefix:
            print("=>loading {} model weights".format(pre))
            for n in models_to_load[pre]:
                n_bkup = n
                n = pre + n
                base = self.opts.teacher_weights_folder if pre != '' else self.opts.load_weights_folder
                print("==>Loading {} weights...".format(n), end=" ")
                path = os.path.join(base, "{}.pth".format(n_bkup))
                model = self.models[n] if not self.opts.use_multi_gpu else self.models[n].module
                model_dict = model.state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                if pre != '':
                    for param in self.models[n].parameters():
                        param.requires_grad = False
                print("√")

        # loading adam state
        load_adam = (self.opts.load_weights_folder is not None and not self.opts.finetune)
        if load_adam:
            optimizer_load_path = os.path.join(self.opts.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights...", end=" ")
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                self.model_optimizer.load_state_dict(optimizer_dict)
                print("√")
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")
