# Core code for training the D4RD model, enhanced from the Monodiffusion codebase.
# Created: 2023-10-5
# Origin used for paper: https://arxiv.org/abs/2404.09831
# Hope you can cite our paper if you use the code for your research.
import os
from typing import Union, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn

from .scheduling_ddim import DDIMScheduler
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import ConvModule


class DiffusionFrameWork(nn.Module):
    def __init__(self, opts, channelsUIO):
        super(DiffusionFrameWork, self).__init__()
        self.opts = opts
        try:
            inference_steps = self.opts.inference_steps
            num_train_timesteps = self.opts.num_train_timesteps
        except:
            print("there is no prefix inference steps loaded, make it to defualt 20/1000")
            inference_steps = 20
            num_train_timesteps = 1000
        channelsUmiddle = 16 if not opts.extra_condition else 19
        self.model = ScheduledCNNRefine(channelsUmiddle=channelsUmiddle, channelsUIO=channelsUIO, opts=opts)
        self.diffusion_inference_steps = inference_steps
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        self.scheduler.set_timesteps(inference_steps, device=opts.device)
        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
        self.no_ph_loss = opts.no_ph
        self.sigmoid = nn.Sigmoid()

    def forward(self, condition, x0, isTrain=True, condition_cst=None):
        output = {}
        refined_depth, ddim_loss = None, None
        if not self.no_ph_loss or not isTrain:
            refined_depth, refined_depth_list = self.pipeline(condition, self.diffusion_inference_steps, x0.shape)
        if isTrain:
            ddim_loss = self.ddim_loss(condition, condition_cst, x0)
        output[("disp", 0)] = refined_depth if not self.opts.dfs_after_sigmod else self.sigmoid(refined_depth)
        output["ddim_loss"] = ddim_loss
        output["x0"] = refined_depth
        return output

    def ddim_loss(self, cdt, cdt_cst, x0):
        # Sample noise to add to the images
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype)
        bs = x0.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=cdt.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(x0, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps, cdt)

        if self.opts.contrast_mode == "trinity":
            #in paper Equation 14
            noise_pred_cst = self.model(noisy_images, timesteps, cdt_cst)
            loss = F.mse_loss(noise_pred, noise) + 0.5 * F.mse_loss(noise_pred_cst, noise_pred) + F.mse_loss(noise_pred_cst, noise)
        elif self.opts.contrast_mode == "contrast":
            noise_pred_cst = self.model(noisy_images, timesteps, cdt_cst)
            loss = F.mse_loss(noise_pred_cst, noise_pred)
        else:
            loss = F.mse_loss(noise_pred, noise)
        return loss


class CNNDDIMPipiline:
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(self, condition, num_inference_steps, shape, generator: Optional[torch.Generator] = None, eta: float = 0.0) -> Union[Dict, Tuple]:
        # call at 35 line
        device = condition.device
        disp = torch.randn(shape, device=device, dtype=condition.dtype, generator=generator)
        disp_list = []
        for t in self.scheduler.timesteps:
            # 1. predict noise
            predict_noise = self.model(disp, t, condition)
            # 2. predict previous mean of disp x_t-1 and add variance depending on eta, eta corresponds to η in paper and should be between [0, 1] ,do x_t -> x_t-1
            disp = self.scheduler.step(predict_noise, t, disp, use_clipped_model_output=True)['prev_sample']
            disp_list.append(disp)

        return (disp, disp_list)


class UpSample_add(nn.Sequential):
    def __init__(self, skip_input, output_disp, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample_add, self).__init__()
        self.convA = ConvModule(skip_input, output_disp, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_disp, output_disp, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, cdt, preDisp):
        cdt = F.interpolate(cdt, size=[preDisp.size(2), preDisp.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(cdt + preDisp))


class ScheduledCNNRefine(BaseModule):
    def __init__(self, channelsUmiddle, channelsUIO, opts, **kwargs):
        super().__init__(**kwargs)

        self.opts = opts
        self.noise_embedding = nn.Conv2d(channelsUIO, channelsUmiddle, kernel_size=3, stride=1, padding=1)
        self.upsample_fuse = UpSample_add(channelsUmiddle, channelsUmiddle)
        self.time_embedding = nn.Embedding(1280, channelsUmiddle)
        self.adoptCNN = nn.Conv2d(4, channelsUmiddle, kernel_size=1)
        self.pred = nn.Sequential(
            # |condition:cdt,t,x_t|→noise
            nn.Conv2d(channelsUmiddle, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, channelsUIO, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, xt, t, condition):
        # call at 62 line
        # the input shape of xt is [B, 3, H, W]
        condition = self.adoptCNN(condition) if self.opts.extra_condition else condition
        conditionT = condition + self.time_embedding(t)[..., None, None]
        conditionTN = conditionT + self.noise_embedding(xt) if not self.C else self.upsample_fuse(conditionT, self.noise_embedding(xt))
        # cdtnon = torch.isnan(condition).any().item()
        # if cdtnon:
        #     print(".", end='')
        noise = self.pred(conditionTN)
        # if not cdtnon and torch.isnan(condition).any().item():
        #     print("*", end='')
        # nan_mask = torch.isnan(noise)
        # noise[nan_mask] = 0
        return noise
