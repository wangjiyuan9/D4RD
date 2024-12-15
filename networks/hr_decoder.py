# Code for training the D4RD model, enhanced from the original MonoViT codebase.
# Created: 2023-10-5
# Origin used for paper: https://arxiv.org/abs/2404.09831
# Hope you can cite our paper if you use the code for your research. We mark '#change' at what we have changed from MonoViT origin.
from __future__ import absolute_import, division, print_function

import torch

from .hr_layers import *
from .diffusion_framework import DiffusionFrameWork


class HR_DepthDecoder(nn.Module):
    def __init__(self, opts, ch_enc=[64, 128, 216, 288, 288], scales=range(4), num_ch_enc=[64, 64, 128, 256, 512], num_output_channels=1):
        super(HR_DepthDecoder, self).__init__()
        self.opts = opts
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()

        # decoder
        self.convs = nn.ModuleDict()

        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4], num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3], num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2], num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1], num_ch_enc[1])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_" + index + "_downsample"] = Conv1x1(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.sigmoid = nn.Sigmoid()

        # change
        self.diffusion = DiffusionFrameWork(opts, 1) if opts.use_diffusion else None
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.robustnet = RobustUnet(opts, 4) if opts.use_CNN else None

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features, type="teacher", x0=None, rgb=None):
        '''
        type: teacher/student/contrast/test, teacher is for 2nd stage training, in paper Figure2 F_T, student is for both 1st and 2nd stages training, in paper Figure2 F_S, contrast is for Trinity/Contrastive Learning, in paper Figure2 (c), test is for evaluation
        x0: the input of the diffusion framework, in paper Figure2 d_0
        rgb: the input of the robustnet, in paper Figure2 I
        '''
        outputs = {}
        feat = {}
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = input_features[0]

        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_" + index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](features["X_{}{}".format(row + 1, col - 1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row + 1, col - 1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        # change
        if self.opts.use_diffusion:
            condition = self.convs["dispconv0"](x)
            condition_org = condition
            if self.opts.multi_scale and type == "student":
                outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features["X_04"]))
                outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features["X_13"]))
                outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))

            if type == "teacher":
                condition = condition if not self.opts.extra_condition else torch.cat([condition, rgb], dim=1)
                outputs = self.diffusion(condition=condition, isTrain=False, x0=self.convs["dispconv0"](x))
                outputs["condition"] = condition if self.opts.use_CNN else condition_org
            elif type == "student":
                condition = condition if not self.opts.extra_condition else torch.cat([condition, rgb], dim=1)
                if self.opts.use_CNN:
                    condition = condition + self.robustnet(condition)
                outputs.update(self.diffusion(condition=condition, x0=x0))
                outputs["condition"] = condition
            elif type == "contrast":
                ##Main##
                # the implement of paper Figure 2, the second row.
                condition_cst = condition[condition.shape[0] // 2:] if not self.opts.extra_condition else torch.cat([condition[condition.shape[0] // 2:], rgb[rgb.shape[0] // 2:]], dim=1)
                condition = condition[:condition.shape[0] // 2] if not self.opts.extra_condition else torch.cat([condition[:condition.shape[0] // 2], rgb[:rgb.shape[0] // 2]], dim=1)
                if self.opts.use_CNN:
                    condition = condition + self.robustnet(condition)
                    delta = self.robustnet(condition_cst)
                    condition_cst = condition_cst + delta
                outputs = self.diffusion(condition=condition, x0=x0, condition_cst=condition_cst)
                if self.opts.use_CNN:
                    outputs["delta_loss"] = abs(delta).mean()
                    outputs["condition"] = condition
                    outputs["contrast_condition"] = condition_cst
                else:
                    outputs["condition"] = condition_org[:condition_org.shape[0] // 2]
                    outputs["contrast_condition"] = condition_org[condition_org.shape[0] // 2:]

            elif type == "test":
                disp_cdt = self.sigmoid(condition)
                condition = condition if not self.opts.extra_condition else torch.cat([condition, rgb], dim=1)
                condition = condition + self.robustnet(condition) if self.opts.use_CNN else condition
                outputs = self.diffusion(condition=condition, isTrain=False, x0=self.convs["dispconv0"](x))  # only use the x0 shape
                outputs["beforeCNN"] = disp_cdt
                outputs["afterCNN"] = condition
        else:
            #no diffusion, for 1st stage's F_T
            outputs["condition"] = None
            outputs["x0"] = self.convs["dispconv0"](x) if self.opts.dfs_after_sigmod else self.sigmoid(self.convs["dispconv0"](x))
            outputs[("disp", 0)] = self.sigmoid(outputs["x0"]) if self.opts.dfs_after_sigmod else outputs["x0"]
            if self.opts.multi_scale:
                outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features["X_04"]))
                outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features["X_13"]))
                outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))
        return outputs
