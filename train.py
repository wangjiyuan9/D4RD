# Code for training the robust monocular depth estimation model, enhanced from the original Monodepth2 codebase.
# Author: Jiyuan Wang
# Created: 2023-10-5
# Origin used for paper: https://arxiv.org/abs/2404.09831
# Hope you can cite our paper if you use the code for your research.
from __future__ import absolute_import, division, print_function
import os
import sys
import warnings
from options import MonodepthOptions

print(' '.join(sys.argv))
options = MonodepthOptions()
opts = options.parse()
warnings.filterwarnings("ignore")
if opts.debug:
    print("set save mode:", opts.save_strategy, "please \033[91mSTOP\033[0m if stll need old model")
if not opts.use_multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.cuda_devices)

if __name__ == "__main__":
    from trainer import Trainer

    trainer = Trainer(opts)
    trainer.train()
