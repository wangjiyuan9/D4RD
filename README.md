<div align="center">
<h2>Digging into Contrastive Learning for Robust Depth Estimation
 with Diffusion Models</h2>
<a href='https://scholar.google.com/citations?user=subRjlcAAAAJ&hl=zh-CN' target='_blank'>Jiyuan Wang</a><sup>1</sup>•<a href='https://scholar.google.com/citations?hl=zh-CN&user=t8xkhscAAAAJ' target='_blank'>Chunyu lin</a><sup>1</sup>•<a href='https://scholar.google.com/citations?hl=zh-CN&user=vo__egkAAAAJ' target='_blank'>Lang Nie</a><sup>1</sup>•<a href='XXX' target='_blank'>Shuwei Shao</a><sup>2</sup>

<sup>1</sup>Beijingjiaotong University <sup>1</sup>Beihang University

**ACM MM 2024**

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2404.09831) [![Static Badge](https://img.shields.io/badge/ZhiHu-Artical-blue)](https://zhuanlan.zhihu.com/p/1894794247015757493) [![Static Badge](https://img.shields.io/badge/BiliBili-Video-green)](https://www.bilibili.com/video/BV1nAd6YbESk)  

<div style="text-align:center">
<img src="assets/pipline.png"  width="100%" height="100%">
</div>
</div>

## 📢 Upcoming releases & Catalog

- [x] [release code for training and testing](#-pretrained-weights-and-evaluation)
- [x] train/validate data release(Please refer to [Weatherdepth](https://github.com/wangjiyuan9/WeatherDepth))
- [ ] finish the ReadMe 
- [x] **!NEW!** [test data release](#-dataset-preparation)([Dense dataset](https://huggingface.co/exander/D4RD/blob/main/dense.tar.gz) are **available**)
- [x] **!NEW!** [model release](#-pretrained-weights-and-evaluation) (D4RD, D4RD+ will are **available** at [huggingface](https://huggingface.co/exander/D4RD/tree/main))
- [x] realease the [introduction video](https://www.bilibili.com/video/BV1nAd6YbESk) for the paper

## 🛠️Environment Setup
To make the reproduction easy, we provide the condapack package of our environment at [here]()

We implement our method on MonoViT, Mono-Diffusion, and WeatherDepth baselines. If the tar.gz file is not adapted to your machine, you can refer to:
- [**MonoViT**(3DV2022)](https://github.com/zxcqlf/MonoViT)
- [**MonoDiffusion**(TCSVT2024)](https://github.com/ShuweiShao/MonoDiffusion)
- [**WeatherDepth**(ICRA2024)](https://github.com/wangjiyuan9/weatherdepth)
  
The training and inference code was tested on:
- Ubuntu 18.04 LTS, Python 3.8.8, CUDA 11.3, GeForce RTX 3090 (pip, Conda)
- Ubuntu 16.04 LTS, Python 3.7.15, CUDA 10.0, GeForce RTX 2080Ti (pip, Conda)

## 🖼️ Dataset Preparation

For WeatherDepth dataset Training, please refer to [WeatherDepth](https://github.com/wangjiyuan9/weatherdepth). The training data is exactly the same as theirs.

For KITTI-C dataset Training, please refer to [ECDepth](https://github.com/RuijieZhu94/EC-Depth). The KITTI-C dataset download and arrangement are basically the same as theirs. We also provide the this dataset image/depth GT at [here](https://huggingface.co/exander/D4RD/blob/main/kitti_c.zip)

For DrivingStereo dataset Testing, please refer to [drivingstereo-website](https://drivingstereo-dataset.github.io/).

For Dense dataset Testing, please download the tar.gz file from [here](https://huggingface.co/exander/D4RD/blob/main/dense.tar.gz) and untar it. The dataset will be arranged as:
```
Path-to-your-Dense-Datset > tree
.
├── gt_depths.npy
└── snowy
    ├── 2018-02-04_11-14-31_00200.png
    ├── 2018-02-04_11-14-31_00400.png
    ├── 2018-02-04_11-18-24_00100.png
    ├── 2018-02-04_11-20-41_00000.png
    ...
```
## 💾 Pretrained weights and evaluation

| Models             | abs rel | sq rel | rmse  | rmse log | a1    | a2    | a3    |
|--------------------|---------|--------|-------|----------|-------|-------|-------|
| [D4RD](https://huggingface.co/exander/D4RD/tree/main/D4RD) On WeatherKITTI   | 0.099   | 0.688  | 4.377 | 0.174    | 0.897 | 0.966 | 0.984 |
| [D4RD+](https://huggingface.co/exander/D4RD/tree/main/D4RD-KITTIC) On KITTI-C | 0.108|0.778|4.652|0.183|0.880|0.961|0.983|


Use the scripts below to inference and evaluate the model:
```
python Evaluate -lwf [Pretrained-Model-Path] --eval_split [Test-Split] --width 640 --height 192 --net_type vit --twt --ud --das -uC --ec
```

The test splits can be choose from:
**stereos**(Sunny subset of Drivingstereo Dataset); **stereoc**(Cloudy subset of Drivingstereo Dataset);**stereof**(Foggy subset of Drivingstereo Dataset);**stereor**(Rainy subset of Drivingstereo Dataset);**dense**(Snowy data in Dense dataset)

## ⏳ Training

**The teacher model** is avalible at ...

## 🎓 Citation
```bibtex
@inproceedings{Wang_2024, series={MM ’24},
   title={Digging into Contrastive Learning for Robust Depth Estimation with Diffusion Models},
   url={http://dx.doi.org/10.1145/3664647.3681168},
   DOI={10.1145/3664647.3681168},
   booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
   publisher={ACM},
   author={Wang, Jiyuan and Lin, Chunyu and Nie, Lang and Liao, Kang and Shao, Shuwei and Zhao, Yao},
   year={2024},
   month=oct, pages={4129–4137},
   collection={MM ’24} }
```

## 📚 Acknowledgements and License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The code is based on the MonoViT , Monodiffusion and WeatherDepth repositories. We thank the authors for their contributions. The data is based on the KITTI, Dense, and DrivingStereo datasets. We thank the authors for their contributions. If you have any questions, please feel free to contact us with issues or email.
