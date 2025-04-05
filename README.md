<h3 align="center"><strong>MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting</strong></h3>

  <p align="center">
    <a href="https://ruijiezhu94.github.io/ruijiezhu/">Ruijie Zhu*</a>,
    <a href="https://rosetta-leong.github.io/">Yanzhe Liang*</a>,
    <a href="">Hanzhi Chang</a>,
    <a href="">Jiacheng Deng</a>,
    <a href="">Jiahao Lu</a>, 
    <br>
    <a href="">Wenfei Yang</a>,
    <a href="http://staff.ustc.edu.cn/~tzzhang/">Tianzhu Zhang</a>,
    <a href="https://dblp.org/pid/z/YongdongZhang.html">Yongdong Zhang</a>
    <br>
    *Equal Contribution.
    <br>
    University of Science and Technology of China
    <br>
    <b>NeurIPS 2024</b>

</p>

<div align="center">
 <a href='https://arxiv.org/abs/2410.07707'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<!-- <a href='https://arxiv.org/abs/[]'><img src='https://img.shields.io/badge/arXiv-[]-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -->
 <a href='https://ruijiezhu94.github.io/MotionGS_page/'><img src='https://img.shields.io/badge/Project-Page-orange'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://www.youtube.com/embed/25DgViuuKFI'><img src='https://img.shields.io/badge/YouTube-Demo-yellow'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
 <a href='https://github.com/RuijieZhu94/MotionGS?tab=MIT-1-ov-file'><img src='https://img.shields.io/badge/License-MIT-green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://ruijiezhu94.github.io/MotionGS'><img src="https://visitor-badge.laobi.icu/badge?page_id=ruijiezhu94.motiongs"/></a>
 <br>
 <br>
</div>


<br>

<p align="center">
<img src="assets/pipeline.png" width="97%"/>
</p>

> The overall architecture of MotionGS. It can be viewed as two data streams: (1) The 2D data stream utilizes the optical flow decoupling module to obtain the motion flow as the 2D motion prior; (2) The 3D data stream involves the deformation and transformation of Gaussians to render the image for the next frame. During training, we alternately optimize 3DGS and camera poses through the camera pose refinement module.


## 🚀 Quick Start

### 🔧 Dataset Preparation
To train MotionGS, you should download the following dataset:
* [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/)
* [Hyper-NeRF](https://hypernerf.github.io/)
* [DyNeRF](https://github.com/facebookresearch/Neural_3D_Video)


We organize the datasets as follows:

```shell
├── data
│   | NeRF-DS
│     ├── as
│     ├── basin
│     ├── ...
│   | HyperNeRF
│     ├── interp
│     ├── misc
│     ├── vrig
│   | DyNeRF
│     ├── coffee_martini
│     ├── cook_spinach
│     ├── ...
```


### 🛠️ Installation
1. Clone this repo:
```bash
git clone git@github.com:RuijieZhu94/MotionGS.git --recursive
```
2. Install dependencies:
```bash
cd MotionGS

conda create -n motiongs python=3.7
conda activate motiongs

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```

### 🌟 Training

**NeRF-DS:**
```shell
expname=NeRF-DS
scenename=as_novel_view
mkdir -p output/$expname/$scenename

python train.py \
    -s data/NeRF-DS/$scenename \
    -m output/$expname/$scenename \
    --eval \
    --use_depth_and_flow \
    --optimize_pose
```

**HyperNeRF:**

```shell
expname=HyperNerf
scenename=broom2
mkdir -p output/$expname/$scenename

python train.py \
    -s data/hypernerf/vrig/$scenename \
    -m output/$expname/$scenename \
    --scene_format nerfies \
    --eval \
    --use_depth_and_flow \
    --optimize_pose
```

**DyNeRF:**
```shell
expname=dynerf
scenename=flame_steak
mkdir -p output/$expname/$scenename

python train.py \
    -s data/dynerf/$scenename \
    -m output/$expname/$scenename \
    --scene_format plenopticVideo \
    --resolution 4 \
    --dataloader \
    --eval \
    --use_depth_and_flow
```

### 🎇 Evaluation 

```shell
python render.py -m output/exp-name --mode render
python metrics.py -m output/exp-name
```

We provide several modes for rendering:

- `render`: render all the test images
- `time`: time interpolation tasks for D-NeRF dataset
- `all`: time and view synthesis tasks for D-NeRF dataset
- `view`: view synthesis tasks for D-NeRF dataset
- `original`: time and view synthesis tasks for real-world dataset

### 📜 Citation

If you find our work useful, please cite:

```bibtex
@article{zhu2024motiongs,
  title={Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting},
  author={Zhu, Ruijie and Liang, Yanzhe and Chang, Hanzhi and Deng, Jiacheng and Lu, Jiahao and Yang, Wenfei and Zhang, Tianzhu and Zhang, Yongdong},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={101790--101817},
  year={2024}
}
```

### 🤝 Acknowledgements
Our code is based on [Deformable3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians), [GaussianFlow](https://github.com/Zerg-Overmind/GaussianFlow), [MonoGS](https://github.com/muskie82/MonoGS), [CF-3DGS](https://github.com/NVlabs/CF-3DGS), [DynPoint](https://github.com/kaichen-z/DynPoint), [MiDas](https://github.com/isl-org/MiDaS), [GMFlow](https://github.com/haofeixu/gmflow) and [MDFlow](https://github.com/ltkong218/MDFlow). We thank the authors for their excellent work!