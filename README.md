# Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment

### AAAI 2025


We propose the B2-3Dnet, a novel uncertainty-aware hierarchical registration network with domain alignment, demonstrating excellent accuracy and strong generalization in image-to-point cloud registration tasks.

<p align="center">
  <img src="B23D.png" width="98%">
</p>

## Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n B23D python=3.8
conda activate B23D

# Install vision3d following https://github.com/qinzheng93/vision3d
```

The code has been tested on Python 3.8, PyTorch 1.13.1, Ubuntu 22.04, GCC 11.3 and CUDA 11.7, but it should work with other configurations.

## 7Scenes

### Data preparation
The data should be organized as follows:

```text
--data--7Scenes--metadata
              |--data--chess
                     |--fire
                     |--heads
                     |--office
                     |--pumpkin
                     |--redkitchen
                     |--stairs
```

### Training

The code for 7Scenes is in `experiments/B23D.7scenes`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH
```

`EPOCH` is the epoch id.

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/path/to/B23D/weights/b23d-7scenes.pth
CUDA_VISIBLE_DEVICES=0 python eval.py --test_epoch=-1
```

## RGB-D Scenes V2

### Data preparation

The data should be organized as follows:

```text
--data--RGBDScenesV2--metadata
              |--data--rgbd-scenes-v2-scene_01
                     |--...
                     |--rgbd-scenes-v2-scene_14
```

### Training

The code for RGB-D Scenes V2 is in `experiments/B23D.rgbdv2`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH
```

`EPOCH` is the epoch id.

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint=/path/to/B23D/weights/B23D-rgbdv2.pth
CUDA_VISIBLE_DEVICES=0 python eval.py --test_epoch=-1
```

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@inproceedings{cheng2025bridge,
  title     = {Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment},
  author    = {Cheng, Zhixin and Deng, Jiacheng and Li, Xinjun and Yin, Baoqun and Zhang, Tianzhu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {2491--2499},
  year      = {2025}
}
```

## Acknowledgements

**We sincerely thank the authors of 2D3D-MATR for their excellent work and publicly available codebase. Our implementation is partially built upon their repository.**




