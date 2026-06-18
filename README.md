# Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment

### AAAI 2025

Official PyTorch implementation of **Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment (B2-3Dnet)**.

B2-3Dnet is a detection-free image-to-point cloud registration network that estimates the rigid transformation `[R, t]` between a 2D image and a 3D point cloud. It follows a coarse-to-fine registration pipeline and introduces uncertainty-aware hierarchical matching and adversarial domain alignment to achieve accurate and robust cross-modal registration.

<p align="center">
  <img src="B23D.png" width="98%">
</p>

## Introduction

Image-to-point cloud registration aims to align a 2D image with a 3D point cloud from the same scene by estimating the rigid transformation from the point cloud coordinate system to the camera coordinate system. This task is important for 3D reconstruction, SLAM, visual localization, and robotic perception.

However, accurate image-to-point cloud registration remains challenging due to two key issues:

1. **Unreliable patch-level matching.**  
   Existing detection-free methods usually match image patches and point cloud patches uniformly. This may cause the network to focus on noisy, occluded, or weakly corresponding image regions while ignoring truly informative patches.

2. **Large cross-modal domain gap.**  
   Images are dense and regular 2D grids, while point clouds are sparse, unordered, and irregular 3D structures. Directly matching features from these two domains can lead to inconsistent descriptors and incorrect correspondences.

To address these issues, we propose **B2-3Dnet**, an uncertainty-aware hierarchical registration network with domain alignment. The network contains two core modules:

* **Uncertainty-aware Hierarchical Matching Module (UHMM)**  
  Models the uncertainty of image patches at multiple scales and performs hierarchical interaction between image and point cloud patches. This allows the model to focus on key informative image regions while suppressing noisy or unreliable patches.

* **Adversarial Modal Alignment Module (AMAM)**  
  Uses adversarial learning with a Gradient Reversal Layer and a domain classifier to reduce the feature distribution gap between image and point cloud modalities.

With these designs, B2-3Dnet improves the reliability of coarse patch matching, enhances dense correspondence refinement, and achieves strong generalization across indoor RGB-D benchmarks.

## Main Contributions

* We propose **B2-3Dnet**, a novel uncertainty-aware hierarchical network for detection-free image-to-point cloud registration.

* We design the **Uncertainty-aware Hierarchical Matching Module (UHMM)** to estimate the importance of image patches through uncertainty modeling. The module assigns lower uncertainty to reliable image patches and higher uncertainty to noisy or misaligned ones, reducing the influence of unreliable matches.

* We introduce hierarchical image patch features at multiple scales and interact them with point cloud patches from coarse to fine, enabling better matching under scale variation, viewpoint changes, and partial overlap.

* We propose the **Adversarial Modal Alignment Module (AMAM)** to bridge the domain gap between 2D image features and 3D point cloud features through adversarial domain alignment.

* Extensive experiments on **RGB-D Scenes V2** and **7-Scenes** demonstrate that B2-3Dnet achieves state-of-the-art performance in image-to-point cloud registration.

## Method Overview

B2-3Dnet consists of four main components:

1. **Feature Extraction Backbone**

   We use ResNet-FPN to extract image features and KPFCNN to extract point cloud features. The extracted 2D and 3D features are processed by self-attention and cross-attention layers to obtain more consistent cross-modal representations.

2. **Adversarial Modal Alignment Module**

   Since image and point cloud features come from different modalities, their feature distributions can be significantly different. AMAM introduces a domain classifier and a Gradient Reversal Layer to encourage the feature extractor to produce modality-invariant representations. During training, the domain classifier tries to distinguish image features from point cloud features, while the feature extractor learns to confuse the classifier, thereby reducing the domain gap.

3. **Uncertainty-aware Hierarchical Matching Module**

   UHMM constructs multi-scale image patch features and models each patch using a Gaussian distribution with predicted mean and variance. The variance represents uncertainty. Reliable image patches tend to have lower uncertainty, while noisy or incorrectly matched patches are assigned higher uncertainty.

   The module then performs hierarchical interaction between image patch features and point cloud patch features. This allows point cloud patches to perceive image regions with different receptive fields and improves the robustness of patch-level matching.

4. **Coarse-to-Dense Matching and Pose Estimation**

   After obtaining the coarse score map, mutual top-k selection is used to extract patch-level correspondences. These correspondences are further refined into dense pixel-to-point matches. Finally, PnP-RANSAC is applied to estimate the rigid transformation `[R, t]`.

## Results

### RGB-D Scenes V2

| Method | Inlier Ratio | Feature Matching Recall | Registration Recall |
|---|---:|---:|---:|
| 2D3D-MATR | 32.4 | 90.8 | 56.4 |
| B2-3Dnet | **35.1** | **94.4** | **63.4** |

On RGB-D Scenes V2, B2-3Dnet improves Registration Recall by **7.0 percentage points** over 2D3D-MATR.

### 7-Scenes

| Method | Inlier Ratio | Feature Matching Recall | Registration Recall |
|---|---:|---:|---:|
| 2D3D-MATR | 50.1 | 92.1 | 75.8 |
| B2-3Dnet | **50.9** | **93.1** | **77.7** |

On 7-Scenes, B2-3Dnet achieves better performance under larger scale variations and challenging scenes with repetitive structures.



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




