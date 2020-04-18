# Depth-Wise Pruning
This repository provides the implementation of the method proposed in our paper [Discriminative Layer Pruning for Convolutional Neural Networks](https://www.researchgate.net/profile/Maiko_Lie/publication/339851235_Discriminative_Layer_Pruning_for_Convolutional_Neural_Networks/links/5e878ded92851c2f527b918c/Discriminative-Layer-Pruning-for-Convolutional-Neural-Networks.pdf). 

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://github.com/fchollet/keras) (Recommended version 2.1.2)
- [Tensorflow](https://www.tensorflow.org/) (Recommended version 1.3.0 or 1.9)
- [Python 3](https://www.python.org/)

## Quick Start
[main.py](main.py) provides an example of our layer pruning approach.

## Parameters
Our method takes two parameters:
1. Number of components of Partial Least Squares (see line 178 in [main.py](main.py))
2. Block index (see line 180 in [main.py](main.py)). The lower the block index the higher the FLOPs reduction and accuracy degradation.

## Results
The table below show the comparison between our method with existing pruning methods. Negative values in accuracy denote improvement regarding the original, unpruned, network. Please check our paper for more detailed results.

ResNet56 on Cifar-10

|     Method     | FLOPs ↓ (%) | Accuracy ↓ (percentage points) |
|:--------------:|:-----:|:----------------:|
| [Jordao et al.](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0601.pdf) |   52.56 |       -0.62       |
| [He et al.](https://drive.google.com/file/d/1sVnCCagzU2U4meFlJZ9f8vRPKM5En3HK/view) | 50.00 |  0.90       |
|   Ours   |   30.00 | 0.98 |
|   Ours + Pruning Filters   |  62.69 | 0.91 |

Please cite our paper in your publications if it helps your research.
```bash
@article{Jordao::2020,
author    = {Artur Jordao,
Maiko Lie and
William Robson Schwartz},
title     = {Discriminative Layer Pruning for Convolutional Neural Networks},
journal   = {IEEE Journal of Selected Topics in Signal Processing},
year      = {2020},
}
```
