# NeuralSkeletonPoseTransfer

This is an implementation of the paper ["3D Mesh Pose Transfer Based on Skeletal Deformation"](https://onlinelibrary.wiley.com/doi/10.1002/cav.2156).

## Prerequisites

There are some important dependencies:
- python=3.9
- pytorch=1.11
- igl=2.2.1
- h5py=3.6.0
- scipy=1.8.0
- meshplot=0.4.0

For more detailed environment information, please refer to `environment.yml`.
You can download and extract the dataset form [One Drive](https://1drv.ms/u/s!AknUAqzhZIMYiymLQgO8yinJev-m?e=sKLugp) and put them under the `data` directory.

## Quick start

Visualization tool **meshplot** relies on interactive environment, we recommend using **jupyter notebook** to run `test_pose_transfer.ipynb`, you can easily use it by installing a plugin on **VSCode**. Our pretrained models are stored in `statedict`, where `skinningNet_finetune_noise.pkl` for human and `skinningNet_finetuen_animal.pkl` for animal.

## Trianing

First you need to train JointNet by `train_JointNet.ipynb` and train WeightNet by `train_WeightBindingNet.ipynb` respectively. Finally finetune whole SkinningNet by `train_SkinningNet.ipynb`.

## Acknowledgement

Partial code is based on [DGCNN](https://github.com/WangYueFt/dgcnn), [NBS](https://github.com/PeizhuoLi/neural-blend-shapes) and other open source projects, thanks to all the contributors.

## Citation

If you found our code is helpful for your research, please cite our paper:
~~~bibtex
@article{https://doi.org/10.1002/cav.2156,
author = {Yang, Shigeng and Yin, Mengxiao and Li, Ming and Li, Guiqing and Chang, Kan and Yang, Feng},
title = {3D mesh pose transfer based on skeletal deformation},
journal = {Computer Animation and Virtual Worlds},
volume = {34},
number = {3-4},
pages = {e2156},
doi = {https://doi.org/10.1002/cav.2156},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/cav.2156},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/cav.2156},
year = {2023}
}
~~~
