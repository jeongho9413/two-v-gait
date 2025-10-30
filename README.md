# 2VGait

* This is the official implementation of our paper, **Learning Viewpoint-Invariant Features for LiDAR-Based Gait Recognition** [\[PDF\]](https://ieeexplore.ieee.org/document/10318107).
* 2VGait is an end-to-end framework for gait recognition using 3D LiDAR, robust to changes in viewing angles and measurement distances.
* You can also implemenet the previous version, **2V-Gait: Gait Recognition using 3D LiDAR Robust to Changes in Walking Direction and Measurement Distance** [\[PDF\]](https://ieeexplore.ieee.org/document/9708899).


## Overview

<p align="center">
  <img src="assets/overview.png" width="900"/></br>
  <span align="center">Overview of the 2VGait, which learns two viewpoint-invariant gait shapes in varying point cloud densities using an attention-based approach.</span> 
</p>


## Data collection

<p align="center">
  <img src="assets/dist_10m.png" height="250px"><br>
  <img src="assets/dist_20m.png" height="250px"><br>
  <span align="center">Data acquisition environment with two distances measured from a VLP-32C, which is visualized in a 3D point cloud format.</span> 
</p>



## Experimental result

<p align="center">
  <img src="assets/results_practicality.png" width="800"/></br>
  <span align="center">Comparison with prior studies for evaluating by limiting viewing angles (%).</span> 
</p>


## Usage

### Prepare the dataset

We are not publish our dataset `KUGait30` at the moment, which was used in this work.
However, you can refer to how we implemented, and modify it for your work through this codebase.


### Training
Transform 3D pedestrian point clouds on the depth representation and build train-sets:

````bash
python build_dataset_for_training.py --dataset_path ./datasets/KUGait_VLP32C_2022-Spring-C/ \
                                                    --yml_path ./configs/KUGait_VLP32C_2022-Spring-C/build_datasets_1020m.yml
````

Train a model with the train-sets:

````bash
python train.py --dataset_path ./datasets/KUGait_VLP32C_2022-Spring-C/train \
                              --model_path ./pretrained_20221201 \
                              --batchsize 42 \
                              --nepoch 50 \
````

### Evaluation
Build test-sets for evaluation:

````bash
python build_dataset_for_test.py --dataset_path ./datasets/KUGait_VLP32C_2022-Spring-C/ \
                                              --yml_path ./configs/KUGait_VLP32C_2022-Spring-C/build_datasets_1020m.yml

````

Evaluate the trained model on test-sets:

````bash
python test.py --model_path ./pretrained_20221201/model_state_dict.pth
                          --test_path ./datasets/KUGait_VLP32C_2022-Spring-C/test

````


## Citation

If you find useful for your work, please cite our paper:

```bibtex
@article{access2023_ahn,
  title   = {Learning Viewpoint-Invariant Features for LiDAR-Based Gait Recognition},
  author  = {Ahn, Jeongho and Nakashima, Kazuto and Yoshino, Koki and Iwashita, Yumi and Kurazume, Ryo},
  journal = {IEEE Access},
  volume  = {11},
  number  = {},
  pages   = {129749-129762},
  year    = {2023},
  paper   = {https://doi.org/10.1109/ACCESS.2023.3333037}
}
```
