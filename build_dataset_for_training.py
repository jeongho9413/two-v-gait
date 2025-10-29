import yaml
import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from twovgait.data.dataset import GenerateOrthoDep, build_train_set


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset_path', type=str, default='./datasets/KUGait_VLP32C_2022-Spring-C/')
    parser.add_argument('--yml_path', type=str, default='./configs/KUGait_VLP32C_2022-Spring-C/build_datasets_1020m.yml')
    args = parser.parse_args()
    return args


def main():
    args = parse_config()
    with open(args.yml_path) as file:
        yml_file = yaml.safe_load(file)

    raw_path = os.path.join(args.dataset_path, "raw")
    train_x, train_y = build_train_set(ped_path=raw_path, yml_file=yml_file)
    net = GenerateOrthoDep()

    rd_seed = 5
    random.Random(rd_seed).shuffle(train_x)
    random.Random(rd_seed).shuffle(train_y)

    train_x_dep_side, train_x_mask_side, train_x_gei_side, train_x_dep_back, train_x_mask_back, train_x_gei_back, train_x_dep_nongde, train_x_mask_nongde, train_x_gei_nongde, train_x_speed = net(train_x)
    train_y = np.asarray(train_y, dtype=np.int16)

    train_path = os.path.join(args.ped_path, "train")
    os.makedirs(train_path, exist_ok=True)

    np.save(os.path.join(train_path, "train_x_dep_side.npy"), train_x_dep_side)
    np.save(os.path.join(train_path, "train_x_mask_side.npy"), train_x_mask_side)
    np.save(os.path.join(train_path, "train_x_gei_side.npy"), train_x_gei_side)

    np.save(os.path.join(train_path, "train_x_dep_back.npy"), train_x_dep_back)
    np.save(os.path.join(train_path, "train_x_mask_back.npy"), train_x_mask_back)
    np.save(os.path.join(train_path, "train_x_gei_back.npy"), train_x_gei_back)

    np.save(os.path.join(train_path, "train_x_dep_nongde.npy"), train_x_dep_nongde)
    np.save(os.path.join(train_path, "train_x_mask_nongde.npy"), train_x_mask_nongde)
    np.save(os.path.join(train_path, "train_x_gei_nongde.npy"), train_x_gei_nongde)

    np.save(os.path.join(train_path, "train_x_speed.npy"), train_x_speed)
    np.save(os.path.join(train_path, "train_y.npy"), train_y)


if __name__ == '__main__':
    main()
