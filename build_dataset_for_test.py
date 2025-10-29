import os
import yaml
import argparse

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from twovgait.data.dataset import GenerateOrthoDep, build_test_set


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
    build_test_set(ped_path=raw_path, yml_file=yml_file)


if __name__ == '__main__':
    main()
