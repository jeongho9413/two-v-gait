import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel as parallel
#from torch.utils.data import TensorDataset, DataLoader
#from torchvision.utils import make_grid
#from torchvision.io import read_image

import matplotlib.pyplot as plt
from matplotlib import cm
#from loss_functions import AngularPenaltySMLoss
from sklearn import neighbors

from twovgait.models.twovgait_2023 import TwoVGaitNonPen
from twovgait.utils.common import t_sne, extract_feature


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_path', type=str, default='./pretrained_20221201/model_state_dict.pth')
    parser.add_argument('--test_path', type=str, default='./datasets/KUGait_VLP32C_2022-Spring-C/test')
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    args = parse_config()

    # load gallery_10m_x_
    gallery_10m_0_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_0_x_dep_side.npy'))
    gallery_10m_45_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_45_x_dep_side.npy'))
    gallery_10m_90_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_90_x_dep_side.npy'))
    gallery_10m_135_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_135_x_dep_side.npy'))
    gallery_10m_180_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_180_x_dep_side.npy'))
    gallery_10m_225_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_225_x_dep_side.npy'))
    gallery_10m_270_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_270_x_dep_side.npy'))
    gallery_10m_315_x_side = torch.tensor(np.load(args.test_path + '/gallery_10m_315_x_dep_side.npy'))
    
    gallery_10m_0_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_0_x_dep_back.npy'))
    gallery_10m_45_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_45_x_dep_back.npy'))
    gallery_10m_90_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_90_x_dep_back.npy'))
    gallery_10m_135_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_135_x_dep_back.npy'))
    gallery_10m_180_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_180_x_dep_back.npy'))
    gallery_10m_225_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_225_x_dep_back.npy'))
    gallery_10m_270_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_270_x_dep_back.npy'))
    gallery_10m_315_x_back = torch.tensor(np.load(args.test_path + '/gallery_10m_315_x_dep_back.npy'))
    
    gallery_10m_0_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_0_x_dist.npy'))
    gallery_10m_45_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_45_x_dist.npy'))
    gallery_10m_90_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_90_x_dist.npy'))
    gallery_10m_135_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_135_x_dist.npy'))
    gallery_10m_180_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_180_x_dist.npy'))
    gallery_10m_225_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_225_x_dist.npy'))
    gallery_10m_270_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_270_x_dist.npy'))
    gallery_10m_315_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_10m_315_x_dist.npy'))
    
    # load gallery_10m_x_
    gallery_20m_0_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_0_x_dep_side.npy'))
    gallery_20m_45_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_45_x_dep_side.npy'))
    gallery_20m_90_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_90_x_dep_side.npy'))
    gallery_20m_135_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_135_x_dep_side.npy'))
    gallery_20m_180_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_180_x_dep_side.npy'))
    gallery_20m_225_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_225_x_dep_side.npy'))
    gallery_20m_270_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_270_x_dep_side.npy'))
    gallery_20m_315_x_side = torch.tensor(np.load(args.test_path + '/gallery_20m_315_x_dep_side.npy'))
    
    gallery_20m_0_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_0_x_dep_back.npy'))
    gallery_20m_45_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_45_x_dep_back.npy'))
    gallery_20m_90_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_90_x_dep_back.npy'))
    gallery_20m_135_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_135_x_dep_back.npy'))
    gallery_20m_180_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_180_x_dep_back.npy'))
    gallery_20m_225_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_225_x_dep_back.npy'))
    gallery_20m_270_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_270_x_dep_back.npy'))
    gallery_20m_315_x_back = torch.tensor(np.load(args.test_path + '/gallery_20m_315_x_dep_back.npy'))
    
    gallery_20m_0_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_0_x_dist.npy'))
    gallery_20m_45_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_45_x_dist.npy'))
    gallery_20m_90_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_90_x_dist.npy'))
    gallery_20m_135_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_135_x_dist.npy'))
    gallery_20m_180_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_180_x_dist.npy'))
    gallery_20m_225_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_225_x_dist.npy'))
    gallery_20m_270_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_270_x_dist.npy'))
    gallery_20m_315_x_speed = torch.tensor(np.load(args.test_typec_path + '/gallery_20m_315_x_dist.npy'))
    
    # load probe_10m_x_
    probe_10m_0_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_0_x_dep_side.npy'))
    probe_10m_45_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_45_x_dep_side.npy'))
    probe_10m_90_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_90_x_dep_side.npy'))
    probe_10m_135_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_135_x_dep_side.npy'))
    probe_10m_180_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_180_x_dep_side.npy'))
    probe_10m_225_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_225_x_dep_side.npy'))
    probe_10m_270_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_270_x_dep_side.npy'))
    probe_10m_315_x_side = torch.tensor(np.load(args.test_path + '/probe_10m_315_x_dep_side.npy'))
    
    probe_10m_0_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_0_x_dep_back.npy'))
    probe_10m_45_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_45_x_dep_back.npy'))
    probe_10m_90_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_90_x_dep_back.npy'))
    probe_10m_135_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_135_x_dep_back.npy'))
    probe_10m_180_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_180_x_dep_back.npy'))
    probe_10m_225_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_225_x_dep_back.npy'))
    probe_10m_270_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_270_x_dep_back.npy'))
    probe_10m_315_x_back = torch.tensor(np.load(args.test_path + '/probe_10m_315_x_dep_back.npy'))
    
    probe_10m_0_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_0_x_dist.npy'))
    probe_10m_45_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_45_x_dist.npy'))
    probe_10m_90_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_90_x_dist.npy'))
    probe_10m_135_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_135_x_dist.npy'))
    probe_10m_180_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_180_x_dist.npy'))
    probe_10m_225_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_225_x_dist.npy'))
    probe_10m_270_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_270_x_dist.npy'))
    probe_10m_315_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_10m_315_x_dist.npy'))
    
    # load probe_20m_x_
    probe_20m_0_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_0_x_dep_side.npy'))
    probe_20m_45_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_45_x_dep_side.npy'))
    probe_20m_90_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_90_x_dep_side.npy'))
    probe_20m_135_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_135_x_dep_side.npy'))
    probe_20m_180_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_180_x_dep_side.npy'))
    probe_20m_225_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_225_x_dep_side.npy'))
    probe_20m_270_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_270_x_dep_side.npy'))
    probe_20m_315_x_side = torch.tensor(np.load(args.test_path + '/probe_20m_315_x_dep_side.npy'))
    
    probe_20m_0_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_0_x_dep_back.npy'))
    probe_20m_45_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_45_x_dep_back.npy'))
    probe_20m_90_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_90_x_dep_back.npy'))
    probe_20m_135_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_135_x_dep_back.npy'))
    probe_20m_180_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_180_x_dep_back.npy'))
    probe_20m_225_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_225_x_dep_back.npy'))
    probe_20m_270_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_270_x_dep_back.npy'))
    probe_20m_315_x_back = torch.tensor(np.load(args.test_path + '/probe_20m_315_x_dep_back.npy'))
    
    probe_20m_0_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_0_x_dist.npy'))
    probe_20m_45_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_45_x_dist.npy'))
    probe_20m_90_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_90_x_dist.npy'))
    probe_20m_135_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_135_x_dist.npy'))i
    probe_20m_180_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_180_x_dist.npy'))
    probe_20m_225_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_225_x_dist.npy'))
    probe_20m_270_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_270_x_dist.npy'))
    probe_20m_315_x_speed = torch.tensor(np.load(args.test_typec_path + '/probe_20m_315_x_dist.npy'))

    # load gallery&probe_10&20m_y
    gallery_10m_0_y = torch.tensor(np.load(args.test_path + '/gallery_10m_0_y.npy')).type(torch.LongTensor)
    gallery_10m_45_y = torch.tensor(np.load(args.test_path + '/gallery_10m_45_y.npy')).type(torch.LongTensor)
    gallery_10m_90_y = torch.tensor(np.load(args.test_path + '/gallery_10m_90_y.npy')).type(torch.LongTensor)
    gallery_10m_135_y = torch.tensor(np.load(args.test_path + '/gallery_10m_135_y.npy')).type(torch.LongTensor)
    gallery_10m_180_y = torch.tensor(np.load(args.test_path + '/gallery_10m_180_y.npy')).type(torch.LongTensor)
    gallery_10m_225_y = torch.tensor(np.load(args.test_path + '/gallery_10m_225_y.npy')).type(torch.LongTensor)
    gallery_10m_270_y = torch.tensor(np.load(args.test_path + '/gallery_10m_270_y.npy')).type(torch.LongTensor)
    gallery_10m_315_y = torch.tensor(np.load(args.test_path + '/gallery_10m_315_y.npy')).type(torch.LongTensor)

    gallery_20m_0_y = torch.tensor(np.load(args.test_path + '/gallery_20m_0_y.npy')).type(torch.LongTensor)
    gallery_20m_45_y = torch.tensor(np.load(args.test_path + '/gallery_20m_45_y.npy')).type(torch.LongTensor)
    gallery_20m_90_y = torch.tensor(np.load(args.test_path + '/gallery_20m_90_y.npy')).type(torch.LongTensor)
    gallery_20m_135_y = torch.tensor(np.load(args.test_path + '/gallery_20m_135_y.npy')).type(torch.LongTensor)
    gallery_20m_180_y = torch.tensor(np.load(args.test_path + '/gallery_20m_180_y.npy')).type(torch.LongTensor)
    gallery_20m_225_y = torch.tensor(np.load(args.test_path + '/gallery_20m_225_y.npy')).type(torch.LongTensor)
    gallery_20m_270_y = torch.tensor(np.load(args.test_path + '/gallery_20m_270_y.npy')).type(torch.LongTensor)
    gallery_20m_315_y = torch.tensor(np.load(args.test_path + '/gallery_20m_315_y.npy')).type(torch.LongTensor)

    probe_10m_0_y = torch.tensor(np.load(args.test_path + '/probe_10m_0_y.npy')).type(torch.LongTensor)
    probe_10m_45_y = torch.tensor(np.load(args.test_path + '/probe_10m_45_y.npy')).type(torch.LongTensor)
    probe_10m_90_y = torch.tensor(np.load(args.test_path + '/probe_10m_90_y.npy')).type(torch.LongTensor)
    probe_10m_135_y = torch.tensor(np.load(args.test_path + '/probe_10m_135_y.npy')).type(torch.LongTensor)
    probe_10m_180_y = torch.tensor(np.load(args.test_path + '/probe_10m_180_y.npy')).type(torch.LongTensor)
    probe_10m_225_y = torch.tensor(np.load(args.test_path + '/probe_10m_225_y.npy')).type(torch.LongTensor)
    probe_10m_270_y = torch.tensor(np.load(args.test_path + '/probe_10m_270_y.npy')).type(torch.LongTensor)
    probe_10m_315_y = torch.tensor(np.load(args.test_path + '/probe_10m_315_y.npy')).type(torch.LongTensor)

    probe_20m_0_y = torch.tensor(np.load(args.test_path + '/probe_20m_0_y.npy')).type(torch.LongTensor)
    probe_20m_45_y = torch.tensor(np.load(args.test_path + '/probe_20m_45_y.npy')).type(torch.LongTensor)
    probe_20m_90_y = torch.tensor(np.load(args.test_path + '/probe_20m_90_y.npy')).type(torch.LongTensor)
    probe_20m_135_y = torch.tensor(np.load(args.test_path + '/probe_20m_135_y.npy')).type(torch.LongTensor)
    probe_20m_180_y = torch.tensor(np.load(args.test_path + '/probe_20m_180_y.npy')).type(torch.LongTensor)
    probe_20m_225_y = torch.tensor(np.load(args.test_path + '/probe_20m_225_y.npy')).type(torch.LongTensor)
    probe_20m_270_y = torch.tensor(np.load(args.test_path + '/probe_20m_270_y.npy')).type(torch.LongTensor)
    probe_20m_315_y = torch.tensor(np.load(args.test_path + '/probe_20m_315_y.npy')).type(torch.LongTensor)

    # configure a model
    net = TwoVGaitNonPen()

    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    net.to(device)

    # implement a test
    test_list = [[gallery_10m_0_x_side, gallery_10m_0_x_back, gallery_10m_0_x_speed, gallery_10m_0_y, probe_10m_0_x_side, probe_10m_0_x_back, probe_10m_0_x_speed, probe_10m_0_y, 'gallery_10m_0deg <- probe_10m_0deg'],  # gallery_10m
                    [gallery_10m_0_x_side, gallery_10m_0_x_back, gallery_10m_0_x_speed, gallery_10m_0_y, probe_20m_0_x_side, probe_20m_0_x_back, probe_20m_0_x_speed, probe_20m_0_y, 'gallery_10m_0deg <- probe_20m_0deg'],
                    [gallery_10m_45_x_side, gallery_10m_45_x_back, gallery_10m_45_x_speed, gallery_10m_45_y, probe_10m_45_x_side, probe_10m_45_x_back, probe_10m_45_x_speed, probe_10m_45_y, 'gallery_10m_45deg <- probe_10m_45deg'],
                    [gallery_10m_45_x_side, gallery_10m_45_x_back, gallery_10m_45_x_speed, gallery_10m_45_y, probe_20m_45_x_side, probe_20m_45_x_back, probe_20m_45_x_speed, probe_20m_45_y, 'gallery_10m_45deg <- probe_20m_45deg'],
                    [gallery_10m_90_x_side, gallery_10m_90_x_back, gallery_10m_90_x_speed, gallery_10m_90_y, probe_10m_90_x_side, probe_10m_90_x_back, probe_10m_90_x_speed, probe_10m_90_y, 'gallery_10m_90deg <- probe_10m_90deg'],
                    [gallery_10m_90_x_side, gallery_10m_90_x_back, gallery_10m_90_x_speed, gallery_10m_90_y, probe_20m_90_x_side, probe_20m_90_x_back, probe_20m_90_x_speed, probe_20m_90_y, 'gallery_10m_90deg <- probe_20m_90deg'],
                    [gallery_10m_135_x_side, gallery_10m_135_x_back, gallery_10m_135_x_speed, gallery_10m_135_y, probe_10m_135_x_side, probe_10m_135_x_back, probe_10m_135_x_speed, probe_10m_135_y, 'gallery_10m_135deg <- probe_10m_135deg'],
                    [gallery_10m_135_x_side, gallery_10m_135_x_back, gallery_10m_135_x_speed, gallery_10m_135_y, probe_20m_135_x_side, probe_20m_135_x_back, probe_20m_135_x_speed, probe_20m_135_y, 'gallery_10m_135deg <- probe_20m_135deg'],
                    [gallery_10m_180_x_side, gallery_10m_180_x_back, gallery_10m_180_x_speed, gallery_10m_180_y, probe_10m_180_x_side, probe_10m_180_x_back, probe_10m_180_x_speed, probe_10m_180_y, 'gallery_10m_180deg <- probe_10m_180deg'],
                    [gallery_10m_180_x_side, gallery_10m_180_x_back, gallery_10m_180_x_speed, gallery_10m_180_y, probe_20m_180_x_side, probe_20m_180_x_back, probe_20m_180_x_speed, probe_20m_180_y, 'gallery_10m_180deg <- probe_20m_180deg'],
                    [gallery_10m_225_x_side, gallery_10m_225_x_back, gallery_10m_225_x_speed, gallery_10m_225_y, probe_10m_225_x_side, probe_10m_225_x_back, probe_10m_225_x_speed, probe_10m_225_y, 'gallery_10m_225deg <- probe_10m_225deg'],
                    [gallery_10m_225_x_side, gallery_10m_225_x_back, gallery_10m_225_x_speed, gallery_10m_225_y, probe_20m_225_x_side, probe_20m_225_x_back, probe_20m_225_x_speed, probe_20m_225_y, 'gallery_10m_225deg <- probe_20m_225deg'],
                    [gallery_10m_270_x_side, gallery_10m_270_x_back, gallery_10m_270_x_speed, gallery_10m_270_y, probe_10m_270_x_side, probe_10m_270_x_back, probe_10m_270_x_speed, probe_10m_270_y, 'gallery_10m_270deg <- probe_10m_270deg'],
                    [gallery_10m_270_x_side, gallery_10m_270_x_back, gallery_10m_270_x_speed, gallery_10m_270_y, probe_20m_270_x_side, probe_20m_270_x_back, probe_20m_270_x_speed, probe_20m_270_y, 'gallery_10m_270deg <- probe_20m_270deg'],
                    [gallery_10m_315_x_side, gallery_10m_315_x_back, gallery_10m_315_x_speed, gallery_10m_315_y, probe_10m_315_x_side, probe_10m_315_x_back, probe_10m_315_x_speed, probe_10m_315_y, 'gallery_10m_315deg <- probe_10m_315deg'],
                    [gallery_10m_315_x_side, gallery_10m_315_x_back, gallery_10m_315_x_speed, gallery_10m_315_y, probe_20m_315_x_side, probe_20m_315_x_back, probe_20m_315_x_speed, probe_20m_315_y, 'gallery_10m_315deg <- probe_20m_315deg'],
                    [gallery_20m_0_x_side, gallery_20m_0_x_back, gallery_20m_0_x_speed, gallery_20m_0_y, probe_10m_0_x_side, probe_10m_0_x_back, probe_10m_0_x_speed, probe_10m_0_y, 'gallery_20m_0deg <- probe_10m_0deg'],  # gallery_20m
                    [gallery_20m_0_x_side, gallery_20m_0_x_back, gallery_20m_0_x_speed, gallery_20m_0_y, probe_20m_0_x_side, probe_20m_0_x_back, probe_20m_0_x_speed, probe_20m_0_y, 'gallery_20m_0deg <- probe_20m_0deg'],
                    [gallery_20m_45_x_side, gallery_20m_45_x_back, gallery_20m_45_x_speed, gallery_20m_45_y, probe_10m_45_x_side, probe_10m_45_x_back, probe_10m_45_x_speed, probe_10m_45_y, 'gallery_20m_45deg <- probe_10m_45deg'],
                    [gallery_20m_45_x_side, gallery_20m_45_x_back, gallery_20m_45_x_speed, gallery_20m_45_y, probe_20m_45_x_side, probe_20m_45_x_back, probe_20m_45_x_speed, probe_20m_45_y, 'gallery_20m_45deg <- probe_20m_45deg'],
                    [gallery_20m_90_x_side, gallery_20m_90_x_back, gallery_20m_90_x_speed, gallery_20m_90_y, probe_10m_90_x_side, probe_10m_90_x_back, probe_10m_90_x_speed, probe_10m_90_y, 'gallery_20m_90deg <- probe_10m_90deg'],
                    [gallery_20m_90_x_side, gallery_20m_90_x_back, gallery_20m_90_x_speed, gallery_20m_90_y, probe_20m_90_x_side, probe_20m_90_x_back, probe_20m_90_x_speed, probe_20m_90_y, 'gallery_20m_90deg <- probe_20m_90deg'],
                    [gallery_20m_135_x_side, gallery_20m_135_x_back, gallery_20m_135_x_speed, gallery_20m_135_y, probe_10m_135_x_side, probe_10m_135_x_back, probe_10m_135_x_speed, probe_10m_135_y, 'gallery_20m_135deg <- probe_10m_135deg'],
                    [gallery_20m_135_x_side, gallery_20m_135_x_back, gallery_20m_135_x_speed, gallery_20m_135_y, probe_20m_135_x_side, probe_20m_135_x_back, probe_20m_135_x_speed, probe_20m_135_y, 'gallery_20m_135deg <- probe_20m_135deg'],
                    [gallery_20m_180_x_side, gallery_20m_180_x_back, gallery_20m_180_x_speed, gallery_20m_180_y, probe_10m_180_x_side, probe_10m_180_x_back, probe_10m_180_x_speed, probe_10m_180_y, 'gallery_20m_180deg <- probe_10m_180deg'],
                    [gallery_20m_180_x_side, gallery_20m_180_x_back, gallery_20m_180_x_speed, gallery_20m_180_y, probe_20m_180_x_side, probe_20m_180_x_back, probe_20m_180_x_speed, probe_20m_180_y, 'gallery_20m_180deg <- probe_20m_180deg'],
                    [gallery_20m_225_x_side, gallery_20m_225_x_back, gallery_20m_225_x_speed, gallery_20m_225_y, probe_10m_225_x_side, probe_10m_225_x_back, probe_10m_225_x_speed, probe_10m_225_y, 'gallery_20m_225deg <- probe_10m_225deg'],
                    [gallery_20m_225_x_side, gallery_20m_225_x_back, gallery_20m_225_x_speed, gallery_20m_225_y, probe_20m_225_x_side, probe_20m_225_x_back, probe_20m_225_x_speed, probe_20m_225_y, 'gallery_20m_225deg <- probe_20m_225deg'],
                    [gallery_20m_270_x_side, gallery_20m_270_x_back, gallery_20m_270_x_speed, gallery_20m_270_y, probe_10m_270_x_side, probe_10m_270_x_back, probe_10m_270_x_speed, probe_10m_270_y, 'gallery_20m_270deg <- probe_10m_270deg'],
                    [gallery_20m_270_x_side, gallery_20m_270_x_back, gallery_20m_270_x_speed, gallery_20m_270_y, probe_20m_270_x_side, probe_20m_270_x_back, probe_20m_270_x_speed, probe_20m_270_y, 'gallery_20m_270deg <- probe_20m_270deg'],
                    [gallery_20m_315_x_side, gallery_20m_315_x_back, gallery_20m_315_x_speed, gallery_20m_315_y, probe_10m_315_x_side, probe_10m_315_x_back, probe_10m_315_x_speed, probe_10m_315_y, 'gallery_20m_315deg <- probe_10m_315deg'],
                    [gallery_20m_315_x_side, gallery_20m_315_x_back, gallery_20m_315_x_speed, gallery_20m_315_y, probe_20m_315_x_side, probe_20m_315_x_back, probe_20m_315_x_speed, probe_20m_315_y, 'gallery_20m_315deg <- probe_20m_315deg']]

    batch_mini = 24
    for test_data in test_list:
        features_g, labels_g = extract_feature(net, test_data[0], test_data[1], test_data[2], device, batch_mini)
        
        #t_sne(features_g, labels_g, test_data[-1])  #  visualize gallery datas.
        #print(features_g.shape)
        #print(labels_g)
        #sys.exit()

        features_p, labels_p = extract_feature(net, test_data[3], test_data[4], test_data[5], device, batch_mini)
        knn = neighbors.KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(features_g, labels_g)
        predictions = knn.predict(features_p)
        success = sum(predictions == labels_p)
        rate = float(success) / float(len(predictions)) * 100
        print("\n{}".format(test_data[6]))
        print("Result: {}/{} [{:.2f}%]".format(success, len(predictions), rate))


if __name__ == '__main__':
    main()