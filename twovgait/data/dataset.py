import glob
import os
import sys
import math

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel as parallel

#from torch.utils.data import TensorDataset, DataLoader
#from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter


class GenerateOrthoDep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        width = 1.78  # default: 1.5
        height = 2.58  # default: 2.2
        width_res = 0.04
        height_res = 0.04
        height_margin = 0.4  # default: 0.2
        f_rps = 10
        norm_speed = 1.5

        B = len(input)  # input.shape: (B, L, N, C), C_info: (x, y, z, intensity, ring's num)
        L = len(input[1])  # L = len(input[2])
        C = input[0][1].size()[1]  # C = input[0][2].size()[1]

        # gait image sequences on orthogonal coordinates
        seq_dep_side = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_dep_side.shape = (B, L, 64, 44)
        seq_dep_back = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_dep_back.shape = (B, L, 64, 44)
        seq_dep_bev = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_dep_bev.shape = (B, L, 64, 44)
        seq_dep_nongde = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_dep_nongde.shape = (B, L, 64, 44)

        seq_mask_side = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_mask_side.shape = (B, L, 64, 44)
        seq_mask_back = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_mask_back.shape = (B, L, 64, 44)
        seq_mask_bev = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_mask_bev.shape = (B, L, 64, 44)
        seq_mask_nongde = torch.zeros(B, L, int(height//height_res), int(width//width_res))  # seq_mask_nongde.shape = (B, L, 64, 44)

        seq_speed = torch.zeros(B, L-2, 1)

        # generate side-view images
        for b in range(B):
            z_min = torch.min(input[b][0], dim=0)[0][2].float()  # z-minimum of an img seq
            cent_xy_start = torch.mean(input[b][0], dim=0)[0:2]
            cent_xy_end = torch.mean(input[b][-1], dim=0)[0:2]
            direct_xy = cent_xy_end - cent_xy_start
            ang_temp = -(math.atan2(direct_xy[0], direct_xy[1])) - math.pi*(3/2)

            for l in range(L):
                # generate seq_speed
                if (l != 0) and (l != L-1):
                    xy_before = torch.mean(input[b][l-1], dim=0)[0:2]
                    xy_after = torch.mean(input[b][l+1], dim=0)[0:2]
                    seq_speed[b][l-1][0] = f_rps * (1/2) * torch.norm(xy_after - xy_before, dim=-1) * (1/norm_speed)

                # input_temp = input[b][l][:, 0:3]
                input_temp = input[b][l].clone().detach()[:, 0:3]
                cent_xy = torch.mean(input_temp, dim=0)[0:2]
                input_temp = input_temp - torch.Tensor([cent_xy[0], cent_xy[1], 0])
                rt_mat = torch.Tensor([[math.cos(ang_temp), math.sin(ang_temp), 0], 
                                        [-math.sin(ang_temp), math.cos(ang_temp), 0], 
                                        [0, 0, 1]]).view(1, 3, 3).repeat(input_temp.size()[0], 1, 1)
                input_xyz = torch.permute(torch.unsqueeze(input_temp, 1), (0, 2, 1)).float()
                input_temp = torch.squeeze(torch.bmm(rt_mat, input_xyz), -1)
                input_temp = input_temp + torch.Tensor([0, 0, -z_min]) + torch.Tensor([0, 0, height_margin]) + torch.Tensor([width/2, width/2, 0]).float()

                for n in range(input_temp.size()[0]):
                    if (( 0 <= input_temp[n][2] < height-height_res ) and ( 0 <= input_temp[n][0] < width-width_res )):
                            if seq_dep_side[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                            width_res, rounding_mode='floor'))] <= input_temp[n][1]:
                                seq_dep_side[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                                width_res, rounding_mode='floor'))] = input_temp[n][1]
                            seq_mask_side[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                            width_res, rounding_mode='floor'))] = 1

        # generate back-view images
        for b in range(B):
            z_min = torch.min(input[b][0], dim=0)[0][2].float()  # z-minimum of an img seq
            cent_xy_start = torch.mean(input[b][0], dim=0)[0:2]
            cent_xy_end = torch.mean(input[b][-1], dim=0)[0:2]
            direct_xy = cent_xy_end - cent_xy_start
            ang_temp = -(math.atan2(direct_xy[0], direct_xy[1])) - math.pi*(2/2)

            for l in range(L):
                # input_temp = input[b][l][:, 0:3]
                input_temp = input[b][l].clone().detach()[:, 0:3]
                cent_xy = torch.mean(input_temp, dim=0)[0:2]
                input_temp = input_temp - torch.Tensor([cent_xy[0], cent_xy[1], 0])
                rt_mat = torch.Tensor([[math.cos(ang_temp), math.sin(ang_temp), 0], 
                                        [-math.sin(ang_temp), math.cos(ang_temp), 0], 
                                        [0, 0, 1]]).view(1, 3, 3).repeat(input_temp.size()[0], 1, 1)
                input_xyz = torch.permute(torch.unsqueeze(input_temp, 1), (0, 2, 1)).float()
                input_temp = torch.squeeze(torch.bmm(rt_mat, input_xyz), -1)
                input_temp = input_temp + torch.Tensor([0, 0, -z_min]) + torch.Tensor([0, 0, height_margin]) + torch.Tensor([width/2, width/2, 0]).float()

                for n in range(input_temp.size()[0]):
                    if (( 0 <= input_temp[n][2] < height-height_res ) and ( 0 <= input_temp[n][0] < width-width_res )):
                            if seq_dep_back[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                            width_res, rounding_mode='floor'))] <= input_temp[n][1]:
                                seq_dep_back[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                                width_res, rounding_mode='floor'))] = input_temp[n][1]
                            seq_mask_back[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                            width_res, rounding_mode='floor'))] = 1

        # generate images with non-gde
        for b in range(B):
            z_min = torch.min(input[b][0], dim=0)[0][2].float()  # z-minimum of an img seq
            cent_xy_start = torch.mean(input[b][0], dim=0)[0:2]
            cent_xy_end = torch.mean(input[b][-1], dim=0)[0:2]
            direct_xy = cent_xy_end - cent_xy_start
            # ang_temp = -(math.atan2(direct_xy[0], direct_xy[1])) - math.pi*(2/2)
            ang_temp = math.pi*(1/2)

            for l in range(L):
                # input_temp = input[b][l][:, 0:3]
                input_temp = input[b][l].clone().detach()[:, 0:3]
                cent_xy = torch.mean(input_temp, dim=0)[0:2]
                input_temp = input_temp - torch.Tensor([cent_xy[0], cent_xy[1], 0])
                rt_mat = torch.Tensor([[math.cos(ang_temp), math.sin(ang_temp), 0], 
                                        [-math.sin(ang_temp), math.cos(ang_temp), 0], 
                                        [0, 0, 1]]).view(1, 3, 3).repeat(input_temp.size()[0], 1, 1)
                input_xyz = torch.permute(torch.unsqueeze(input_temp, 1), (0, 2, 1)).float()
                input_temp = torch.squeeze(torch.bmm(rt_mat, input_xyz), -1)
                input_temp = input_temp + torch.Tensor([0, 0, -z_min]) + torch.Tensor([0, 0, height_margin]) + torch.Tensor([width/2, width/2, 0]).float()

                for n in range(input_temp.size()[0]):
                    if (( 0 <= input_temp[n][2] < height-height_res ) and ( 0 <= input_temp[n][0] < width-width_res )):
                            if seq_dep_nongde[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                            width_res, rounding_mode='floor'))] <= input_temp[n][1]:
                                seq_dep_nongde[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                                width_res, rounding_mode='floor'))] = input_temp[n][1]
                            seq_mask_nongde[b][l][int(torch.div(input_temp[n][2], height_res, rounding_mode='floor'))][int(torch.div(input_temp[n][0], 
                                            width_res, rounding_mode='floor'))] = 1

        # side-view
        seq_dep_side = torch.flip(seq_dep_side, [2, 3]) / width
        seq_mask_side = torch.flip(seq_mask_side, [2, 3])
        gei_side = torch.sum(seq_mask_side, dim=1, keepdim=True)
        gei_side = torch.div(gei_side, L)
        # gei_side = torch.squeeze(gei_side, dim=1)

        # back-view
        seq_dep_back = torch.flip(seq_dep_back, [2, 3]) / width
        seq_mask_back = torch.flip(seq_mask_back, [2, 3])
        gei_back = torch.sum(seq_mask_back, dim=1, keepdim=True)
        gei_back = torch.div(gei_back, L)
        # gei_back = torch.squeeze(gei_back, dim=1)

        # bev
        seq_dep_bev = torch.flip(seq_dep_bev, [2, 3]) / width
        seq_mask_bev = torch.flip(seq_mask_bev, [2, 3])
        gei_bev = torch.sum(seq_mask_bev, dim=1, keepdim=True)
        gei_bev = torch.div(gei_bev, L)
        # gei_bev = torch.squeeze(gei_bev, dim=1)

        # nongde
        seq_dep_nongde = torch.flip(seq_dep_nongde, [2, 3]) / width
        seq_mask_nongde = torch.flip(seq_mask_nongde, [2, 3])
        gei_nongde = torch.sum(seq_mask_nongde, dim=1, keepdim=True)
        gei_nongde = torch.div(gei_nongde, L)
        # gei_nongde = torch.squeeze(gei_nongde, dim=1)

        # torch2numpy
        seq_dep_side = seq_dep_side.cpu().detach().numpy()
        seq_mask_side = seq_mask_side.cpu().detach().numpy()
        gei_side = gei_side.cpu().detach().numpy()

        seq_dep_back = seq_dep_back.cpu().detach().numpy()
        seq_mask_back = seq_mask_back.cpu().detach().numpy()
        gei_back = gei_back.cpu().detach().numpy()

        seq_dep_bev = seq_dep_bev.cpu().detach().numpy()
        seq_mask_bev = seq_mask_bev.cpu().detach().numpy()
        gei_bev = gei_bev.cpu().detach().numpy()

        seq_dep_nongde = seq_dep_nongde.cpu().detach().numpy()
        seq_mask_nongde = seq_mask_nongde.cpu().detach().numpy()
        gei_nongde = gei_nongde.cpu().detach().numpy()

        seq_speed = seq_speed.cpu().detach().numpy()
            
        # check the data shape
        print('\nseq_dep_side.shape = ', seq_dep_side.shape)
        print('seq_mask_side.shape = ', seq_mask_side.shape)
        print('gei_side.shape = ', gei_side.shape)

        print('\nseq_dep_back.shape = ', seq_dep_back.shape)
        print('seq_mask_back.shape = ', seq_mask_back.shape)
        print('gei_back.shape = ', gei_back.shape)

        print('\nseq_dep_bev.shape = ', seq_dep_bev.shape)
        print('seq_mask_bev.shape = ', seq_mask_bev.shape)
        print('gei_bev.shape = ', gei_bev.shape)

        print('\nseq_dep_nongde.shape = ', seq_dep_nongde.shape)
        print('seq_mask_nongde.shape = ', seq_mask_nongde.shape)
        print('gei_nongde.shape = ', gei_nongde.shape)

        return seq_dep_side, seq_mask_side, gei_side, seq_dep_back, seq_mask_back, gei_back, seq_dep_nongde, seq_mask_nongde, gei_nongde, seq_speed


def build_train_set(ped_path, yml_file):
    database_name = "kugait_vlp32c_2022-spring-c"
    
    if yml_file['database_name'] != database_name:
        raise ValueError(
            f"Unsupported database_name '{database_name}'."
        )
    
    print(f"selected database: {database_name}")

    # dist_eval
    if yml_file['dist_eval'] == '1020m':
        dist_list = ['10m', '20m']
    elif yml_file['dist_eval'] == '102030m':
        dist_list = ['10m', '20m', '30m']
    else:
        raise ValueError(f"Unsupported dist_eval '{yml_file['dist_eval']}'.")

    # seq_frames
    if yml_file['seq_frames'] == 10:
        seq_frames = 10
    elif yml_file['seq_frames'] == 15:
        seq_frames = 15
    elif yml_file['seq_frames'] == 20:
        seq_frames = 20
    else:
        raise ValueError(f"Unsupported seq_frames '{yml_file['seq_frames']}'.")

    # split_TrainVal
    if yml_file['split_TrainVal'] == '6and0':
        train_num_rate = 6/6
        val_num_rate = 0/6
    elif yml_file['split_TrainVal'] == '5and1':
        train_num_rate = 5/6
        val_num_rate = 1/6
    elif yml_file['split_TrainVal'] == '4and2':
        train_num_rate = 4/6
        val_num_rate = 2/6
    else:
        raise ValueError(f"Unsupported split_train-val '{yml_file['split_TrainVal']}'.")

    # create lists
    train_x_list = list()
    train_y_list = list()
    # val_x_list = list()
    # val_y_list = list()

    # build_datasets
    ang_list = ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']
    for train_ped in list(yml_file['subject_train']):
        for dist_temp in dist_list:
            for ang_temp in ang_list:
                num_list_path = os.path.join(ped_path, train_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))

                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))

                train_len = int(len(num_list) * train_num_rate)
                train_num_list = num_list[:train_len]
                val_num_list = num_list[train_len:]

                for num_temp in train_num_list:
                    npy_path = os.path.join(ped_path, train_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, num_temp)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    train_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        train_seq.append(torch.from_numpy(npy_data))  # when using the data as a torch tensor
                        #train_seq.append(npy_data)  # when using the data as a npy
                    train_x_list.append(train_seq)

                    if train_ped == 'ahn':
                        train_y_list.append(0)
                    elif train_ped == 'an':
                        train_y_list.append(1)
                    elif train_ped == 'aoki':
                        train_y_list.append(2)
                    elif train_ped == 'arise':
                        train_y_list.append(3)
                    elif train_ped == 'azumi':
                        train_y_list.append(4)
                    elif train_ped == 'chang':
                        train_y_list.append(5)
                    elif train_ped == 'danzyou':
                        train_y_list.append(6)
                    elif train_ped == 'fukuda':
                        train_y_list.append(7)
                    elif train_ped == 'hayashi':
                        train_y_list.append(8)
                    elif train_ped == 'itsuka':
                        train_y_list.append(9)
                    elif train_ped == 'kawasaki2':
                        train_y_list.append(10)
                    elif train_ped == 'kihara':
                        train_y_list.append(11)
                    elif train_ped == 'koga':
                        train_y_list.append(12)
                    elif train_ped == 'kurazume':
                        train_y_list.append(13)
                    elif train_ped == 'kuroda':
                        train_y_list.append(14)
                    elif train_ped == 'matsumoto':
                        train_y_list.append(15)
                    elif train_ped == 'morinaga':
                        train_y_list.append(16)
                    elif train_ped == 'nakashima':
                        train_y_list.append(17)
                    elif train_ped == 'nakashima2':
                        train_y_list.append(18)
                    elif train_ped == 'nishiura':
                        train_y_list.append(19)
                    else:
                        raise ValueError(f"Unsupported train_ped '{train_ped}'.")

    print('\nlen(train_x_list): ', len(train_x_list))
    print('len(train_y_list): ', len(train_y_list))
    #print('len(val_x_list): ', len(val_x_list))
    #print('len(val_y_list): ', len(val_y_list))

    return train_x_list, train_y_list


def build_test_set(ped_path, yml_file):
    database_name = "kugait_vlp32c_2022-spring-c"
    
    if yml_file['database_name'] != database_name:
        raise ValueError(
            f"Unsupported database_name '{database_name}'."
        )
    
    print(f"selected database: {database_name}")

    # dist_eval
    if yml_file['dist_eval'] == '1020m':
        dist_list = ['10m', '20m']
    elif yml_file['dist_eval'] == '102030m':
        dist_list = ['10m', '20m', '30m']
    else:
        raise ValueError(f"Unsupported dist_eval '{yml_file['dist_eval']}'.")

    # seq_frames
    if yml_file['seq_frames'] == 10:
        seq_frames = 10
    elif yml_file['seq_frames'] == 15:
        seq_frames = 15
    elif yml_file['seq_frames'] == 20:
        seq_frames = 20
    else:
        raise ValueError(f"Unsupported seq_frames '{yml_file['seq_frames']}'.")

    # split_gallery-probe
    if yml_file['split_GalleryProbe'] == '2and4':
        gallery_num_rate = 2/6
        probe_num_rate = 4/6
    elif yml_file['split_TrainVal'] == '4and2':
        gallery_num_rate = 4/6
        probe_num_rate = 2/6
    else:
        raise ValueError(f"Unsupported split_GalleryProbe '{yml_file['split_GalleryProbe']}'.")
    
    # create x_lists
    gallery_10m_0_x_list = []
    gallery_10m_45_x_list = []
    gallery_10m_90_x_list = []
    gallery_10m_135_x_list = []
    gallery_10m_180_x_list = []
    gallery_10m_225_x_list = []
    gallery_10m_270_x_list = []
    gallery_10m_315_x_list = []

    gallery_20m_0_x_list = []
    gallery_20m_45_x_list = []
    gallery_20m_90_x_list = []
    gallery_20m_135_x_list = []
    gallery_20m_180_x_list = []
    gallery_20m_225_x_list = []
    gallery_20m_270_x_list = []
    gallery_20m_315_x_list = []

    probe_10m_0_x_list = []
    probe_10m_45_x_list = []
    probe_10m_90_x_list = []
    probe_10m_135_x_list = []
    probe_10m_180_x_list = []
    probe_10m_225_x_list = []
    probe_10m_270_x_list = []
    probe_10m_315_x_list = []

    probe_20m_0_x_list = []
    probe_20m_45_x_list = []
    probe_20m_90_x_list = []
    probe_20m_135_x_list = []
    probe_20m_180_x_list = []
    probe_20m_225_x_list = []
    probe_20m_270_x_list = []
    probe_20m_315_x_list = []

    # create y_lists
    gallery_10m_0_y_list = []
    gallery_10m_45_y_list = []
    gallery_10m_90_y_list = []
    gallery_10m_135_y_list = []
    gallery_10m_180_y_list = []
    gallery_10m_225_y_list = []
    gallery_10m_270_y_list = []
    gallery_10m_315_y_list = []

    gallery_20m_0_y_list = []
    gallery_20m_45_y_list = []
    gallery_20m_90_y_list = []
    gallery_20m_135_y_list = []
    gallery_20m_180_y_list = []
    gallery_20m_225_y_list = []
    gallery_20m_270_y_list = []
    gallery_20m_315_y_list = []

    probe_10m_0_y_list = []
    probe_10m_45_y_list = []
    probe_10m_90_y_list = []
    probe_10m_135_y_list = []
    probe_10m_180_y_list = []
    probe_10m_225_y_list = []
    probe_10m_270_y_list = []
    probe_10m_315_y_list = []

    probe_20m_0_y_list = []
    probe_20m_45_y_list = []
    probe_20m_90_y_list = []
    probe_20m_135_y_list = []
    probe_20m_180_y_list = []
    probe_20m_225_y_list = []
    probe_20m_270_y_list = []
    probe_20m_315_y_list = []

    for test_ped in list(yml_file['subject_test']):
        print('test_ped: ', str(test_ped))

        for dist_temp in ['10m']:

            # gallery_10m_0_x_list
            # gallery_10m_0_y_list
            for ang_temp in ['45deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_0_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_0_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_0_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_0_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_0_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_0_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_0_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_0_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_0_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_0_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_0_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_10m_45_x_list
            # gallery_10m_45_y_list
            for ang_temp in ['0deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_45_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_45_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_45_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_45_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_45_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_45_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_45_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_45_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_45_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_45_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_45_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_10m_90_x_list
            # gallery_10m_90_y_list
            for ang_temp in ['0deg', '45deg', '135deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_90_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_90_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_90_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_90_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_90_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_90_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_90_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_90_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_90_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_90_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_90_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_10m_135_x_list
            # gallery_10m_135_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_135_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_135_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_135_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_135_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_135_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_135_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_135_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_135_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_135_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_135_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_135_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_10m_180_x_list
            # gallery_10m_180_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_180_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_180_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_180_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_180_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_180_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_180_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_180_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_180_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_180_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_180_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_180_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_10m_225_x_list
            # gallery_10m_225_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '180deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_225_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_225_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_225_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_225_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_225_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_225_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_225_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_225_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_225_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_225_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_225_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_10m_270_x_list
            # gallery_10m_270_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_270_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_270_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_270_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_270_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_270_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_270_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_270_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_270_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_270_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_270_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_270_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_10m_315_x_list
            # gallery_10m_315_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '270deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_10m_315_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_10m_315_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_10m_315_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_10m_315_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_10m_315_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_10m_315_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_10m_315_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_10m_315_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_10m_315_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_10m_315_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_10m_315_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

        for dist_temp in ['20m']:

            # gallery_20m_0_x_list
            # gallery_20m_0_y_list
            for ang_temp in ['45deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_0_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_0_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_0_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_0_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_0_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_0_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_0_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_0_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_0_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_0_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_0_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_20m_45_x_list
            # gallery_20m_45_y_list
            for ang_temp in ['0deg', '90deg', '135deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_45_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_45_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_45_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_45_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_45_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_45_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_45_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_45_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_45_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_45_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_45_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_20m_90_x_list
            # gallery_20m_90_y_list
            for ang_temp in ['0deg', '45deg', '135deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_90_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_90_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_90_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_90_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_90_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_90_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_90_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_90_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_90_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_90_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_90_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_20m_135_x_list
            # gallery_20m_135_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '180deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_135_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_135_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_135_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_135_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_135_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_135_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_135_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_135_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_135_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_135_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_135_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_20m_180_x_list
            # gallery_20m_180_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '225deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_180_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_180_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_180_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_180_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_180_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_180_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_180_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_180_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_180_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_180_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_180_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_20m_225_x_list
            # gallery_20m_225_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '180deg', '270deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_225_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_225_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_225_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_225_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_225_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_225_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_225_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_225_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_225_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_225_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_225_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_20m_270_x_list
            # gallery_20m_270_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_270_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_270_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_270_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_270_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_270_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_270_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_270_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_270_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_270_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_270_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_270_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # gallery_20m_315_x_list
            # gallery_20m_315_y_list
            for ang_temp in ['0deg', '45deg', '90deg', '135deg', '180deg', '225deg', '270deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for gallery_num in gallery_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, gallery_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    gallery_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        gallery_seq.append(torch.from_numpy(npy_data))
                        #gallery_seq.append(npy_data)
                    gallery_20m_315_x_list.append(gallery_seq)

                    if test_ped == 'ohki':
                        gallery_20m_315_y_list.append(0)
                    elif test_ped == 'ohno':
                        gallery_20m_315_y_list.append(1)
                    elif test_ped == 'takenaka':
                        gallery_20m_315_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        gallery_20m_315_y_list.append(3)
                    elif test_ped == 'tanaka':
                        gallery_20m_315_y_list.append(4)
                    elif test_ped == 'tian':
                        gallery_20m_315_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        gallery_20m_315_y_list.append(6)
                    elif test_ped == 'uno':
                        gallery_20m_315_y_list.append(7)
                    elif test_ped == 'xin':
                        gallery_20m_315_y_list.append(8)
                    elif test_ped == 'yoshino':
                        gallery_20m_315_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

        for dist_temp in ['10m']:

            # probe_10m_0_x_list
            # probe_10m_0_y_list
            for ang_temp in ['0deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_0_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_0_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_0_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_0_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_0_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_0_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_0_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_0_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_0_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_0_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_0_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_10m_45_x_list
            # probe_10m_45_y_list
            for ang_temp in ['45deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_45_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_45_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_45_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_45_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_45_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_45_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_45_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_45_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_45_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_45_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_45_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_10m_90_x_list
            # probe_10m_90_y_list
            for ang_temp in ['90deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_90_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_90_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_90_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_90_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_90_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_90_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_90_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_90_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_90_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_90_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_90_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_10m_135_x_list
            # probe_10m_135_y_list
            for ang_temp in ['135deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_135_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_135_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_135_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_135_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_135_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_135_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_135_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_135_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_135_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_135_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_135_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_10m_180_x_list
            # probe_10m_180_y_list
            for ang_temp in ['180deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_180_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_180_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_180_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_180_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_180_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_180_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_180_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_180_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_180_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_180_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_180_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_10m_225_x_list
            # probe_10m_225_y_list
            for ang_temp in ['225deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_225_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_225_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_225_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_225_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_225_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_225_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_225_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_225_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_225_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_225_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_225_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_10m_270_x_list
            # probe_10m_270_y_list
            for ang_temp in ['270deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_270_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_270_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_270_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_270_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_270_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_270_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_270_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_270_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_270_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_270_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_270_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_10m_315_x_list
            # probe_10m_315_y_list
            for ang_temp in ['315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_10m_315_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_10m_315_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_10m_315_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_10m_315_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_10m_315_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_10m_315_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_10m_315_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_10m_315_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_10m_315_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_10m_315_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_10m_315_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

        for dist_temp in ['20m']:

            # probe_20m_0_x_list
            # probe_20m_0_y_list
            for ang_temp in ['0deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_0_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_0_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_0_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_0_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_0_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_0_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_0_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_0_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_0_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_0_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_0_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_20m_45_x_list
            # probe_20m_45_y_list
            for ang_temp in ['45deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_45_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_45_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_45_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_45_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_45_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_45_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_45_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_45_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_45_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_45_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_45_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_20m_90_x_list
            # probe_20m_90_y_list
            for ang_temp in ['90deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_90_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_90_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_90_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_90_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_90_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_90_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_90_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_90_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_90_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_90_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_90_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_20m_135_x_list
            # probe_20m_135_y_list
            for ang_temp in ['135deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_135_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_135_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_135_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_135_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_135_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_135_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_135_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_135_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_135_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_135_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_135_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_20m_180_x_list
            # probe_20m_180_y_list
            for ang_temp in ['180deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_180_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_180_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_180_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_180_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_180_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_180_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_180_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_180_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_180_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_180_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_180_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_20m_225_x_list
            # probe_20m_225_y_list
            for ang_temp in ['225deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_225_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_225_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_225_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_225_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_225_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_225_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_225_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_225_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_225_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_225_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_225_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_20m_270_x_list
            # probe_20m_270_y_list
            for ang_temp in ['270deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_270_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_270_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_270_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_270_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_270_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_270_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_270_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_270_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_270_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_270_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_270_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

            # probe_20m_315_x_list
            # probe_20m_315_y_list
            for ang_temp in ['315deg']:
                num_list_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp)
                num_list_temp = sorted(glob.glob(num_list_path + '/*'))
                num_list = []
                for i in num_list_temp:
                    num_list.append(os.path.basename(i))
                gallery_len = int(len(num_list) * gallery_num_rate)
                gallery_num_list = num_list[:gallery_len]
                probe_num_list = num_list[gallery_len:]

                for probe_num in probe_num_list:
                    npy_path = os.path.join(ped_path, test_ped, 'npy', 'ped_'+str(seq_frames)+'frames', dist_temp, ang_temp, probe_num)
                    npy_list = sorted(glob.glob(npy_path + '/*.npy'))
                    probe_seq = []
                    for npy_file in npy_list:
                        npy_data = np.load(npy_file)
                        probe_seq.append(torch.from_numpy(npy_data))
                        #probe_seq.append(npy_data)
                    probe_20m_315_x_list.append(probe_seq)

                    if test_ped == 'ohki':
                        probe_20m_315_y_list.append(0)
                    elif test_ped == 'ohno':
                        probe_20m_315_y_list.append(1)
                    elif test_ped == 'takenaka':
                        probe_20m_315_y_list.append(2)
                    elif test_ped == 'tamashiro':
                        probe_20m_315_y_list.append(3)
                    elif test_ped == 'tanaka':
                        probe_20m_315_y_list.append(4)
                    elif test_ped == 'tian':
                        probe_20m_315_y_list.append(5)
                    elif test_ped == 'tomoeda':
                        probe_20m_315_y_list.append(6)
                    elif test_ped == 'uno':
                        probe_20m_315_y_list.append(7)
                    elif test_ped == 'xin':
                        probe_20m_315_y_list.append(8)
                    elif test_ped == 'yoshino':
                        probe_20m_315_y_list.append(9)
                    else:
                        raise ValueError("Pedestrian not found.")

    # check gallery x lists (dist: 10-20m, gallery-probe: 2-4, total gallery nums: 2940)
    print('\nlen(gallery_10m_0_x_list): ', len(gallery_10m_0_x_list))
    print('len(gallery_10m_45_x_list): ', len(gallery_10m_45_x_list))
    print('len(gallery_10m_90_x_list): ', len(gallery_10m_90_x_list)) 
    print('len(gallery_10m_135_x_list): ', len(gallery_10m_135_x_list)) 
    print('len(gallery_10m_180_x_list): ', len(gallery_10m_180_x_list)) 
    print('len(gallery_10m_225_x_list): ', len(gallery_10m_225_x_list))
    print('len(gallery_10m_270_x_list): ', len(gallery_10m_270_x_list))
    print('len(gallery_10m_315_x_list): ', len(gallery_10m_315_x_list)) 

    print('\nlen(gallery_20m_0_x_list): ', len(gallery_20m_0_x_list))
    print('len(gallery_20m_45_x_list): ', len(gallery_20m_45_x_list))
    print('len(gallery_20m_90_x_list): ', len(gallery_20m_90_x_list))
    print('len(gallery_20m_135_x_list): ', len(gallery_20m_135_x_list))
    print('len(gallery_20m_180_x_list): ', len(gallery_20m_180_x_list))
    print('len(gallery_20m_225_x_list): ', len(gallery_20m_225_x_list))
    print('len(gallery_20m_270_x_list): ', len(gallery_20m_270_x_list))
    print('len(gallery_20m_315_x_list): ', len(gallery_20m_315_x_list))

    # check probe x lists (dist: 10-20m, gallery-probe: 2-4, total probe nums: 840)
    print('\nlen(probe_10m_0_x_list): ', len(probe_10m_0_x_list))
    print('len(probe_10m_45_x_list): ', len(probe_10m_45_x_list))
    print('len(probe_10m_90_x_list): ', len(probe_10m_90_x_list))
    print('len(probe_10m_135_x_list): ', len(probe_10m_135_x_list))
    print('len(probe_10m_180_x_list): ', len(probe_10m_180_x_list))
    print('len(probe_10m_225_x_list): ', len(probe_10m_225_x_list))
    print('len(probe_10m_270_x_list): ', len(probe_10m_270_x_list))
    print('len(probe_10m_315_x_list): ', len(probe_10m_315_x_list))

    print('\nlen(probe_20m_0_x_list): ', len(probe_20m_0_x_list))
    print('len(probe_20m_45_x_list): ', len(probe_20m_45_x_list))
    print('len(probe_20m_90_x_list): ', len(probe_20m_90_x_list))
    print('len(probe_20m_135_x_list): ', len(probe_20m_135_x_list))
    print('len(probe_20m_180_x_list): ', len(probe_20m_180_x_list))
    print('len(probe_20m_225_x_list): ', len(probe_20m_225_x_list))
    print('len(probe_20m_270_x_list): ', len(probe_20m_270_x_list))
    print('len(probe_20m_315_x_list): ', len(probe_20m_315_x_list))

    # 
    net = GenerateOrthoDep()

    # save datasets
    file_path = './datasets/KUGait_VLP32C_2022-Spring-C/test_'
    os.makedirs(file_path, exist_ok=True)

    # create gallery_10m_x_inputs
    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_0_x_list)
    np.save(file_path+'/gallery_10m_0_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_0_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_0_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_0_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_0_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_0_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_0_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_0_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_0_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_0_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_45_x_list)
    np.save(file_path+'/gallery_10m_45_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_45_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_45_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_45_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_45_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_45_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_45_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_45_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_45_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_45_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_90_x_list)
    np.save(file_path+'/gallery_10m_90_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_90_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_90_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_90_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_90_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_90_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_90_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_90_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_90_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_90_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_135_x_list)
    np.save(file_path+'/gallery_10m_135_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_135_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_135_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_135_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_135_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_135_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_135_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_135_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_135_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_135_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_180_x_list)
    np.save(file_path+'/gallery_10m_180_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_180_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_180_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_180_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_180_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_180_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_180_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_180_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_180_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_180_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_225_x_list)
    np.save(file_path+'/gallery_10m_225_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_225_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_225_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_225_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_225_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_225_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_225_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_225_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_225_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_225_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_270_x_list)
    np.save(file_path+'/gallery_10m_270_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_270_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_270_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_270_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_270_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_270_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_270_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_270_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_270_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_270_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_10m_315_x_list)
    np.save(file_path+'/gallery_10m_315_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_10m_315_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_10m_315_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_10m_315_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_10m_315_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_10m_315_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_10m_315_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_10m_315_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_10m_315_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_10m_315_x_speed.npy', data_x_speed)

    # create gallery_20m_x_inputs
    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_0_x_list)
    np.save(file_path+'/gallery_20m_0_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_0_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_0_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_0_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_0_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_0_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_0_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_0_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_0_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_0_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_45_x_list)
    np.save(file_path+'/gallery_20m_45_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_45_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_45_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_45_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_45_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_45_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_45_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_45_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_45_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_45_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_90_x_list)
    np.save(file_path+'/gallery_20m_90_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_90_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_90_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_90_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_90_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_90_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_90_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_90_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_90_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_90_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_135_x_list)
    np.save(file_path+'/gallery_20m_135_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_135_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_135_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_135_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_135_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_135_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_135_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_135_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_135_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_135_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_180_x_list)
    np.save(file_path+'/gallery_20m_180_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_180_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_180_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_180_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_180_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_180_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_180_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_180_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_180_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_180_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_225_x_list)
    np.save(file_path+'/gallery_20m_225_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_225_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_225_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_225_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_225_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_225_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_225_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_225_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_225_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_225_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_270_x_list)
    np.save(file_path+'/gallery_20m_270_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_270_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_270_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_270_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_270_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_270_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_270_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_270_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_270_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_270_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(gallery_20m_315_x_list)
    np.save(file_path+'/gallery_20m_315_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/gallery_20m_315_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/gallery_20m_315_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/gallery_20m_315_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/gallery_20m_315_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/gallery_20m_315_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/gallery_20m_315_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/gallery_20m_315_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/gallery_20m_315_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/gallery_20m_315_x_speed.npy', data_x_speed)

    # create probe_10m_x_inputs
    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_0_x_list)
    np.save(file_path+'/probe_10m_0_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_0_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_0_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_0_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_0_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_0_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_0_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_0_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_0_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_0_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_45_x_list)
    np.save(file_path+'/probe_10m_45_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_45_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_45_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_45_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_45_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_45_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_45_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_45_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_45_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_45_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_90_x_list)
    np.save(file_path+'/probe_10m_90_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_90_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_90_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_90_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_90_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_90_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_90_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_90_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_90_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_90_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_135_x_list)
    np.save(file_path+'/probe_10m_135_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_135_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_135_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_135_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_135_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_135_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_135_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_135_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_135_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_135_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_180_x_list)
    np.save(file_path+'/probe_10m_180_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_180_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_180_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_180_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_180_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_180_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_180_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_180_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_180_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_180_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_225_x_list)
    np.save(file_path+'/probe_10m_225_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_225_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_225_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_225_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_225_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_225_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_225_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_225_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_225_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_225_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_270_x_list)
    np.save(file_path+'/probe_10m_270_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_270_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_270_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_270_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_270_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_270_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_270_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_270_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_270_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_270_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_10m_315_x_list)
    np.save(file_path+'/probe_10m_315_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_10m_315_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_10m_315_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_10m_315_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_10m_315_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_10m_315_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_10m_315_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_10m_315_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_10m_315_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_10m_315_x_speed.npy', data_x_speed)

    # create probe_20m_x_inputs
    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_0_x_list)
    np.save(file_path+'/probe_20m_0_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_0_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_0_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_0_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_0_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_0_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_0_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_0_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_0_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_0_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_45_x_list)
    np.save(file_path+'/probe_20m_45_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_45_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_45_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_45_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_45_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_45_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_45_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_45_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_45_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_45_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_90_x_list)
    np.save(file_path+'/probe_20m_90_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_90_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_90_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_90_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_90_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_90_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_90_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_90_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_90_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_90_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_135_x_list)
    np.save(file_path+'/probe_20m_135_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_135_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_135_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_135_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_135_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_135_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_135_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_135_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_135_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_135_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_180_x_list)
    np.save(file_path+'/probe_20m_180_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_180_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_180_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_180_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_180_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_180_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_180_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_180_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_180_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_180_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_225_x_list)
    np.save(file_path+'/probe_20m_225_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_225_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_225_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_225_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_225_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_225_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_225_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_225_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_225_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_225_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_270_x_list)
    np.save(file_path+'/probe_20m_270_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_270_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_270_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_270_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_270_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_270_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_270_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_270_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_270_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_270_x_speed.npy', data_x_speed)

    data_x_dep_side, data_x_mask_side, data_x_gei_side, data_x_dep_back, data_x_mask_back, data_x_gei_back, data_x_dep_nongde, data_x_mask_nongde, data_x_gei_nongde, data_x_speed = net(probe_20m_315_x_list)
    np.save(file_path+'/probe_20m_315_x_dep_side.npy', data_x_dep_side)
    np.save(file_path+'/probe_20m_315_x_gei_side.npy', data_x_gei_side)
    np.save(file_path+'/probe_20m_315_x_mask_side.npy', data_x_mask_side)

    np.save(file_path+'/probe_20m_315_x_dep_back.npy', data_x_dep_back)
    np.save(file_path+'/probe_20m_315_x_gei_back.npy', data_x_gei_back)
    np.save(file_path+'/probe_20m_315_x_mask_back.npy', data_x_mask_back)

    np.save(file_path+'/probe_20m_315_x_dep_nongde.npy', data_x_dep_nongde)
    np.save(file_path+'/probe_20m_315_x_gei_nongde.npy', data_x_gei_nongde)
    np.save(file_path+'/probe_20m_315_x_mask_nongde.npy', data_x_mask_nongde)

    np.save(file_path+'/probe_20m_315_x_speed.npy', data_x_speed)

    # create gallery y & probe y
    gallery_10m_0_y = np.asarray(gallery_10m_0_y_list, dtype=np.int16)
    gallery_10m_45_y = np.asarray(gallery_10m_45_y_list, dtype=np.int16)
    gallery_10m_90_y = np.asarray(gallery_10m_90_y_list, dtype=np.int16)
    gallery_10m_135_y = np.asarray(gallery_10m_135_y_list, dtype=np.int16)
    gallery_10m_180_y = np.asarray(gallery_10m_180_y_list, dtype=np.int16)
    gallery_10m_225_y = np.asarray(gallery_10m_225_y_list, dtype=np.int16)
    gallery_10m_270_y = np.asarray(gallery_10m_270_y_list, dtype=np.int16)
    gallery_10m_315_y = np.asarray(gallery_10m_315_y_list, dtype=np.int16)

    gallery_20m_0_y = np.asarray(gallery_20m_0_y_list, dtype=np.int16)
    gallery_20m_45_y = np.asarray(gallery_20m_45_y_list, dtype=np.int16)
    gallery_20m_90_y = np.asarray(gallery_20m_90_y_list, dtype=np.int16)
    gallery_20m_135_y = np.asarray(gallery_20m_135_y_list, dtype=np.int16)
    gallery_20m_180_y = np.asarray(gallery_20m_180_y_list, dtype=np.int16)
    gallery_20m_225_y = np.asarray(gallery_20m_225_y_list, dtype=np.int16)
    gallery_20m_270_y = np.asarray(gallery_20m_270_y_list, dtype=np.int16)
    gallery_20m_315_y = np.asarray(gallery_20m_315_y_list, dtype=np.int16)

    probe_10m_0_y = np.asarray(probe_10m_0_y_list, dtype=np.int16)
    probe_10m_45_y = np.asarray(probe_10m_45_y_list, dtype=np.int16)
    probe_10m_90_y = np.asarray(probe_10m_90_y_list, dtype=np.int16)
    probe_10m_135_y = np.asarray(probe_10m_135_y_list, dtype=np.int16)
    probe_10m_180_y = np.asarray(probe_10m_180_y_list, dtype=np.int16)
    probe_10m_225_y = np.asarray(probe_10m_225_y_list, dtype=np.int16)
    probe_10m_270_y = np.asarray(probe_10m_270_y_list, dtype=np.int16)
    probe_10m_315_y = np.asarray(probe_10m_315_y_list, dtype=np.int16)

    probe_20m_0_y = np.asarray(probe_20m_0_y_list, dtype=np.int16)
    probe_20m_45_y = np.asarray(probe_20m_45_y_list, dtype=np.int16)
    probe_20m_90_y = np.asarray(probe_20m_90_y_list, dtype=np.int16)
    probe_20m_135_y = np.asarray(probe_20m_135_y_list, dtype=np.int16)
    probe_20m_180_y = np.asarray(probe_20m_180_y_list, dtype=np.int16)
    probe_20m_225_y = np.asarray(probe_20m_225_y_list, dtype=np.int16)
    probe_20m_270_y = np.asarray(probe_20m_270_y_list, dtype=np.int16)
    probe_20m_315_y = np.asarray(probe_20m_315_y_list, dtype=np.int16)

    # save gallery y lists
    np.save(file_path+'/gallery_10m_0_y.npy', gallery_10m_0_y)
    np.save(file_path+'/gallery_10m_45_y.npy', gallery_10m_45_y)
    np.save(file_path+'/gallery_10m_90_y.npy', gallery_10m_90_y)
    np.save(file_path+'/gallery_10m_135_y.npy', gallery_10m_135_y)
    np.save(file_path+'/gallery_10m_180_y.npy', gallery_10m_180_y)
    np.save(file_path+'/gallery_10m_225_y.npy', gallery_10m_225_y)
    np.save(file_path+'/gallery_10m_270_y.npy', gallery_10m_270_y)
    np.save(file_path+'/gallery_10m_315_y.npy', gallery_10m_315_y)

    np.save(file_path+'/gallery_20m_0_y.npy', gallery_20m_0_y)
    np.save(file_path+'/gallery_20m_45_y.npy', gallery_20m_45_y)
    np.save(file_path+'/gallery_20m_90_y.npy', gallery_20m_90_y)
    np.save(file_path+'/gallery_20m_135_y.npy', gallery_20m_135_y)
    np.save(file_path+'/gallery_20m_180_y.npy', gallery_20m_180_y)
    np.save(file_path+'/gallery_20m_225_y.npy', gallery_20m_225_y)
    np.save(file_path+'/gallery_20m_270_y.npy', gallery_20m_270_y)
    np.save(file_path+'/gallery_20m_315_y.npy', gallery_20m_315_y)

    # save probe y lists
    np.save(file_path+'/probe_10m_0_y.npy', probe_10m_0_y)
    np.save(file_path+'/probe_10m_45_y.npy', probe_10m_45_y)
    np.save(file_path+'/probe_10m_90_y.npy', probe_10m_90_y)
    np.save(file_path+'/probe_10m_135_y.npy', probe_10m_135_y)
    np.save(file_path+'/probe_10m_180_y.npy', probe_10m_180_y)
    np.save(file_path+'/probe_10m_225_y.npy', probe_10m_225_y)
    np.save(file_path+'/probe_10m_270_y.npy', probe_10m_270_y)
    np.save(file_path+'/probe_10m_315_y.npy', probe_10m_315_y)

    np.save(file_path+'/probe_20m_0_y.npy', probe_20m_0_y)
    np.save(file_path+'/probe_20m_45_y.npy', probe_20m_45_y)
    np.save(file_path+'/probe_20m_90_y.npy', probe_20m_90_y)
    np.save(file_path+'/probe_20m_135_y.npy', probe_20m_135_y)
    np.save(file_path+'/probe_20m_180_y.npy', probe_20m_180_y)
    np.save(file_path+'/probe_20m_225_y.npy', probe_20m_225_y)
    np.save(file_path+'/probe_20m_270_y.npy', probe_20m_270_y)
    np.save(file_path+'/probe_20m_315_y.npy', probe_20m_315_y)