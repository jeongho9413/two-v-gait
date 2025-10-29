import argparse
import os
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from twovgait.models.twovgait_2023 import TwoVGaitNonPen


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset_path', type=str, default='./datasets/KUGait_VLP32C_2022-Spring-C/train')
    parser.add_argument('--model_path', type=str, default='./pretrained_20221201')
    parser.add_argument('--batchsize', type=int, default='42')
    parser.add_argument('--nepoch', type=int, default='50')
    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_grad_enabled(False)

    args = parse_config()

    os.makedirs(args.model_path, exist_ok=True)

    train_x_dep = np.load(os.path.join(args.dataset_path, "train_x_mask_gde.npy"))
    train_x_speed = np.load(os.path.join(args.dataset_path, "train_x_speed.npy"))
    train_y = np.load(os.path.join(args.dataset_path, "train_y.npy"))

    train_x_dep = torch.tensor(train_x_dep).to(device)
    train_x_speed =  torch.tensor(train_x_speed).to(device)
    train_y = torch.tensor(train_y).type(torch.LongTensor).to(device)

    # dataset = TensorDataset(train_x, train_y)
    # dataloader = DataLoader(dataset, batchsize=args.batchsize, shuffle=False)

    num_batch = int(len(train_x_dep)/args.batchsize)

    net = TwoVGaitNonPen().to(device)  # model: TwoVGait with non-pen
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    # rd_seed = 5
    # random.Random(rd_seed).shuffle(train_x)
    # random.Random(rd_seed).shuffle(train_y)

    writer = SummaryWriter(os.path.join("runs", args.model_path))

    for epoch in range(args.nepoch):
        for batch_idx in range(num_batch):
            sample_x_dep = train_x_dep[(batch_idx * args.batchsize):((batch_idx+1) * args.batchsize)]
            sample_x_speed = train_x_speed[(batch_idx * args.batchsize):((batch_idx+1) * args.batchsize)]
            sample_y = train_y[(batch_idx * args.batchsize):((batch_idx+1) * args.batchsize)]

            optimizer.zero_grad()
            net = net.train()

            pred_x = net(x_shape=sample_x_dep, x_speed=sample_x_speed, embed=False)
            loss = criterion(pred_x, sample_y)  # loss=F.nll_loss(pred_x, sample_y)

            # total_loss += loss.item()  # check
            # total_correct += get_num_correct(pred_x, sample_y)  # check
            loss.backward()
            optimizer.step()
            pred_choice = pred_x.data.max(1)[1]
            correct = pred_choice.eq(sample_y.data).cpu().sum()
            print('[%d/%d: %d/%d] train loss: %f accuracy: %f' % (epoch, args.nepoch, batch_idx, num_batch, loss.item(), correct.item()/float(args.batchsize)))

            batch_idx_interval = 42
            writter_interval = epoch * (num_batch)+batch_idx
            if batch_idx % batch_idx_interval == 0:
                writer.add_scalar('total_loss', loss.item(), writter_interval)
                writer.add_scalar('total_acc', correct.item()/float(args.batchsize), writter_interval)
                print('checkpoint!_' + str(writter_interval))

    #model_scripted = torch.jit.script(net) # Export to TorchScript
    #model_scripted.save(PATH+'_model_scripted.pt')
    torch.save(net, './model.pth')
    torch.save(net.state_dict(), './model_state_dict.pth')
    torch.save(optimizer.state_dict(), './opt_state_dict.pth')


if __name__ == '__main__':
    main()