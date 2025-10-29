from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.nn.parallel as parallel


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = 4 * self.hidden_dim,
                              kernel_size = self.kernel_size,
                              padding = self.padding,
                              bias = self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # since the init is done in forward. 
            # can send image size here.
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class TwoVGait(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convlstm1 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        self.fc4 = nn.Linear(8064, 128)    ## res004: 8064
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm1d(129, affine=False)

    def forward(self, input_dep, input_speed):

        # mini-batchsize
        B = input_dep.size()[0]
        L = input_dep.size()[1]

        # value normalization
        # seq_depth = torch.flip(seq_depth, [2, 3])
        # seq_speed = (seq_speed/norm_speed).cuda()
        # seq_depth = (torch.unsqueeze(input_shape, dim=2))/width
        # seq_speed = input_speed

        # density adaptive encoding
        dep_res_high = input_dep
        # dep_res_high = dep_res_high.view(B*L, 1, dep_res_high.size(3), dep_res_high.size(4))
        dep_res_high = dep_res_high.view(B*L, 1, dep_res_high.size(2), dep_res_high.size(3))
        dep_res_low = self.up1(self.pool1(dep_res_high))

        output_list = []
        for dep_map in [dep_res_high, dep_res_low]:
            dep_map = F.relu(self.conv2(dep_map))
            dep_map = self.pool3(F.relu(self.conv3(dep_map)))
            output_list.append(dep_map)

        ssf = (1/2) * (output_list[0] + output_list[1])

        # temporal feature aggregating
        ssf = ssf.view(B, L, 32, ssf.size(2), ssf.size(3))
        ssf = self.convlstm1(ssf)
        ssf = (ssf[0][0])[:, -1, :, :, :]
        ssf = self.bn4(self.pool4(ssf))
        ssf = torch.flatten(ssf, 1)
        f_sptial = self.fc4(ssf)

        ## positional feature concatenating
        input_speed = input_speed.view(B*(L-2), 1, 1)
        input_speed = self.conv5(input_speed)
        input_speed = input_speed.view(B, (L-2))
        f_speed = torch.mean(input_speed, 1)
        f_speed = torch.unsqueeze(f_speed, 1)

        f_gait = torch.cat((f_sptial, f_speed), dim=1)
        f_gait = self.bn6(F.relu(f_gait))

        return f_gait
    

class TwoVGaitNonPen(nn.Module):
    def __init__(self, num_peds=20):
        super(TwoVGaitNonPen, self).__init__()
        self.identifier = TwoVGait()
        self.fc_final = nn.Linear(129, num_peds)

    def forward(self, x_shape, x_speed, embed=False):
        f_gait = self.identifier(x_shape, x_speed)
        #print(x.size())

        if embed:
            return f_gait
        pred_x = self.fc_final(f_gait)
        #print(pred_x.size())
        return pred_x