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
        
        #self.weight_conv2d_1 = torch.randn(32, 1, 5, 5).cuda()
        
        self.conv1_2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, dilation=1, padding=0, stride=1)
        self.conv1_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, dilation=1, padding=0, stride=1)
        #self.conv1_2_dil = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, dilation=2, padding=2, stride=1)
        #self.conv1_3_dil = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, dilation=2, padding=2, stride=1)
        self.pool1_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        factor_c_attn = 2
        self.conv2_enc = nn.Conv2d(in_channels=32, out_channels=32//factor_c_attn, kernel_size=1, dilation=1, padding=0, stride=1)
        self.conv2_dec = nn.Conv2d(in_channels=32//factor_c_attn, out_channels=32, kernel_size=1, dilation=1, padding=0, stride=1)
        self.bn_attn = nn.BatchNorm2d(32, affine=False)    ## If two shape features are being added.
        #self.bn_attn = nn.BatchNorm2d(64, affine=False)    ## If two shape features are being concatenated.
        
        self.convlstm1 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)    ## element-wise add.
        #self.convlstm1 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)    ## concat.
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        #self.fc4 = nn.Linear(8064, 128)    ## res004: 8064
        self.fc4 = nn.Linear(16128, 128)
        
        self.conv2_q = nn.Conv2d(in_channels=128, out_channels=128//factor_c_attn, kernel_size=1, dilation=1, padding=0, stride=1)
        self.conv2_k = nn.Conv2d(in_channels=128, out_channels=128//factor_c_attn, kernel_size=1, dilation=1, padding=0, stride=1)
        self.conv2_v = nn.Conv2d(in_channels=128, out_channels=128//factor_c_attn, kernel_size=1, dilation=1, padding=0, stride=1)
        self.conv2_global = nn.Conv2d(in_channels=128//factor_c_attn, out_channels=128, kernel_size=1, dilation=1, padding=0, stride=1)
        self.bn_attn_2 = nn.BatchNorm2d(128, affine=False)
        
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1)
        #self.bn6 = nn.BatchNorm1d(129, affine=False)
        self.bn6 = nn.BatchNorm1d(128, affine=False)

    def forward(self, input_side, input_back, input_speed):
        B, L, H, W = input_side.size()

        ## Resolution Adaptive Encoding Module
        imgs_side_original = input_side
        imgs_side_original = imgs_side_original.view(B*L, 1, H, W)
        imgs_back_original = input_back
        imgs_back_original = imgs_back_original.view(B*L, 1, H, W)
        
        imgs_side_low_res = imgs_side_original.view(B*L, 1, H, W)
        imgs_side_low_res = self.up1(self.pool1(imgs_side_low_res))
        imgs_back_low_res = imgs_back_original.view(B*L, 1, H, W)
        imgs_back_low_res = self.up1(self.pool1(imgs_back_low_res))
            
        output_side_list = []    
        output_back_list = [] 
        for dep_map in [imgs_side_original, imgs_side_low_res]:
            dep_map = F.relu(self.conv1_2(dep_map))
            dep_map = F.relu(self.conv1_3(dep_map))
            dep_map = self.pool1_3(dep_map)
            output_side_list.append(dep_map)
            
        for dep_map in [imgs_back_original, imgs_back_low_res]:
            dep_map = F.relu(self.conv1_2(dep_map))
            dep_map = F.relu(self.conv1_3(dep_map))
            dep_map = self.pool1_3(dep_map)
            output_back_list.append(dep_map)
            
        imgs_side_original = output_side_list[0]
        imgs_side_low_res = output_side_list[1]
        imgs_back_original = output_back_list[0]
        imgs_back_low_res = output_back_list[1]
        
        ## Channel Attention in CNNs based on SE-Net [J.Hu+, 2018]
        ## dep_map_side
        dep_map_side = [imgs_side_original, imgs_side_low_res]
        _, c_attn, h_attn, w_attn = dep_map_side[1].size()
        imgs_low_res = dep_map_side[1].view(B, L, c_attn, h_attn, w_attn)
        imgs_original = dep_map_side[0].view(B, L, c_attn, h_attn, w_attn)
            
        imgs_low_res_attn_map = torch.permute(imgs_low_res, (0, 2, 1, 3, 4))
        imgs_low_res_attn_map = torch.reshape(imgs_low_res_attn_map, (B, c_attn, L*h_attn, w_attn))
        imgs_low_res_attn_map = F.avg_pool2d(imgs_low_res_attn_map, imgs_low_res_attn_map.size()[2:])
        imgs_low_res_attn_map = F.relu(self.conv2_enc(imgs_low_res_attn_map)) 
        imgs_low_res_attn_map = self.conv2_dec(imgs_low_res_attn_map)
        imgs_low_res_attn_map = imgs_low_res_attn_map.view(B, 1, c_attn, 1, 1)
            
        imgs_low_res_attn_map = torch.sigmoid(imgs_low_res_attn_map)
        imgs_original_attn_map = 1 - imgs_low_res_attn_map
            
        imgs_low_res = torch.mul(imgs_low_res, imgs_low_res_attn_map)
        imgs_original = torch.mul(imgs_original, imgs_original_attn_map)
            
        ssf = imgs_original + imgs_low_res
        ssf = ssf.view(B*L, c_attn, h_attn, w_attn)
        ssf_side = self.bn_attn(ssf)
        
        ## dep_map_back
        dep_map_back = [imgs_back_original, imgs_back_low_res]
        _, c_attn, h_attn, w_attn = dep_map_back[1].size()
        imgs_low_res = dep_map_back[1].view(B, L, c_attn, h_attn, w_attn)
        imgs_original = dep_map_back[0].view(B, L, c_attn, h_attn, w_attn)
            
        imgs_low_res_attn_map = torch.permute(imgs_low_res, (0, 2, 1, 3, 4))
        imgs_low_res_attn_map = torch.reshape(imgs_low_res_attn_map, (B, c_attn, L*h_attn, w_attn))
        imgs_low_res_attn_map = F.avg_pool2d(imgs_low_res_attn_map, imgs_low_res_attn_map.size()[2:])
        imgs_low_res_attn_map = F.relu(self.conv2_enc(imgs_low_res_attn_map)) 
        imgs_low_res_attn_map = self.conv2_dec(imgs_low_res_attn_map)
        imgs_low_res_attn_map = imgs_low_res_attn_map.view(B, 1, c_attn, 1, 1)
            
        imgs_low_res_attn_map = torch.sigmoid(imgs_low_res_attn_map)
        imgs_original_attn_map = 1 - imgs_low_res_attn_map
            
        imgs_low_res = torch.mul(imgs_low_res, imgs_low_res_attn_map)
        imgs_original = torch.mul(imgs_original, imgs_original_attn_map)
            
        ssf = imgs_original + imgs_low_res
        ssf = ssf.view(B*L, c_attn, h_attn, w_attn)
        ssf_back = self.bn_attn(ssf)

        output_list_2 = []
        for dep_maps in [ssf_side, ssf_back]:
            f_st = dep_maps.view(B, L, dep_maps.size(-3), dep_maps.size(-2), dep_maps.size(-1))
            f_st = self.convlstm1(f_st)
            f_st = (f_st[0][0])[:, -1, :, :, :]
            f_st = self.bn4(self.pool4(f_st))
            #f_st = f_st.view(B, f_st.size(-3), f_st.size(-2)*f_st.size(-1))
            output_list_2.append(f_st)
        
        f_st_total = torch.cat((output_list_2[0], output_list_2[1]), dim=1)
        
        ## reference: Non-local Neural Networks [X. Wang+, CVPR'18]
        f_st_total_q = self.conv2_q(f_st_total)
        b_attn, c_attn, h_attn, w_attn = f_st_total_q.size(0), f_st_total_q.size(1), f_st_total_q.size(2), f_st_total_q.size(3)
        f_st_total_q = f_st_total_q.permute(0, 2, 3, 1)
        f_st_total_q = f_st_total_q.view(b_attn, h_attn*w_attn, c_attn)
        
        f_st_total_k = self.conv2_k(f_st_total)
        f_st_total_k = f_st_total_k.view(b_attn, c_attn, h_attn*w_attn)
        
        attn_map = torch.bmm(f_st_total_q, f_st_total_k)
        attn_map = F.softmax(attn_map, dim=1)
        
        f_st_total_v = self.conv2_v(f_st_total)
        f_st_total_v = f_st_total_v.permute(0, 2, 3, 1)
        f_st_total_v = f_st_total_v.view(b_attn, h_attn*w_attn, c_attn)
        f_st_attn_map = torch.bmm(attn_map, f_st_total_v)
        f_st_attn_map = f_st_attn_map.view(b_attn, h_attn, w_attn, c_attn)
        f_st_attn_map = f_st_attn_map.permute(0, 3, 1, 2)
        
        f_st_attn_map = self.conv2_global(f_st_attn_map)
        f_st_total = f_st_total + f_st_attn_map
        f_st_total = self.bn_attn_2(f_st_total)
        
        f_st_total = torch.flatten(f_st_total, 1)
        f_st_total = self.fc4(f_st_total)

        ## positional feature concatenating
        #input_speed = input_speed.view(B*(L-2), 1, 1)
        #input_speed = self.conv5(input_speed)
        #input_speed = input_speed.view(B, (L-2))
        #f_speed = torch.mean(input_speed, 1)
        #f_speed = torch.unsqueeze(f_speed, 1)

        #f_gait = torch.cat((f_spatiotemporal, f_speed), dim=1)
        f_gait = f_st_total
        f_gait = self.bn6(F.relu(f_gait))

        return f_gait
    

class TwoVGaitNonPen(nn.Module):
    def __init__(self, num_peds=20):
        super(TwoVGaitNonPen, self).__init__()
        self.identifier = TwoVGait()
        #self.fc_final = nn.Linear(129, num_peds)
        self.fc_final = nn.Linear(128, num_peds)

    def forward(self, x_side, x_back, x_speed, embed=False):
        f_gait = self.identifier(x_side, x_back, x_speed)
        #print(x.size())

        if embed:
            return f_gait
        pred_x = self.fc_final(f_gait)
        #print(pred_x.size())

        return pred_x