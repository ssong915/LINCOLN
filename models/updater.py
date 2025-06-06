import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import math

from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GRULayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRULayer, self).__init__()
        
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())

    def forward(self, aggr_node_feature, prev_node_feature, device):        
        H_prev = prev_node_feature.to(device) 
        X = aggr_node_feature.to(device) 
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde 

        H_out = H_gru
        return H_out, H_out

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, past_node_feature, curr_node_feature, device):    
        curr_node_feature = curr_node_feature.to(device) 
        past_node_feature = past_node_feature.to(device) 
                
        x = curr_node_feature.unsqueeze(0).permute(1, 0, 2).contiguous()
        h0 = past_node_feature.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = past_node_feature.unsqueeze(0).repeat(self.num_layers, 1, 1)

        out, _ = self.lstm(x, (h0, c0))
        
        return out[:, -1, :], out[:, -1, :]
