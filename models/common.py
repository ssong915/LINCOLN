import torch
import torch.nn as nn
from utils import *
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class FeatureAggregator(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def max_min(self, embeddings):
        max_val, _ = torch.max(embeddings, dim=0)
        min_val, _ = torch.min(embeddings, dim=0)
        return max_val - min_val

    def average(self, embeddings):
        embedding = embeddings.mean(dim=0).squeeze()
        return embedding
    
    def forward(self, embeddings, mask):
        hedge_embeddings = []
        for i in range(embeddings.size(0)):  
            valid_features = embeddings[i][mask[i] == 1] 
            hedge_embedding = self.max_min(valid_features)
            hedge_embeddings.append(hedge_embedding)
        feat_e = torch.stack(hedge_embeddings)

        return feat_e

# ============================== Global Time encoding =================================#
class TimeIntervalEncoding(nn.Module):
    def __init__(self, dim_time, layers):
        super(TimeIntervalEncoding, self).__init__()
        
        # 1) start time/ end time + frequency function    
        self.w = torch.nn.Linear(1, dim_time)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time)))
                                       .float().reshape(dim_time, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim_time).float())
        
        # 2) start time/ end time --> snapshot time embedding
        Layers = []
        for i in range(layers-1):
            Layers.append(nn.Linear(dim_time, dim_time))
            if i != (layers-2):
                Layers.append(nn.ReLU(True))
                
        self.time_l1 = torch.nn.Linear(dim_time*2, dim_time)
        self.embedder = nn.Sequential(*Layers)
        self.time_norm = torch.nn.LayerNorm(dim_time)
        
        

    def forward(self, time_interval, device): 
        '''
            input: snapshot start time, snapshot end time
            output: snapshot time embedding
        '''
        # 1) start time/ end time + frequency function    
        start_time = torch.tensor([time_interval[0].item()]).to(device)
        end_time = torch.tensor([time_interval[-1].item()]).to(device)
        start_feat = self.w(torch.cos(start_time))
        end_feat = self.w(torch.cos(end_time))
        
        # 2) start time/ end time --> snapshot time embedding
        snapshot_time = torch.cat([start_feat, end_feat], dim=-1)
        snapshot_feat = self.time_l1(snapshot_time)
        snapshot_feat = self.embedder(snapshot_feat)
        snapshot_feat = self.time_norm(snapshot_feat)

        return snapshot_feat   