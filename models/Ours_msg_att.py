import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import math

from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from dgl import DGLGraph
import gc
from utils import *
from models.common import *
from models.attention import *

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim) # bias = False is also ok.
        
    def forward(self, F):
        output = self.linear(F)
        return output

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, p=0.5):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
        self.gcn_layer2 = GCNLayer(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, X, A):
        X = self.dropout(X)
        F = torch.mm(A, X)
        F = self.gcn_layer1(F)
        F = self.layer_norm(F)
        F = self.dropout(F)
        F = torch.mm(A, F)
        output = self.gcn_layer2(F)
        return output
        
class Ours_att(nn.Module):
    def __init__(self, time_layer, time_concat, edge_concat, time_concat_mode, edge_concat_mode,
                 dim_vertex, dim_hidden, dim_edge, dim_time, node_edge_aggregator):
        super(Ours_att, self).__init__()
        self.vtx_lin_1layer = torch.nn.Linear(dim_vertex, dim_vertex)
                
        # 1. time-aware node representation
        if time_concat == 'true':
            self.time_encoder = TimeIntervalEncoding(dim_time, time_layer)            
            if time_concat_mode == 'att':
                self.time_att = MAB(dim_vertex, dim_time, dim_time)  
            elif time_concat_mode == 'mlp':     
                self.time_mlp = MLP(dim_vertex + dim_time , dim_hidden , dim_vertex)
            elif time_concat_mode == 'concat':
                pass  
                
        # 2. node to edge
        if node_edge_aggregator == 'in_hedges':
            # self.node_self_att = SAB(dim_vertex, dim_vertex)
            self.node_self_att = ISAB(dim_vertex, dim_vertex)
            self.ve_att = MAB(dim_edge, dim_vertex, dim_vertex)                    
        
        self.edge_layer_norm = nn.LayerNorm(dim_edge)  
            
        # 3. time-aware hyperedge representation
        if edge_concat == 'true':
            self.temp_GCN = GCN(dim_edge, dim_edge)
            self.struct_GCN = GCN(dim_edge, dim_edge)       
                 
            if edge_concat_mode == 'att':  
                self.edge_att = MAB(dim_edge, dim_edge, dim_edge) 
            elif edge_concat_mode == 'mlp':
                self.edge_mlp = MLP(dim_edge * 2, dim_hidden ,dim_edge)   
            elif edge_concat_mode == 'concat': 
                pass
            
        # 4. edge to node
        if node_edge_aggregator == 'in_hedges':
            # self.edge_self_att = SAB(dim_edge, dim_edge)
            self.edge_self_att = ISAB(dim_edge, dim_edge)
            self.ev_att = MAB(dim_vertex, dim_edge, dim_edge)                    
      
                                 
        self.node_layer_norm = nn.LayerNorm(dim_vertex)         

        # for Contrastive loss
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
            
    def weight_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']
        return {'weight': weight}
    
    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'weight': edges.data['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        aggr = torch.sum(Weight * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}
            
    def get_ssl_loss(self, struct_emb, temp_emb):
        def cosine_similarity(feat1: Tensor, feat2: Tensor):
            cosine_similarity = self.cosine(feat1, feat2)
            return torch.mean(cosine_similarity)
        
        def shuffled(x1):
            perm = torch.randperm(x1.size(0))
            shuffled_emb = x1[perm, :]
            return shuffled_emb

        pos = cosine_similarity(struct_emb, temp_emb) 
        neg1 = cosine_similarity(struct_emb, shuffled(struct_emb)) 
        neg2 = cosine_similarity(temp_emb, shuffled(temp_emb))
        ssl_loss = -torch.sum(-torch.log(torch.sigmoid(pos))-torch.log(1-torch.sigmoid(neg1))-torch.log(1-torch.sigmoid(neg2)))
        
        return ssl_loss  
    
    def forward(self, g, args, time_interval, vfeat, efeat, v_reg_weight, v_reg_sum, 
                e_reg_weight, e_reg_sum, temp_edge_G, struct_edge_G, device, edge_set, node_set):
        
        with g.local_scope():
            feat_v = self.vtx_lin_1layer(vfeat)
            feat_e = efeat
            
            # 1. time-aware node representation
            if args.time_concat == 'true':
                time_interval = time_interval.to(device)
                global_time = self.time_encoder(time_interval).unsqueeze(0)
                
                if args.time_concat_mode == 'att':
                    # 1) Q: node feature, K,V: time
                    feat_v = self.time_att(feat_v, global_time)
                    del time_interval, global_time
                                    
                elif args.time_concat_mode == 'mlp':  
                    all_global_time = global_time.expand(feat_v.shape[0], -1)
                    concat_nfeat = torch.cat([feat_v, all_global_time], dim=-1)
                    feat_v = self.time_mlp(concat_nfeat) 
                    del time_interval, all_global_time, concat_nfeat, global_time
                                        
                elif args.time_concat_mode == 'concat':
                    all_global_time = global_time.expand(feat_v.shape[0], -1)
                    feat_v = torch.cat([feat_v, all_global_time], dim=-1)
                    del time_interval, all_global_time               
                        
            # 2. node to edge
            if args.node_edge_aggregator == 'in_hedges':                                      
                for i in range(feat_e.shape[0]):
                    # 2) Q,K,V: node feature
                    att_node = self.node_self_att(feat_v[edge_set[i]])
                    # 3) Q: hyperedge feature, key,value: node feature
                    feat_e[i] = self.ve_att(feat_e[i].unsqueeze(0), att_node)
                    del att_node
                
            feat_e = self.edge_layer_norm(feat_e)
                
            # 3. time-aware hyperedge representation
            ssl_loss = 0
            if args.edge_concat == 'true':    
                struct_efeat = self.struct_GCN(feat_e, struct_edge_G).detach().cpu()       
                temp_efeat = self.temp_GCN(feat_e, temp_edge_G).detach().cpu()   
                ssl_loss = self.get_ssl_loss(struct_efeat, temp_efeat)          
                
                if args.edge_concat_mode == 'att':
                    # feat_e = self.edge_att(struct_efeat, temp_efeat)
                    feat_e = self.edge_att(temp_efeat, struct_efeat)
                    
                elif args.edge_concat_mode == 'mlp':  
                    concat_efeat = torch.cat([struct_efeat, temp_efeat], dim=-1).to(device)
                    feat_e = self.edge_mlp(concat_efeat)
                    del struct_efeat, temp_efeat, concat_efeat
                    
                elif args.edge_concat_mode == 'concat':
                    feat_e = torch.cat([struct_efeat, temp_efeat], dim=-1)
                             
            # 4. edge to node
            if args.node_edge_aggregator == 'in_hedges':                                         
                for i in range(feat_v.shape[0]):
                    # 2) Q,K,V: hyperdge feature
                    att_edge = self.edge_self_att(feat_e[node_set[i]])
                    # 3) Q: node feature, key,value: hyperege feature
                    feat_v[i] = self.ev_att(feat_v[i].unsqueeze(0), att_edge) 
                    del att_edge    

                
            feat_v = self.node_layer_norm(feat_v)            
            
            return feat_v, ssl_loss