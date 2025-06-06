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
        
class Ours_msg_mlp(nn.Module):
    def __init__(self, time_layer, time_concat, edge_concat, time_concat_mode, edge_concat_mode,
                 dim_vertex, dim_hidden, dim_edge, dim_time, num_heads, num_inds):
        super(Ours_msg_mlp, self).__init__()
        self.vtx_lin_1layer = torch.nn.Linear(dim_vertex, dim_vertex)
                
        # 1. Periodic-time Injection
        if time_concat == 'true':
            self.time_encoder = TimeIntervalEncoding(dim_time, time_layer)            
            if time_concat_mode == 'att':
                self.time_att = MAB(dim_vertex, dim_time, dim_time, num_heads)  
            elif time_concat_mode == 'mlp':     
                self.time_mlp = MLP(dim_vertex + dim_time , dim_hidden , dim_vertex)
            elif time_concat_mode == 'concat':
                pass  
                
        # 2. Node-to-Hyperedge Aggregation
        if time_concat == 'true' and time_concat_mode =='concat':
            self.ve_lin = torch.nn.Linear(dim_vertex + dim_time, dim_edge)
        else:
            self.ve_lin = torch.nn.Linear(dim_vertex, dim_edge) 
        
        self.edge_layer_norm = nn.LayerNorm(dim_edge)  
        self.activation = F.relu
            
        # 3. Multi-view Disentanglement
        if edge_concat == 'true':
            self.temp_GCN = GCN(dim_edge, dim_edge)
            self.struct_GCN = GCN(dim_edge, dim_edge)       
                 
            if edge_concat_mode == 'att':  
                self.edge_att = MAB(dim_edge, dim_edge, dim_edge, num_heads) 
            elif edge_concat_mode == 'mlp':
                self.edge_mlp = MLP(dim_edge * 2, dim_hidden ,dim_edge)   
            elif edge_concat_mode == 'concat': 
                pass
            
        # 4. Hyperedge-to-Node Aggregation
        self.ev_lin = torch.nn.Linear(dim_edge, dim_vertex)  
                                 
        self.node_layer_norm = nn.LayerNorm(dim_vertex)   
        # for residual (edge) 
        self.efeat_lin = torch.nn.Linear(dim_edge, dim_edge)                 

        # for Contrastive loss
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

        dimension = dim_vertex
        self.dropout_rate = 0.6
        self.dropout = nn.Dropout(self.dropout_rate)
        self.enc_v = nn.ModuleList()
        
        self.enc_v.append(ISAB(dimension, dim_hidden, num_heads, num_inds))
        dimension = dim_hidden
        #     Aggregate part
        self.dec_v = nn.ModuleList()
        self.dec_v.append(MAB(dim_edge, dimension, dim_hidden, num_heads))
        self.dec_v.append(nn.Dropout(self.dropout_rate))
        self.dec_v.append(nn.Linear(dim_hidden, dim_edge))
        
        # For Hyperedge -> Node
        #     Attention part: create node-dependent embedding
        dimension = dim_edge
        self.enc_e = nn.ModuleList()
       
        self.enc_e.append(ISAB(dimension, dim_hidden, num_heads, num_inds))   
        dimension = dim_hidden
        #     Aggregate part
        self.dec_e = nn.ModuleList()
        self.dec_e.append(MAB(dim_vertex, dimension, dim_hidden, num_heads))
        self.dec_e.append(nn.Dropout(self.dropout_rate))
        self.dec_e.append(nn.Linear(dim_hidden, dim_vertex))

        self.weight_flag = False
            
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
    
    def forward(self, args, g1, g2,  vfeat, efeat, time_interval, temp_edge_G, struct_edge_G, 
                device, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum):
        
        feat_v = self.vtx_lin_1layer(vfeat)
        feat_e = efeat
        
        # 1. Periodic Time Injection
        if args.time_concat == 'true':
            time_interval = time_interval.to(device)
            global_time = self.time_encoder(time_interval, device).unsqueeze(0)
            
            if args.time_concat_mode == 'att':
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
            
        # 2. Node-to-Hyperedge Aggregation
        with g1.local_scope():
            given_efeat = efeat
            # edge aggregation
            g1.srcnodes['node'].data['h'] = vfeat
            g1.srcnodes['node'].data['Wh'] = vfeat
            g1.srcnodes['node'].data['reg_weight'] = v_reg_weight[:g1['in'].num_src_nodes()]
            g1.srcnodes['node'].data['reg_sum'] = v_reg_sum[:g1['in'].num_src_nodes()]
            g1.dstnodes['edge'].data['reg_weight'] = e_reg_weight[:g1['in'].num_dst_nodes()]
            g1.dstnodes['edge'].data['reg_sum'] = e_reg_sum[:g1['in'].num_dst_nodes()]
            g1.apply_edges(self.weight_fn, etype='in')
            g1.update_all(self.message_func, self.reduce_func, etype='in')
            norm_vfeat = g1.dstnodes['edge'].data['h']
            if self.activation is not None:
                efeat = self.activation(self.ve_lin(norm_vfeat))
            else:
                efeat = self.ve_lin(norm_vfeat)
            efeat = self.edge_layer_norm(efeat)
            emb = self.efeat_lin(given_efeat)
            efeat = efeat + emb        
            
        # 3. Multi-view Disentanglement
        ssl_loss = 0
        if args.edge_concat == 'true':  
            struct_efeat = self.struct_GCN(efeat, struct_edge_G).detach().cpu()       
            temp_efeat = self.temp_GCN(efeat, temp_edge_G).detach().cpu()   
            ssl_loss = self.get_ssl_loss(struct_efeat, temp_efeat)          
            
            if args.edge_concat_mode == 'att':
                feat_e = self.edge_att(temp_efeat.to(device), struct_efeat.to(device))
                
            elif args.edge_concat_mode == 'mlp':  
                concat_efeat = torch.cat([struct_efeat, temp_efeat], dim=-1).to(device)
                feat_e = self.edge_mlp(concat_efeat)
                del struct_efeat, temp_efeat, concat_efeat
                
            elif args.edge_concat_mode == 'concat':
                feat_e = struct_efeat * args.edge_concat_ratio + temp_efeat * (1- args.edge_concat_ratio)  
                feat_e = feat_e.to(device)
                del temp_efeat, struct_efeat    
              
        # 4. Hyperedge-to-Node Aggregation  
        with g2.local_scope():
            # node aggregattion
            g2.srcnodes['edge'].data['Wh'] = feat_e
            g2.srcnodes['edge'].data['reg_weight'] = e_reg_weight[:g2['con'].num_src_nodes()]
            g2.srcnodes['edge'].data['reg_sum'] = e_reg_sum[:g2['con'].num_src_nodes()]
            g2.dstnodes['node'].data['reg_weight'] = v_reg_weight[:g2['con'].num_dst_nodes()]
            g2.dstnodes['node'].data['reg_sum'] = v_reg_sum[:g2['con'].num_dst_nodes()]
            g2.apply_edges(self.weight_fn, etype='con')
            g2.update_all(self.message_func, self.reduce_func, etype='con')
            norm_efeat = g2.dstnodes['node'].data['h']
            if self.activation is not None:
                vfeat = self.activation(self.ev_lin(norm_efeat))
            else:
                vfeat = self.ev_lin(norm_efeat) 
            feat_v = self.node_layer_norm(vfeat)
            
            
            return feat_v, feat_e, ssl_loss