import wandb
import numpy as np
import torch
import random
import argparse
import warnings
import logging

import math, sys
import numpy as np

from sklearn import metrics
from torchmetrics import AveragePrecision
from torchmetrics.retrieval import RetrievalMRR

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Hypergraph representation learning')

    ##### training hyperparameter #####
    parser.add_argument("--dataset_name", type=str, default='email-Enron', help='dataset name: _')
    parser.add_argument('--seed', type=int, default=808, metavar='S', help='Random seed (default: 1111)')
    parser.add_argument("--folder_name", default='exp1', help='experiment number')
    parser.add_argument("--gpu", type=int, default=0, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--exp_num", default=1, type=int, help='number of experiments')
    parser.add_argument("--epochs", default=50, type=int, help='number of epochs')
    parser.add_argument("--early_stop", default=10, type=int, help='number of early stop')
    parser.add_argument("--training", type=str, default='wgan', help='loss objective: wgan, none')
    parser.add_argument("--lr", default=0.0005, type=float, help='learning rate')
    parser.add_argument("--eval_mode", type=str, default='live_update', help='live_update or fixed_split evaluation')
     
    # neg and contrastive loss   
    parser.add_argument("--ns_mode", type=str, default='mns', help='train시 neg method: mns, sns, cns')
    parser.add_argument("--neg_ratio", default=2, type=float, help='pos:neg = 1:ratio')
    
    parser.add_argument('--use_contrastive', type=str, default='true', help='Use Contrastive Loss: true (use) or false (no use)')
    parser.add_argument('--contrast_ratio', type=float, default=0.3, help='Contrastive loss control factor') 
    
    # snapshot split method
    parser.add_argument("--snapshot_size", default=2628000, type=int, help='snapshot size')
    parser.add_argument("--batch_size", default=16, type=int, help='batch size')
    
    # encoder
    parser.add_argument("--model", default='Ours_msg_mlp', type=str, help='encoder: hgnn')
    parser.add_argument("--num_layers", default=2, type=int, help='number of layers')
    parser.add_argument("--alpha_e", default=0, type=float, help='normalization term for hnhn')
    parser.add_argument("--alpha_v", default=0, type=float, help='normalization term for hnhn')
    
    parser.add_argument("--node_edge_aggregator", default='node_hedges', type=str, help='in_hedges/ node_hedges')
    
    parser.add_argument("--num_heads", default=2, type=int, help='multi-attention heads')
    parser.add_argument("--num_inds", default=2, type=int, help='self-attention inducing point')
    
    # updater
    parser.add_argument("--updater", default='gru', type=str, help='updater')
    parser.add_argument('--avg_ratio', type=float, default=0.7, help='Moving average ratio control factor') 
    
    # time-aware node
    parser.add_argument("--time_concat", default='true', type=str, help='time encoder feature used')
    parser.add_argument("--time_concat_mode", default='att', type=str, help='time, feature concat method')
    parser.add_argument("--time_layers", default=3, type=int, help='number of time layers')
    
    # time-aware hyperedge    
    parser.add_argument("--edge_concat", default='true', type=str, help='edge graph feature used')
    parser.add_argument("--edge_concat_mode", default='mlp', type=str, help='edge graph feature concat')
    parser.add_argument("--edge_concat_ratio", default=0.5, type=float, help='edge concat ratio between struct & temporal ')
    parser.add_argument("--alpha_edge", default=0.5, type=float, help='edge graph feature concat')
    
    # decoder
    parser.add_argument("--aggregator", type=str, default='Avg', help='aggregator: maxmin, average')
    
    # feature dimmension
    parser.add_argument("--dim_hidden", default=32, type=int, help='dimension of hidden vector')
    parser.add_argument("--dim_vertex", default=32, type=int, help='dimension of vertex hidden vector')
    parser.add_argument("--dim_edge", default=32, type=int, help='dimension of edge hidden vector')
    parser.add_argument("--dim_time", default=32, type=int, help='dimension of time hidden vector')

    args = parser.parse_args()
    
    return args


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
@torch.no_grad()    
def measure(label, pred):
    average_precision = AveragePrecision(task='binary')
    auc_roc = metrics.roc_auc_score(np.array(label), np.array(pred))

    label = torch.tensor(label)
    label = label.type(torch.int64)
    ap = average_precision(torch.tensor(pred).squeeze(1), torch.tensor(label))
                    
    return round(auc_roc,3), round(ap.item(),3)

def reindex_snapshot(snapshot_edges):
    org_node_index = []
    reindex_snapshot_edges = [[0 for _ in row] for row in snapshot_edges]
    for i, edge in enumerate(snapshot_edges):
        for j, node in enumerate(edge):
            if node not in org_node_index:
                org_node_index.append(node)
            new_idx = org_node_index.index(node)
            reindex_snapshot_edges[i][j] = new_idx
    
    return reindex_snapshot_edges, org_node_index

def split_edges(dataset, time_info):

    time_hyperedge = (dataset)
    snapshot_time = (time_info)
    total_size = len(time_hyperedge)
    idcs = np.arange(len(time_hyperedge)).tolist()
    
    test_size = int(math.ceil(total_size * 0.1))
    valid_size = int(math.ceil(total_size * 0.2))
        
    test_hyperedge = time_hyperedge[-test_size:]
    test_time = snapshot_time[-test_size:]
    
    valid_hyperedge = time_hyperedge[-(test_size + valid_size):-test_size]
    valid_time = snapshot_time[-(test_size + valid_size):-test_size]
    
    train_hyperedge = time_hyperedge[:-(test_size + valid_size)]
    train_time = snapshot_time[:-(test_size + valid_size)]        

    return train_hyperedge, train_time, valid_hyperedge, valid_time, test_hyperedge, test_time

