import random
import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import dgl
import preprocess, utils
import math


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H) # ( nv * ne )
    DV = np.sum(H, axis=1) + 1e-5
    DE = np.sum(H, axis=0) + 1e-5

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -1)))
    H = np.mat(H)
    HT = H.T

    G = DV2 * H * invDE * HT * DV2
    
    return G
    
        
def gen_init_data(args, num_node):
    args.device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    
    args.nv = num_node 
    args.input_dim = args.dim_vertex 
    
    args.v_feat = torch.rand(args.nv, args.input_dim)  
    
    return args

def gen_data(args, snapshot_data, snapshot_time):
    data_dict = {}  

    nv = args.nv
    ne = len(snapshot_data) 
    args.ne = ne
    data_dict['e_feat'] = torch.rand(ne, args.dim_edge)
    _,snapshot_node_index = utils.reindex_snapshot(snapshot_data)
        
    # calculate snapshot's time interval
    start_time = torch.FloatTensor([min(snapshot_time)])
    end_time = torch.FloatTensor([max(snapshot_time)])
    args.time_interval = torch.cat((start_time, end_time), dim=0)
    
    # 1. structural proximity incidence matrix
    incidence = torch.zeros(nv, ne)
    for edge_idx, node_set in enumerate(snapshot_data):
        for node_idx in node_set:
            incidence[node_idx, edge_idx] += 1
    data_dict['h_incidence'] = incidence.T # ne, nv
    struct_edge_G = generate_G_from_H(data_dict['h_incidence']) # ne, ne
    
    # 2. temporal proximity incidence matrix
    temp_edge_G = torch.zeros(ne, ne)
    non_zero_indices = np.nonzero(struct_edge_G)
    
    rows, cols = non_zero_indices        
    for row, col in zip(rows, cols):
        edge_time_1 = snapshot_time[row]
        edge_time_2 = snapshot_time[col]
        difference_time = abs(edge_time_1 - edge_time_2)
        if difference_time != 0:
            temp_edge_G[row,col] = 1/ difference_time        
    
    data_dict['temp_edge_G'] = temp_edge_G  # ne, ne            
    data_dict['struct_edge_G'] = torch.tensor(struct_edge_G)  # ne, ne
            
    # HNHN terms
    data_dict['v_weight'] = torch.zeros(nv, 1)
    data_dict['e_weight'] = torch.zeros(ne, 1)
    node2sum = defaultdict(list)
    edge2sum = defaultdict(list)
    e_reg_weight = torch.zeros(ne)
    v_reg_weight = torch.zeros(nv)

    for edge_idx, node_set in enumerate(snapshot_data):
        for node_idx in node_set:
            e_wt = data_dict['e_weight'][edge_idx]
            e_reg_wt = e_wt ** args.alpha_e
            e_reg_weight[edge_idx] = e_reg_wt
            node2sum[node_idx].append(e_reg_wt)

            v_wt = data_dict['v_weight'][node_idx]
            v_reg_wt = v_wt ** args.alpha_v
            v_reg_weight[node_idx] = v_reg_wt
            edge2sum[edge_idx].append(v_reg_wt)

    v_reg_sum = torch.zeros(nv)
    e_reg_sum = torch.zeros(ne)

    for node_idx, wt_l in node2sum.items():
        v_reg_sum[node_idx] = sum(wt_l)
    for edge_idx, wt_l in edge2sum.items():
        e_reg_sum[edge_idx] = sum(wt_l)

    e_reg_sum[e_reg_sum == 0] = 1
    v_reg_sum[v_reg_sum == 0] = 1
    data_dict['v_reg_weight'] = torch.Tensor(v_reg_weight).unsqueeze(-1)
    data_dict['v_reg_sum'] = torch.Tensor(v_reg_sum).unsqueeze(-1)
    data_dict['e_reg_weight'] = torch.Tensor(e_reg_weight).unsqueeze(-1)
    data_dict['e_reg_sum'] = torch.Tensor(e_reg_sum).unsqueeze(-1)
    
    return data_dict, snapshot_node_index

def load_snapshot(args, DATA):
     
    # 1. get feature and edge index        
    r_data = pd.read_csv('./data/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = preprocess.get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = preprocess.get_full_data(args, data_info)
    snapshot_data, snapshot_time = preprocess.split_in_snapshot(args,all_hyperedges,timestamps)
       
    return snapshot_data, snapshot_time, num_node

def load_fulldata(args, DATA):
     
    # 1. get feature and edge index        
    r_data = pd.read_csv('./data/{}/new_hyper_{}.csv'.format(DATA, DATA))
    
    # 2. make snapshot  
    data_info = preprocess.get_datainfo(r_data)
    all_hyperedges, timestamps, num_node = preprocess.get_full_data(args, data_info)
    
    return all_hyperedges, timestamps, num_node


def get_dataloaders(data_dict, batch_size, device, ns_mode, ns_ratio, label='Train'):

    if label == 'Train':
        train_pos_data = data_dict["train_pos"]
        
        train_pos_labels = [1 for i in range(len(train_pos_data))] # training positive hyperedges
        train_pos_dataloader = BatchDataloader(train_pos_data, train_pos_labels, batch_size, device, is_Train=True)

        train_neg_data = None
        if ns_mode == 'sns':
            train_neg_data = data_dict["train_sns"]
        elif ns_mode == 'mns':
            train_neg_data = data_dict["train_mns"]
        elif ns_mode == 'cns':
            train_neg_data = data_dict["train_cns"]
        elif ns_mode == 'mix':
            d = len(data_dict["train_sns"]) // 3
            if d < 1:
                d = 1
            train_neg_data = data_dict["train_sns"][0:d] + data_dict["train_mns"][0:d] + data_dict["train_cns"][0:d]
            random.shuffle(train_neg_data)
            
        train_neg_labels = [0 for i in range(len(train_neg_data))] # training negative hyperedges
        neg_batch_size = int(math.ceil(batch_size*ns_ratio))
        train_neg_dataloader = BatchDataloader(train_neg_data, train_neg_labels, neg_batch_size, device, is_Train=True)

        return train_pos_dataloader, train_neg_dataloader

    elif label == 'Valid':
        val_pos_data = data_dict["valid_pos"]
        val_pos_labels = [1 for i in range(len(val_pos_data))] # validation positive hyperedges
        val_pos_dataloader = BatchDataloader(val_pos_data, val_pos_labels, batch_size, device, is_Train=False)

        val_neg_sns_data = data_dict["valid_sns"]
        val_neg_mns_data = data_dict["valid_mns"]
        val_neg_cns_data = data_dict["valid_cns"]

        val_neg_labels = [0 for i in range(len(val_neg_sns_data))] # validation negative hyperedges
        val_neg_sns_dataloader = BatchDataloader(val_neg_sns_data, val_neg_labels, batch_size, device, is_Train=False)
        val_neg_mns_dataloader = BatchDataloader(val_neg_mns_data, val_neg_labels, batch_size, device, is_Train=False)
        val_neg_cns_dataloader = BatchDataloader(val_neg_cns_data, val_neg_labels, batch_size, device, is_Train=False)

        return val_pos_dataloader, val_neg_sns_dataloader, val_neg_mns_dataloader, val_neg_cns_dataloader

    elif label == 'Test':
        test_pos_data = data_dict["test_pos"]
        test_pos_labels = [1 for i in range(len(test_pos_data))] # validation positive hyperedges
        test_pos_dataloader = BatchDataloader(test_pos_data, test_pos_labels, batch_size, device, is_Train=False)

        test_neg_sns_data = data_dict["test_sns"]
        test_neg_mns_data = data_dict["test_mns"]
        test_neg_cns_data = data_dict["test_cns"]

        test_neg_labels = [0 for i in range(len(test_neg_sns_data))] # validation negative hyperedges
        test_neg_sns_dataloader = BatchDataloader(test_neg_sns_data, test_neg_labels, batch_size, device, is_Train=False)
        test_neg_mns_dataloader = BatchDataloader(test_neg_mns_data, test_neg_labels, batch_size, device, is_Train=False)
        test_neg_cns_dataloader = BatchDataloader(test_neg_cns_data, test_neg_labels, batch_size, device, is_Train=False)

        return test_pos_dataloader, test_neg_sns_dataloader, test_neg_mns_dataloader, test_neg_cns_dataloader


class BatchDataloader(object):
    def __init__(self, hyperedges, labels, batch_size, device, is_Train=False):
        """Creates an instance of Hyperedge Batch Dataloader.
        Args:
            hyperedges: List(frozenset). List of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size. int. Batch size of each batch.
        """
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self._cursor = 0
        self.device = device
        self.is_Train = is_Train
        self.time_end = False

        if is_Train:
            self.shuffle()

    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]

    def __iter__(self):
        self._cursor = 0
        return self

    def next(self):
        return self._next_batch()

    def _next_batch(self):
        ncursor = self._cursor+self.batch_size # next cursor position

        next_hyperedges = None
        next_labels = None

        if ncursor >= len(self.hyperedges): # end of each epoch
            next_hyperedges = self.hyperedges[self._cursor:]
            next_labels = self.labels[self._cursor:]
            self._cursor = 0
            self.time_end = True

            if self.is_Train:
                self.shuffle() # data shuffling at every epoch

        else:
            next_hyperedges = self.hyperedges[self._cursor:self._cursor + self.batch_size]
            next_labels = self.labels[self._cursor:self._cursor + self.batch_size]
            self._cursor = ncursor % len(self.hyperedges)

        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in next_hyperedges]
        labels = torch.FloatTensor(next_labels).to(self.device)

        return hyperedges, labels, self.time_end