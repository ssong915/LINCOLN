import math
import numpy as np
import torch
import random
from sampler import *
import dgl

def parse_string_to_list(string):
    numbers = np.fromstring(string[1:-1], dtype=int, sep=', ')
    return numbers.tolist()

def get_datainfo(r_dataset):
    r_dataset['node_set'] = r_dataset['node_set'].apply(parse_string_to_list)
    # r_dataset = r_dataset.sort_values(by="time")
    r_dataset.reset_index(inplace=True) 
        
    return r_dataset


def reindex_node_index(all_data):

    org_node_index = []

    for idx, row in all_data.iterrows():
        node_set = row['node_set']
        # node_set = [int(node.strip()) for node in node_set_str.strip('[]').split(',')]
        new_node_set = []
        for n_idx in node_set:
            if n_idx not in org_node_index:
                org_node_index.append(n_idx)
            new_idx = org_node_index.index(n_idx)
            new_node_set.append(new_idx)
        all_data.at[idx, 'node_set'] = new_node_set
    
    return all_data

def split_in_snapshot(args,all_hyperedges,timestamps):
    snapshot_data = list()
    snapshot_time = list()
    
    freq_sec = args.snapshot_size 
    split_criterion = timestamps // freq_sec
    groups = np.unique(split_criterion)
    groups = np.sort(groups)
    merge_edge_data = []
    merge_time_data = []
    for t in groups:
        period_members = (split_criterion == t) 
        edge_data = list(all_hyperedges[period_members]) 
        time_data  = list(timestamps[period_members])
        
        unique_set = set()
        unique_list = []

        for item in edge_data:
            tuple_item = tuple(item)
            if tuple_item not in unique_set:
                unique_set.add(tuple_item)
                unique_list.append(item)
        
        if len(edge_data) < 4 or len(unique_list) < 2: 
            merge_edge_data = merge_edge_data + edge_data
            merge_time_data = merge_time_data + time_data
            
            if len(merge_edge_data) >= 4:
                snapshot_data.append(merge_edge_data)
                merge_time_data = list(map(int,merge_time_data))
                snapshot_time.append(merge_time_data)
                merge_time_data = []
                merge_edge_data = []
        else :
            if len(merge_edge_data) != 0 :
                edge_data = merge_edge_data + edge_data
                time_data = merge_time_data + time_data
            snapshot_data.append(edge_data)
            time_data = list(map(int,time_data))
            snapshot_time.append(time_data)
            merge_edge_data = []
            merge_time_data = []
    lengths = []
    for hyperedge in snapshot_data:
        lengths.append(len(hyperedge))
        
    average_length = sum(lengths) / len(lengths)

    return snapshot_data, snapshot_time

    
def get_full_data(args, dataset):

    time = dataset['time']
    ts_start = time.min()
    ts_end = time.max() 
    filter_data = dataset[(dataset['time'] >= ts_start) & (dataset['time']<=ts_end)]

    max_node_idx = max(max(row) for row in list(filter_data['node_set']))
    num_node = max_node_idx + 1
    
    all_hyperedges = filter_data['node_set']
    timestamps = filter_data['time']
    
    return all_hyperedges, timestamps, num_node   
    
    
def neg_generator(mode, HE, pred_num):
    
    if mode == 'mns' :
        mns = MNSSampler(pred_num)
        t_mns = mns(set(tuple(x) for x in HE))
        t_mns = list(t_mns)
        neg_hedges = [list(edge) for edge in t_mns]        
        
    elif mode == 'sns':
        sns = SNSSampler(pred_num)
        t_sns = sns(set(tuple(x) for x in HE))
        t_sns = list(t_sns)
        neg_hedges = [list(edge) for edge in t_sns]    
        
    elif mode == 'cns'or mode == 'none':
        cns = CNSSampler(pred_num)
        t_cns = cns(set(tuple(x) for x in HE))    
        t_cns = list(t_cns)    
        neg_hedges = [list(edge) for edge in t_cns]
        
    
    return neg_hedges

def get_all_neg_samples(args):
    DATA = args.dataset_name 
    feat_folder = f'./data/{DATA}/'
    sns_neg_hedge = torch.load(feat_folder + f'sns_{args.dataset_name}.pt')
    cns_neg_hedge = torch.load(feat_folder + f'cns_{args.dataset_name}.pt')
    mns_neg_hedge = torch.load(feat_folder + f'mns_{args.dataset_name}.pt')
        
    return sns_neg_hedge, cns_neg_hedge, mns_neg_hedge
    
def get_next_samples(next_snapshot, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, ratio):    
    pos_size = len(next_snapshot)
    neg_size = int(math.ceil(pos_size * ratio))
    if neg_size < 3:
        neg_size = 3
    if neg_size > len(all_sns_neg_hedge):
        sns_neg_hedge = random.choices(all_sns_neg_hedge, k=neg_size)
        cns_neg_hedge = random.choices(all_cns_neg_hedge, k=neg_size)
        mns_neg_hedge = random.choices(all_mns_neg_hedge, k=neg_size)
    else:  
        sns_neg_hedge = random.sample(all_sns_neg_hedge, neg_size)
        cns_neg_hedge = random.sample(all_cns_neg_hedge, neg_size)
        mns_neg_hedge = random.sample(all_mns_neg_hedge, neg_size)
    
    return next_snapshot, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge

def split_data(data):
    total_size = len(data)
    test_size = int(math.ceil(total_size * 0.1))
    valid_size = int(math.ceil(total_size * 0.2))
    
    test_data = data[:test_size]
    valid_data = data[test_size:test_size + valid_size]
    train_data = data[test_size + valid_size:]
    
    return test_data, valid_data, train_data
    
def get_data_dict(next_pos_edges, sns_hedge, cns_hedge, mns_hedge, eval_mode):
    if eval_mode == 0 or eval_mode =='fixed_split':
        train_edges = {'train_pos':next_pos_edges, 'train_sns':sns_hedge, 
                        'train_cns':cns_hedge, 'train_mns':mns_hedge}
        valid_edges = {'valid_pos':next_pos_edges, 'valid_sns':sns_hedge, 
                    'valid_cns':cns_hedge, 'valid_mns':mns_hedge}
        test_edges = {'test_pos':next_pos_edges, 'test_sns':sns_hedge, 
                    'test_cns':cns_hedge, 'test_mns':mns_hedge}
        return train_edges, valid_edges, test_edges
            
    test_pos_edges, valid_pos_edges, train_pos_edges = split_data(next_pos_edges)
    test_sns_hedge, valid_sns_hedge, train_sns_hedge = split_data(sns_hedge)
    test_cns_hedge, valid_cns_hedge, train_cns_hedge = split_data(cns_hedge)
    test_mns_hedge, valid_mns_hedge, train_mns_hedge = split_data(mns_hedge)

    train_edges = {'train_pos':train_pos_edges, 'train_sns':train_sns_hedge, 
                   'train_cns':train_cns_hedge, 'train_mns':train_mns_hedge}
    valid_edges = {'valid_pos':valid_pos_edges, 'valid_sns':valid_sns_hedge, 
                   'valid_cns':valid_cns_hedge, 'valid_mns':valid_mns_hedge}
    test_edges = {'test_pos':test_pos_edges, 'test_sns':test_sns_hedge, 
                   'test_cns':test_cns_hedge, 'test_mns':test_mns_hedge}

    return train_edges, valid_edges, test_edges

class Hypergraph:
    def __init__(self, args, dataset_name, snapshot):
        self.dataname = dataset_name
        
        self.hedge2node = []
        self.node2hedge = [] 
        self.hedge2nodepos = [] # hyperedge index -> node positions (after binning)
        self._hedge2nodepos = [] # hyperedge index -> node positions (before binning)
        self.node2hedgePE = []
        self.hedge2nodePE = []
        self.weight_flag = True
        self.hedge2nodeweight = []
        self.node2hedgeweight = []
        self.numhedges = 0
        self.numnodes = 0
        
        self.order_dim = 0 
   
        
        self.hedgeindex = {} # papaercode -> index
        self.hedgename = {} # index -> papercode
        self.e_feat = []

        self.node_reindexing = {} # nodeindex -> reindex
        self.node_orgindex = {} # reindex -> nodeindex
        self.v_feat = [] # (V, 1)
        
        self.load_graph(args, snapshot) 

        
    def load_graph(self, args, snapshot):
        
        self.max_len = 0

        for hedge in snapshot:
            hidx = self.numhedges
            self.numhedges += 1
            self.hedgeindex[hidx] = hidx
            self.hedgename[hidx] = hidx
            self.hedge2node.append([])
            self.hedge2nodepos.append([])
            self._hedge2nodepos.append([])
            self.hedge2nodePE.append([])
            self.hedge2nodeweight.append([])
            self.e_feat.append([])

            if (self.max_len < len(hedge)):
                self.max_len = len(hedge)

            for node in hedge:
                if node not in self.node_reindexing:
                    node_reindex = self.numnodes
                    self.numnodes += 1 
                    self.node_reindexing[node] = node_reindex
                    self.node_orgindex[node_reindex] = node 
                    self.node2hedge.append([])
                    self.node2hedgePE.append([])
                    self.node2hedgeweight.append([])
                    self.v_feat.append([])
                nodeindex = self.node_reindexing[node]
                self.hedge2node[hidx].append(nodeindex)
                self.node2hedge[nodeindex].append(hidx)
                self.hedge2nodePE[hidx].append([])
                self.node2hedgePE[nodeindex].append([])
                    
        for vhedges in self.node2hedge:
            if self.max_len < len(vhedges):
                self.max_len = len(vhedges)
        self.v_feat = torch.rand(len(self.v_feat), args.dim_edge)
        self.e_feat = torch.rand(len(self.e_feat), args.dim_edge)
        self.test_index = []
        self.valid_index = []
        self.validsize = 0
        self.testsize = 0
        self.trainsize = 0
        self.hedge2type = torch.zeros(self.numhedges)
        
        self.trainsize = self.numhedges
        
        
        
    def get_data(self, type=0):
        hedgelist = ((self.hedge2type == type).nonzero(as_tuple=True)[0])
        return hedgelist
    
# Generate DGL Graph ==============================================================================================
def gen_DGLGraph(args, hedge2node):
    data_dict = defaultdict(list)
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            
    g = dgl.heterograph(data_dict)

    return g

def gen_weighted_DGLGraph(args, hedge2node, hedge2nodePE):
    edgefeat_dim = 0
    for efeat_list in hedge2nodePE:
        efeat_dim = len(efeat_list[0])
        edgefeat_dim = max(edgefeat_dim, efeat_dim)
    
    data_dict = defaultdict(list)
    in_edge_weights = []
    con_edge_weights = []
    
    for hidx, hedge in enumerate(hedge2node):
        for vorder, v in enumerate(hedge):
            # connection
            data_dict[('node', 'in', 'edge')].append((v, hidx))
            data_dict[('edge', 'con', 'node')].append((hidx, v))
            # edge feat
            efeat = hedge2nodePE[hidx][vorder]
            efeat += np.zeros(edgefeat_dim - len(efeat)).tolist()
            in_edge_weights.append(efeat)
            con_edge_weights.append(efeat)

    in_edge_weights = torch.Tensor(in_edge_weights)
    con_edge_weights = torch.Tensor(con_edge_weights)

    g = dgl.heterograph(data_dict)
    g['in'].edata['weight'] = in_edge_weights
    g['con'].edata['weight'] = con_edge_weights
    
    return g

def gen_HG_data(args, snapshot_data):   
    device = args.device
    all_v_feat = args.v_feat
    data = Hypergraph(args, args.dataset_name, snapshot_data)
    
    data.v_feat = all_v_feat[list(data.node_orgindex.values())].to(device)
    data.e_feat = data.e_feat.to(device)

    if data.weight_flag:
        g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE)
    else:
        g = gen_DGLGraph(args, data.hedge2node)
    g = g.to(device)
    
    return data, g