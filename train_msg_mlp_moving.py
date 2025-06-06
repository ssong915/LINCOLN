from collections import defaultdict
import dgl
import torch 
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import AveragePrecision
from sklearn import metrics
import os
from tqdm import tqdm
import pandas as pd
import gc
import time
import random


import utils, preprocess, data_load
import models.decoder as decoder
from models.common import *
from preprocess import *

# import wandb

def model_train(args, data_dict, data, g, train_pos_loader, train_neg_loader,  
                encoder, decoder, optimizer, scheduler, 
                prev_node_feature, all_v_feat, snapshot_node_index):  
    device = args.device
    aggregator = args.aggregator
    
    encoder.train()
    decoder.train()
    optimizer.zero_grad()      
          
    while True:  
        for l in range(args.num_layers):
            # 0. Initialize
            time_interval = args.time_interval   
            temp_edge_G = data_dict['temp_edge_G'].to(device)
            struct_edge_G = data_dict['struct_edge_G'].to(device)
            
            # 1-1. Update hypergraph node features (HNE)      
            v_reg_weight = data_dict['v_reg_weight'].to(device)
            v_reg_sum = data_dict['v_reg_sum'].to(device)
            e_reg_weight = data_dict['e_reg_weight'].to(device)
            e_reg_sum = data_dict['e_reg_sum'].to(device)
            
            if l == 0:
                v_feat = args.v_feat[g.nodes('node').cpu()].to(device)
                e_feat = data.e_feat[g.nodes('edge').cpu()].to(device)
            else:
                v_feat = update_v_feat.squeeze(0)
                e_feat = update_e_feat
            
            update_v_feat, update_e_feat, ssl_loss = encoder(args, g, g, v_feat, e_feat, time_interval, temp_edge_G, struct_edge_G, device
                                                             ,v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum)
            update_node_feat = update_v_feat
     
            v_feat = update_node_feat.detach().cpu()
            all_v_feat[snapshot_node_index] = v_feat    
            
            # 1-2. Temporal Feature Encoder (Moving avg)        
            concat_v_feat = all_v_feat * args.avg_ratio + prev_node_feature[l] * (1- args.avg_ratio)
            all_v_feat, prev_node_feature[l] = concat_v_feat, concat_v_feat
            
            all_v_feat = all_v_feat.detach().cpu()            
            prev_node_feature[l] = prev_node_feature[l].detach().cpu()

            del v_feat, e_feat, temp_edge_G, struct_edge_G
            gc.collect()
        
        del update_e_feat, update_v_feat

        pos_hedges, pos_labels, is_last = train_pos_loader.next()
        neg_hedges, neg_labels, _ = train_neg_loader.next()
            
        # 2. Hyperedge prediction  
        pos_preds = decoder(all_v_feat.to(device), pos_hedges, aggregator)
        pos_preds = pos_preds.squeeze(1)
        
        neg_preds = decoder(all_v_feat.to(device), neg_hedges, aggregator)
        neg_preds = neg_preds.squeeze(1)

        # 3. Compute training loss and update parameters
        criterion = nn.BCELoss()
        real_loss = criterion(pos_preds, pos_labels)
        fake_loss = criterion(neg_preds, neg_labels)
        
        if args.ns_mode == 'none':
            train_loss = real_loss
        else:
            train_loss = (real_loss + fake_loss) / 2
        
        if args.use_contrastive == 'true':
            train_loss = train_loss + (ssl_loss*args.contrast_ratio)
        else:
            train_loss = train_loss
               
        train_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(train_loss)

        if is_last:
            break
        
    return all_v_feat, prev_node_feature

@torch.no_grad()
def model_eval(args, data_dict, data, g, dataloader, 
                encoder, decoder, 
                t_1_prev_node_feature, t_1_v_feat, snapshot_node_index):
    
    device = args.device
    Aggregator = args.aggregator
    num_layers = args.num_layers
    n_feat = t_1_v_featt
    
    encoder.eval()
    decoder.eval()
    
    preds_list = []
    labels_list = []
    
    while True: 
        for l in range(num_layers):
            # 0. Initialize
            time_interval = args.time_interval   
            temp_edge_G = data_dict['temp_edge_G'].to(device)
            struct_edge_G = data_dict['struct_edge_G'].to(device)
            
            # 1-1. Update hypergraph node features (message passing)      
            v_reg_weight = data_dict['v_reg_weight'].to(device)
            v_reg_sum = data_dict['v_reg_sum'].to(device)
            e_reg_weight = data_dict['e_reg_weight'].to(device)
            e_reg_sum = data_dict['e_reg_sum'].to(device)
            
            if l == 0:
                v_feat = args.v_feat[g.nodes('node').cpu()].to(device)
                e_feat = data.e_feat[g.nodes('edge').cpu()].to(device)
            else:
                v_feat = update_v_feat
                e_feat = update_e_feat          

            update_v_feat, update_e_feat, _ = encoder(args, g.to(device), g.to(device), v_feat, e_feat, time_interval, temp_edge_G, struct_edge_G, device
                                                      ,v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum) 
            n_feat[snapshot_node_index] = update_v_feat.detach().cpu()
                
             # 1-2. Aggregate hypergraph node features (Moving avg)          
            concat_v_feat = n_feat * args.avg_ratio + t_1_prev_node_feature[l] * (1- args.avg_ratio)       
            n_feat, t_1_prev_node_feature[l] = concat_v_feat, concat_v_feat

            del v_feat, e_feat, temp_edge_G, struct_edge_G
            gc.collect()        
        
        del update_v_feat, update_e_feat

        hedges, labels, is_last = dataloader.next() 
                 
        # 2. Hyperedge prediction  
        preds = decoder(n_feat.to(device), hedges, Aggregator)
                
        preds_list += preds.tolist() 
        labels_list += labels.tolist() 
            
        if is_last:
            break
        
    return preds_list, labels_list, n_feat, t_1_prev_node_feature
    
    
def live_update(args, encoder, decoder, optimizer, scheduler, snapshot_data, snapshot_time, f_log, j):
    total_time = len(snapshot_data) 
        
    device = args.device
    patience = args.early_stop
    num_layers = args.num_layers
    
    prev_node_feature = [args.v_feat] * num_layers
    
    sns_avg_roc, sns_avg_ap = [], []
    mns_avg_roc, mns_avg_ap = [], []
    cns_avg_roc, cns_avg_ap = [], []
    mixed_avg_roc, mixed_avg_ap  = [], []
    average_avg_roc, average_avg_ap = [], []
    
    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = preprocess.get_all_neg_samples(args)
    
    for t in tqdm(range(total_time-1)): # for each snapshot
        best_roc = 0
        best_ap = 0
        best_vfeat = args.v_feat 
        best_prev_node_feature = prev_node_feature 
        
        if t != 0:
            # load t-1 best model parameter
            encoder.load_state_dict(torch.load(f"{args.folder_name}/encoder_{j}.pkt"))
            decoder.load_state_dict(torch.load(f"{args.folder_name}/decoder_{j}.pkt"))
        
        # load t & t+1 snapshot data 
        # calculate snapshot's time interval 
        start_time = torch.FloatTensor([min(snapshot_time[t])])
        end_time = torch.FloatTensor([max(snapshot_time[t])])
        args.time_interval = torch.cat((start_time, end_time), dim=0)

        # load t snapshot (for training)  
        train_snapshot_edges, train_snapshot_times = snapshot_data[t], snapshot_time[t]   
        data_dict, _ = data_load.gen_data(args, train_snapshot_edges, train_snapshot_times)
        _, snapshot_node_index = utils.reindex_snapshot(train_snapshot_edges)
               
        data = Hypergraph(args, args.dataset_name, snapshot_data[t])

        if data.weight_flag:
            g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE)
        else:
            g = gen_DGLGraph(args, data.hedge2node)
     
        g = g.to(device)
        data.v_feat = data.v_feat.to(device)
        data.e_feat = data.e_feat.to(device)

        # load t+1 snapshot (for link prediction)
        all_pos_hedge = snapshot_data[t+1]
        next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
        train_edges, valid_edges, test_edges = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
        
        train_pos_loader, train_neg_loader = data_load.get_dataloaders(train_edges, args.batch_size, device , args.ns_mode, args.neg_ratio, label='Train')
        valid_pos_loader, valid_neg_sns_loader, valid_neg_mns_loader, valid_neg_cns_loader = data_load.get_dataloaders(valid_edges, args.batch_size, device, None, args.neg_ratio, label='Valid')
        test_pos_loader, test_neg_sns_loader, test_neg_mns_loader, test_neg_cns_loader = data_load.get_dataloaders(test_edges, args.batch_size, device, None,  args.neg_ratio, label='Test')
         
        t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
        t_1_v_feat =  args.v_feat.clone().detach()
                       
        for epoch in (range(args.epochs)):   
            prev_node_feature = t_1_prev_node_feature
            all_v_feat = t_1_v_feat   
            
            # [Train] 
            all_v_feat, prev_node_feature = model_train(args, data_dict, data, g, train_pos_loader, train_neg_loader,  
                                                        encoder, decoder, optimizer, scheduler, 
                                                        t_1_prev_node_feature, t_1_v_feat, snapshot_node_index) 
            
            # [Validation]                 
            val_pred_pos, val_label_pos, _ , _ = model_eval(args, data_dict, data, g, valid_pos_loader, 
                                               encoder, decoder, 
                                               t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)         
            val_pred_sns, val_label_sns, _ , _ = model_eval(args, data_dict, data, g, valid_neg_sns_loader, 
                                               encoder, decoder, 
                                               t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)            
            val_pred_cns, val_label_cns, _ , _ = model_eval(args, data_dict, data, g, valid_neg_cns_loader, 
                                               encoder, decoder, 
                                               t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)           
            val_pred_mns, val_label_mns, _ , _= model_eval(args, data_dict, data, g, valid_neg_mns_loader, 
                                               encoder, decoder, 
                                               t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)
            

            # SNS validation set
            roc_sns, ap_sns = utils.measure(val_label_pos+val_label_sns, val_pred_pos+val_pred_sns) 
            
            # MNS validation set
            roc_mns, ap_mns = utils.measure(val_label_pos+val_label_mns, val_pred_pos+val_pred_mns) 

            # CNS validation set
            roc_cns, ap_cns = utils.measure(val_label_pos+val_label_cns, val_pred_pos+val_pred_cns) 

            # Mixed validation set
            d = len(val_pred_pos) // 3
            if d < 1 :
                d = 1
            val_label_mixed = val_label_pos + val_label_sns[0:d]+val_label_mns[0:d]+val_label_cns[0:d]
            val_pred_mixed = val_pred_pos + val_pred_sns[0:d]+val_pred_mns[0:d]+val_pred_cns[0:d]
            roc_mixed, ap_mixed = utils.measure(val_label_pos+val_label_mixed, val_pred_pos+val_pred_mixed) 
            
            roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
            ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4
           
            # early stop    
            if best_roc < roc_average or epoch==0:
                best_roc = roc_average         
                best_ap = ap_average
                best_vfeat = all_v_feat
                best_prev_node_feature = prev_node_feature
                torch.save(encoder.state_dict(), f"{args.folder_name}/encoder_{j}.pkt")
                torch.save(decoder.state_dict(), f"{args.folder_name}/decoder_{j}.pkt")
                no_improvement_count = 0          
                if epoch == args.epochs:
                    prev_node_feature = best_prev_node_feature
                    break
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience: 
                    prev_node_feature = best_prev_node_feature
                    del best_vfeat, best_prev_node_feature
                    break 
        
        encoder.load_state_dict(torch.load(f"{args.folder_name}/encoder_{j}.pkt"))
        decoder.load_state_dict(torch.load(f"{args.folder_name}/decoder_{j}.pkt"))
                    
        # [Test]                  
        test_pred_pos, test_label_pos, _, _ = model_eval(args, data_dict, data, g, test_pos_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)         
        test_pred_sns, test_label_sns, _, _= model_eval(args, data_dict, data, g, test_neg_sns_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat,  snapshot_node_index)            
        test_pred_cns, test_label_cns, _, _ = model_eval(args,data_dict, data, g, test_neg_cns_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat,  snapshot_node_index)           
        test_pred_mns, test_label_mns, _, _ = model_eval(args, data_dict, data, g, test_neg_mns_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)
    
        # SNS validation set
        roc_sns, ap_sns = utils.measure(test_label_pos+test_label_sns, test_pred_pos+test_pred_sns) 
        
        # MNS validation set
        roc_mns, ap_mns = utils.measure(test_label_pos+test_label_mns, test_pred_pos+test_pred_mns) 

        # CNS validation set
        roc_cns, ap_cns = utils.measure(test_label_pos+test_label_cns, test_pred_pos+test_pred_cns) 

        # Mixed validation set
        d = len(test_pred_pos) // 3
        if d < 1 :
            d = 1
        test_label_mixed = test_label_pos + test_label_sns[0:d]+test_label_mns[0:d]+test_label_cns[0:d]
        test_pred_mixed = test_pred_pos + test_pred_sns[0:d]+test_pred_mns[0:d]+test_pred_cns[0:d]
        roc_mixed, ap_mixed = utils.measure(test_label_pos+test_label_mixed, test_pred_pos+test_pred_mixed)       
        mixed_avg_roc.append(roc_mixed)
        mixed_avg_ap.append(ap_mixed)  
        
        # Avg score
        roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
        ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4        
        average_avg_roc.append(roc_average)
        average_avg_ap.append(ap_average)
        
    # final_sns_roc = sum(sns_avg_roc)/len(sns_avg_roc)
    # final_mns_roc = sum(mns_avg_roc)/len(mns_avg_roc)
    # final_cns_roc = sum(cns_avg_roc)/len(cns_avg_roc)
    # final_mixed_roc = sum(mixed_avg_roc)/len(mixed_avg_roc)
    final_average_roc = sum(average_avg_roc)/len(average_avg_roc)

    # final_sns_ap = sum(sns_avg_ap)/len(sns_avg_ap)
    # final_mns_ap = sum(mns_avg_ap)/len(mns_avg_ap)
    # final_cns_ap = sum(cns_avg_ap)/len(cns_avg_ap)
    # final_mixed_ap = sum(mixed_avg_ap)/len(mixed_avg_ap)
    final_average_ap = sum(average_avg_ap)/len(average_avg_ap)


    return final_average_roc, final_average_ap 


def fixed_split(args, encoder, decoder, optimizer, scheduler, full_data, full_time, f_log, j):
    device = args.device
    patience = args.early_stop
    num_layers = args.num_layers
        
    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = preprocess.get_all_neg_samples(args)
    
    train_hyperedge, train_time, valid_hyperedge, valid_time, test_hyperedge, test_time = utils.split_edges(full_data, full_time)
    snapshot_data, snapshot_time = preprocess.split_in_snapshot(args, train_hyperedge, train_time)
    val_snapshot_data, val_snapshot_time = preprocess.split_in_snapshot(args, valid_hyperedge, valid_time)
    test_snapshot_data, test_snapshot_time = preprocess.split_in_snapshot(args, test_hyperedge, test_time)
    best_roc = 0
    best_ap = 0
    
    init_prev_node_feature = [args.v_feat] * num_layers
    init_all_v_feat = args.v_feat
    
    total_roc = []
    total_ap = []
       
    for epoch in tqdm(range(args.epochs)):
        prev_node_feature = init_prev_node_feature
        all_v_feat = init_all_v_feat
        # [Train]
        for t in (range(len(snapshot_data)-1)):
            t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
            t_1_v_feat =  all_v_feat.clone().detach()
            
            # load t & t+1 snapshot 
            # calculate snapshot's time interval
            start_time = torch.FloatTensor([min(snapshot_time[t])])
            end_time = torch.FloatTensor([max(snapshot_time[t])])
            args.time_interval = torch.cat((start_time, end_time), dim=0)
            
            train_snapshot_edges, train_snapshot_times = snapshot_data[t], snapshot_time[t]   
            data_dict, _ = data_load.gen_data(args, train_snapshot_edges, train_snapshot_times)
            reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(train_snapshot_edges)
        
            data = Hypergraph(args, args.dataset_name, snapshot_data[t])
            before_snapshot_data = data.get_data(0) 

            ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
            full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)

            if data.weight_flag:
                g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE)
            else:
                g = gen_DGLGraph(args, data.hedge2node)
            try:
                sampler = dgl.dataloading.NeighborSampler(ls)
                fullsampler = dgl.dataloading.NeighborSampler(full_ls)
            except:
                sampler = dgl.dataloading.MultiLayerNeighborSampler(ls, False)
                fullsampler = dgl.dataloading.MultiLayerNeighborSampler(full_ls)

            g = g.to(device)
            before_snapshot_data = before_snapshot_data.to(device)
            data.v_feat = data.v_feat.to(device)
            data.e_feat = data.e_feat.to(device)
            
            # load t+1 snapshot (for link prediction)
            all_pos_hedge = snapshot_data[t+1]
            next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
            train_edges, _, _ = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
            train_pos_loader, train_neg_loader = data_load.get_dataloaders(train_edges, args.batch_size, device , args.ns_mode, args.neg_ratio, label='Train')
                      
            all_v_feat, prev_node_feature = model_train(args,data_dict, data, g, train_pos_loader, train_neg_loader,  
                                                        encoder, decoder, optimizer, scheduler, 
                                                        t_1_prev_node_feature, t_1_v_feat, snapshot_node_index) 
         
        # [Validation]  
        total_roc = []
        total_ap = []
        for t in (range(len(val_snapshot_data)-1)):
            t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
            t_1_v_feat =  all_v_feat.clone().detach()
            
            # calculate snapshot's time interval
            start_time = torch.FloatTensor([min(val_snapshot_time[t])])
            end_time = torch.FloatTensor([max(val_snapshot_time[t])])
            args.time_interval = torch.cat((start_time, end_time), dim=0)
            
            # load t snapshot (for training)         
            valid_snapahot_edges, valid_snapahot_times = val_snapshot_data[t], val_snapshot_time[t]   
            val_data_dict = data_load.gen_data(args, valid_snapahot_edges, valid_snapahot_times)
            reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(valid_snapahot_edges)

            valid_data = Hypergraph(args, args.dataset_name, val_snapshot_data[t])
            before_val_snapshot_data = valid_data.get_data(0) 

            ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
            full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)

            if data.weight_flag:
                val_g = gen_weighted_DGLGraph(args, valid_data.hedge2node, valid_data.hedge2nodePE)
            else:
                val_g = gen_DGLGraph(args, valid_data.hedge2node)

            val_g = val_g.to(device)
            before_val_snapshot_data = before_val_snapshot_data.to(device)
            valid_data.v_feat = valid_data.v_feat.to(device)
            valid_data.e_feat = valid_data.e_feat.to(device)
            
            # load t+1 snapshot (for link prediction)
            all_pos_hedge = val_snapshot_data[t+1]
            next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
            _, valid_edges, _ = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
            valid_pos_loader, valid_neg_sns_loader, valid_neg_mns_loader, valid_neg_cns_loader = data_load.get_dataloaders(valid_edges, args.batch_size, device, None, label='Valid')
                 
            # [Validation]                 
            val_pred_pos, val_label_pos, all_v_feat, prev_node_feature = model_eval(args, val_data_dict, valid_data, val_g, valid_pos_loader, 
                                                encoder, decoder, 
                                                t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)         
            val_pred_sns, val_label_sns, _, _ = model_eval(args, val_data_dict, valid_data, val_g, valid_neg_sns_loader, 
                                                encoder, decoder, 
                                                t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)            
            val_pred_cns, val_label_cns, _, _ = model_eval(args, val_data_dict, valid_data, val_g, valid_neg_cns_loader, 
                                                encoder, decoder, 
                                                t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)           
            val_pred_mns, val_label_mns, _, _ = model_eval(args,val_data_dict, valid_data, val_g, valid_neg_mns_loader, 
                                                encoder, decoder, 
                                                t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)
            # SNS validation set
            roc_sns, ap_sns = utils.measure(val_label_pos+val_label_sns, val_pred_pos+val_pred_sns) 
            
            # MNS validation set
            roc_mns, ap_mns = utils.measure(val_label_pos+val_label_mns, val_pred_pos+val_pred_mns) 

            # CNS validation set
            roc_cns, ap_cns = utils.measure(val_label_pos+val_label_cns, val_pred_pos+val_pred_cns) 

            # Mixed validation set
            d = len(val_pred_pos) // 3
            if d < 1 :
                d = 1
            val_label_mixed = val_label_pos + val_label_sns[0:d]+val_label_mns[0:d]+val_label_cns[0:d]
            val_pred_mixed = val_pred_pos + val_pred_sns[0:d]+val_pred_mns[0:d]+val_pred_cns[0:d]
            roc_mixed, ap_mixed = utils.measure(val_label_pos+val_label_mixed, val_pred_pos+val_pred_mixed) 
            
            roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
            ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4 
            total_roc.append(roc_average)           
            total_ap.append(ap_average)           
          
        # early stop    
        total_roc_average = sum(total_roc) / len(total_roc)
        total_ap_average = sum(total_ap) / len(total_ap)
        if best_roc < roc_average or epoch==0:
            best_roc = roc_average         
            best_ap = ap_average
            best_vfeat = all_v_feat
            best_prev_node_feature = prev_node_feature
            torch.save(encoder.state_dict(), f"{args.folder_name}/encoder_{j}.pkt")
            torch.save(decoder.state_dict(), f"{args.folder_name}/decoder_{j}.pkt")
            no_improvement_count = 0          
            if epoch == args.epochs:
                prev_node_feature = best_prev_node_feature
                break
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience: 
                prev_node_feature = best_prev_node_feature
                del best_vfeat, best_prev_node_feature
                break 
                
    encoder.load_state_dict(torch.load(f"{args.folder_name}/encoder_{j}.pkt"))
    decoder.load_state_dict(torch.load(f"{args.folder_name}/decoder_{j}.pkt"))
                
    # [Test]   
    total_roc = []
    total_ap = []   
    t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
    t_1_v_feat =  args.v_feat.clone().detach()          
    for t in (range(len(test_snapshot_data)-1)): 
        t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
        t_1_v_feat = all_v_feat.clone().detach()
        
        # calculate snapshot's time interval
        start_time = torch.FloatTensor([min(test_snapshot_time[t])])
        end_time = torch.FloatTensor([max(test_snapshot_time[t])])
        args.time_interval = torch.cat((start_time, end_time), dim=0)
        
        # load t snapshot        
        test_snapahot_edges, test_snapahot_times = test_snapshot_data[t], test_snapshot_time[t]   
        test_data_dict = data_load.gen_data(args, test_snapahot_edges, test_snapahot_times)
        reindexed_snapshot_edges, snapshot_node_index = utils.reindex_snapshot(test_snapahot_edges)

        test_data = Hypergraph(args, args.dataset_name, test_snapahot_edges)
        before_test_snapshot_data = test_data.get_data(0) 

        ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
        full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)

        if data.weight_flag:
            test_g = gen_weighted_DGLGraph(args, test_data.hedge2node, test_data.hedge2nodePE)
        else:
            test_g = gen_DGLGraph(args, test_data.hedge2node)

        test_g = test_g.to(device)
        before_test_snapshot_data = before_test_snapshot_data.to(device)
        test_data.v_feat = test_data.v_feat.to(device)
        test_data.e_feat = test_data.e_feat.to(device)
        
        # load t+1 snapshot (for link prediction)
        all_pos_hedge = test_snapshot_data[t+1]
        next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
        _, _, test_edges = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
        test_pos_loader, test_neg_sns_loader, test_neg_mns_loader, test_neg_cns_loader = data_load.get_dataloaders(test_edges, args.batch_size, device, None,  args.neg_ratio, label='Test')
            
        test_pred_pos, test_label_pos, all_v_feat, prev_node_feature = model_eval(args, test_data_dict, test_data, test_g, test_pos_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)         
        test_pred_sns, test_label_sns, _ , _ = model_eval(args, test_data_dict, test_data, test_g, test_neg_sns_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat,  snapshot_node_index)            
        test_pred_cns, test_label_cns, _ , _ = model_eval(args, test_data_dict, test_data, test_g, test_neg_cns_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)           
        test_pred_mns, test_label_mns, _ , _ = model_eval(args, test_data_dict, test_data, test_g, test_neg_mns_loader, 
                                            encoder, decoder, 
                                            t_1_prev_node_feature, t_1_v_feat, snapshot_node_index)
    
        # SNS validation set
        roc_sns, ap_sns = utils.measure(test_label_pos+test_label_sns, test_pred_pos+test_pred_sns) 
        
        # MNS validation set
        roc_mns, ap_mns = utils.measure(test_label_pos+test_label_mns, test_pred_pos+test_pred_mns) 

        # CNS validation set
        roc_cns, ap_cns = utils.measure(test_label_pos+test_label_cns, test_pred_pos+test_pred_cns) 

        # Mixed validation set
        d = len(test_pred_pos) // 3
        if d < 1 :
            d = 1
        test_label_mixed = test_label_pos + test_label_sns[0:d]+test_label_mns[0:d]+test_label_cns[0:d]
        test_pred_mixed = test_pred_pos + test_pred_sns[0:d]+test_pred_mns[0:d]+test_pred_cns[0:d]
        roc_mixed, ap_mixed = utils.measure(test_label_pos+test_label_mixed, test_pred_pos+test_pred_mixed)    
        
        # Avg score
        roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
        ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4        
        total_roc.append(roc_average)
        total_roc.append(ap_average) 
               
    total_roc_average = sum(total_roc) / len(total_roc)
    total_ap_average = sum(total_ap) / len(total_ap)
    
    return total_roc_average, total_ap_average 
        