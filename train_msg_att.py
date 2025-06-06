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
                encoder, updater, decoder, optimizer, scheduler, 
                prev_node_feature, snapshot_node_index):  
    device = args.device
    aggregator = args.aggregator
    all_v_feat = args.v_feat
    
    encoder.train()
    updater.train()
    decoder.train()
    optimizer.zero_grad()      
          
    while True:  
        for l in range(args.num_layers):
            # 0. time interval encoding 반영 
            time_interval = args.time_interval   
            temp_edge_G = data_dict['temp_edge_G'].to(device)
            struct_edge_G = data_dict['struct_edge_G'].to(device)
            
            # 1. Update hypergraph node features (message passing)      
            if l == 0:
                v_feat = all_v_feat[g.nodes('node').cpu()].to(device)
                e_feat = data.e_feat[g.nodes('edge').cpu()].to(device)
            else:
                v_feat = update_v_feat.squeeze(0)
                e_feat = update_e_feat

            update_v_feat, update_e_feat, ssl_loss = encoder(args, g, g, v_feat, e_feat, time_interval, temp_edge_G, struct_edge_G, device)
            all_v_feat[snapshot_node_index] = update_v_feat.detach().cpu() # ~Ht_l   
            
            # 2. Aggregate hypergraph node features (GRU): ~Ht_l + Ht-1_l -> Ht_l      
            if args.updater == 'mlp':
                concat_nfeat = torch.cat([all_v_feat, prev_node_feature[l]], dim=-1).to(device)
                updated_nfeat = updater(concat_nfeat) 
                all_v_feat, prev_node_feature[l] = updated_nfeat, updated_nfeat
            else:  
                all_v_feat, prev_node_feature[l] = updater(all_v_feat, prev_node_feature[l], device)      
            
            all_v_feat = all_v_feat.detach().cpu()
            prev_node_feature[l] = prev_node_feature[l].detach().cpu()
            
            del v_feat, e_feat, temp_edge_G, struct_edge_G
            gc.collect()
        
        del update_e_feat, update_v_feat
        
        pos_hedges, pos_labels, is_last = train_pos_loader.next()
        neg_hedges, neg_labels, _ = train_neg_loader.next()
            
        # 2. Hyperedge prediction  
        # t+1 시점의 positive hyperedge + positive 기반 negative hyperege prediction    
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
        
    all_v_feat = all_v_feat.detach().cpu()        
    prev_node_feature = [i.clone().detach() for i in prev_node_feature]
        
    return all_v_feat, prev_node_feature

@torch.no_grad()
def model_eval(args, data_dict, data, g, dataloader, 
                sns_dataloader, cns_dataloader, mns_dataloader,
                encoder, updater, decoder, 
                t_1_prev_node_feature, snapshot_node_index):
    
    device = args.device
    Aggregator = args.aggregator
    num_layers = args.num_layers
    n_feat = args.v_feat
    
    encoder.eval()
    updater.eval()
    decoder.eval()
    
    preds_list = []
    sns_preds_list = []
    cns_preds_list = []
    mns_preds_list = []
    
    labels_list = []
    ns_labels_list = []
    
    while True: 
        for l in range(num_layers):
            # 0. time interval encoding 반영 
            time_interval = args.time_interval   
            temp_edge_G = data_dict['temp_edge_G'].to(device)
            struct_edge_G = data_dict['struct_edge_G'].to(device)
            
            # 1. Update hypergraph node features (message passing)      
            if l == 0:
                v_feat = n_feat[g.nodes('node').cpu()].to(device)
                e_feat = data.e_feat[g.nodes('edge').cpu()].to(device)
            else:
                v_feat = update_v_feat
                e_feat = update_e_feat          

            update_v_feat, update_e_feat, _ = encoder(args, g.to(device), g.to(device), v_feat, e_feat, time_interval, temp_edge_G, struct_edge_G, device) 
            n_feat[snapshot_node_index] = update_v_feat.detach().cpu()
                
            # 2. Aggregate hypergraph node features (GRU)          
            if args.updater == 'mlp':
                concat_nfeat = torch.cat([n_feat, t_1_prev_node_feature[l]], dim=-1).to(device)
                updated_nfeat = updater(concat_nfeat) 
                n_feat, t_1_prev_node_feature[l] = updated_nfeat, updated_nfeat
            else:           
                n_feat, t_1_prev_node_feature[l] = updater(n_feat, t_1_prev_node_feature[l], device)  # t-1 prev_node_feature

            n_feat = n_feat.detach().cpu()
            t_1_prev_node_feature[l] = t_1_prev_node_feature[l].detach().cpu()
            
            del v_feat, e_feat, temp_edge_G, struct_edge_G
            gc.collect()        
        
        del update_v_feat, update_e_feat

        hedges, labels, is_last = dataloader.next() 
        sns_hedges, sns_labels, is_last = sns_dataloader.next() 
        cns_hedges, cns_labels, is_last = cns_dataloader.next() 
        mns_hedges, mns_labels, is_last = mns_dataloader.next() 
                 
        # 2. Hyperedge prediction  
        preds = decoder(n_feat.to(device), hedges, Aggregator)
        sns_preds = decoder(n_feat.to(device), sns_hedges, Aggregator)
        cns_preds = decoder(n_feat.to(device), cns_hedges, Aggregator)
        mns_preds = decoder(n_feat.to(device), mns_hedges, Aggregator)
                
        preds_list += preds.tolist() 
        labels_list += labels.tolist() 
        
        sns_preds_list += sns_preds.tolist() 
        cns_preds_list += cns_preds.tolist() 
        mns_preds_list += mns_preds.tolist() 
        ns_labels_list += sns_labels.tolist()
        
        if is_last:
            break
        
    n_feat = n_feat.detach().cpu()        
    t_1_prev_node_feature = [i.clone().detach() for i in t_1_prev_node_feature]
    
    del n_feat, t_1_prev_node_feature
        
    return [preds_list, labels_list], [sns_preds_list, cns_preds_list, mns_preds_list, ns_labels_list]
    
    
def live_update(args, encoder, updater, decoder, optimizer, scheduler, snapshot_data, snapshot_time, f_log, j):
    total_time = len(snapshot_data) 
        
    device = args.device
    patience = args.early_stop
    num_layers = args.num_layers
    
    prev_node_feature = [args.v_feat] * num_layers
    best_prev_node_feature = [args.v_feat] * num_layers

    mixed_avg_roc, mixed_avg_ap  = [], []
    average_avg_roc, average_avg_ap = [], []
    
    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = preprocess.get_all_neg_samples(args)
    
    for t in tqdm(range(total_time-1)): 
        best_roc = 0
        best_ap = 0
        if t != 0:
            # load t-1 best model parameter
            prev_node_feature = best_prev_node_feature
            encoder.load_state_dict(torch.load(f"{args.folder_name}/encoder_{j}.pkt"))
            updater.load_state_dict(torch.load(f"{args.folder_name}/updater_{j}.pkt"))
            decoder.load_state_dict(torch.load(f"{args.folder_name}/decoder_{j}.pkt"))
        
        # load t & t+1 snapshot data (t data : st-han input /t+1 data : pos label + neg label)
        # load t snapshot
        data_dict, snapshot_node_index = data_load.gen_data(args, snapshot_data[t], snapshot_time[t])
        data, g = gen_HG_data(args, snapshot_data[t])    

        # load t+1 snapshot (for link prediction)
        all_pos_hedge = snapshot_data[t+1]
        next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
        train_edges, valid_edges, test_edges = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
        
        train_pos_loader, train_neg_loader = data_load.get_dataloaders(train_edges, args.batch_size, device , args.ns_mode, args.neg_ratio, label='Train')
        valid_pos_loader, valid_neg_sns_loader, valid_neg_mns_loader, valid_neg_cns_loader = data_load.get_dataloaders(valid_edges, args.batch_size, device, None, args.neg_ratio, label='Valid')
        test_pos_loader, test_neg_sns_loader, test_neg_mns_loader, test_neg_cns_loader = data_load.get_dataloaders(test_edges, args.batch_size, device, None,  args.neg_ratio, label='Test')
        
        t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
                       
        for epoch in (range(args.epochs)):              
            # [Train] 
            _, prev_node_feature = model_train(args, data_dict, data, g, train_pos_loader, train_neg_loader,  
                                                        encoder, updater, decoder, optimizer, scheduler, 
                                                        t_1_prev_node_feature, snapshot_node_index) 
            
            # [Validation] 
            pos_list, neg_list = model_eval(args, data_dict, data, g, 
                                            valid_pos_loader, valid_neg_sns_loader, valid_neg_cns_loader, valid_neg_mns_loader,
                                            encoder, updater, decoder, 
                                            t_1_prev_node_feature, snapshot_node_index)                      
            val_pred_pos, val_label_pos = pos_list
            val_pred_sns, val_pred_cns, val_pred_mns, val_label_ns = neg_list

            # SNS validation set
            roc_sns, ap_sns = utils.measure(val_label_pos+val_label_ns, val_pred_pos+val_pred_sns) 
            
            # MNS validation set
            roc_mns, ap_mns = utils.measure(val_label_pos+val_label_ns, val_pred_pos+val_pred_mns) 

            # CNS validation set
            roc_cns, ap_cns = utils.measure(val_label_pos+val_label_ns, val_pred_pos+val_pred_cns) 

            # Mixed validation set
            d = len(val_pred_pos) // 3
            if d < 1 :
                d = 1
            val_label_mixed = val_label_pos + val_label_ns[0:d]+val_label_ns[0:d]+val_label_ns[0:d]
            val_pred_mixed = val_pred_pos + val_pred_sns[0:d]+val_pred_mns[0:d]+val_pred_cns[0:d]
            roc_mixed, ap_mixed = utils.measure(val_label_pos+val_label_mixed, val_pred_pos+val_pred_mixed) 
            
            roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
            ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4
           
            # print(f' {epoch}: \t {loss:.4f} \t {roc_sns:.4f} {roc_mns:.4f} {roc_cns:.4f} {roc_mixed:.4f} {roc_average:.4f} \t {ap_sns:.4f} {ap_mns:.4f} {ap_cns:.4f} {ap_mixed:.4f} {ap_average:.4f}')
          
            # early stop    
            if best_roc < roc_average or epoch==0:
                best_roc = roc_average         
                best_ap = ap_average
                best_prev_node_feature = prev_node_feature
                torch.save(encoder.state_dict(), f"{args.folder_name}/encoder_{j}.pkt")
                torch.save(updater.state_dict(), f"{args.folder_name}/updater_{j}.pkt")
                torch.save(decoder.state_dict(), f"{args.folder_name}/decoder_{j}.pkt")
                no_improvement_count = 0          
                if epoch == args.epochs:
                    break
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience: 
                    break 
        
        encoder.load_state_dict(torch.load(f"{args.folder_name}/encoder_{j}.pkt"))
        updater.load_state_dict(torch.load(f"{args.folder_name}/updater_{j}.pkt"))
        decoder.load_state_dict(torch.load(f"{args.folder_name}/decoder_{j}.pkt"))
                    
        # [Test]               
        pos_list, neg_list = model_eval(args, data_dict, data, g, 
                                        test_pos_loader, test_neg_sns_loader, test_neg_cns_loader, test_neg_mns_loader,
                                        encoder, updater, decoder, 
                                        t_1_prev_node_feature, snapshot_node_index)                      
        test_pred_pos, test_label_pos = pos_list
        test_pred_sns, test_pred_cns, test_pred_mns, test_label_ns = neg_list  
    
        # SNS validation set
        roc_sns, ap_sns = utils.measure(test_label_pos+test_label_ns, test_pred_pos+test_pred_sns) 
        
        # MNS validation set
        roc_mns, ap_mns = utils.measure(test_label_pos+test_label_ns, test_pred_pos+test_pred_mns) 

        # CNS validation set
        roc_cns, ap_cns = utils.measure(test_label_pos+test_label_ns, test_pred_pos+test_pred_cns) 

        # Mixed validation set
        d = len(test_pred_pos) // 3
        if d < 1 :
            d = 1
        test_label_mixed = test_label_pos + test_label_ns[0:d]+test_label_ns[0:d]+test_label_ns[0:d]
        test_pred_mixed = test_pred_pos + test_pred_sns[0:d]+test_pred_mns[0:d]+test_pred_cns[0:d]
        roc_mixed, ap_mixed = utils.measure(test_label_pos+test_label_mixed, test_pred_pos+test_pred_mixed)       
        mixed_avg_roc.append(roc_mixed)
        mixed_avg_ap.append(ap_mixed)  
        
        # Avg score
        roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
        ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4        
        average_avg_roc.append(roc_average)
        average_avg_ap.append(ap_average) 
        
        if t % 10 == 0 :
            print(f"\n[{t} time Val AP : {ap_average} / AUROC : {roc_average} ]")       
        
    final_average_roc = sum(average_avg_roc)/len(average_avg_roc)
    final_average_ap = sum(average_avg_ap)/len(average_avg_ap)

    return final_average_roc, final_average_ap 


def fixed_split(args, encoder, updater, decoder, optimizer, scheduler, full_data, full_time, f_log, j):
    device = args.device
    patience = args.early_stop
    num_layers = args.num_layers
        
    all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge = preprocess.get_all_neg_samples(args)
    
    train_hyperedge, train_time, valid_hyperedge, valid_time, test_hyperedge, test_time = utils.split_edges(full_data, full_time)
    snapshot_data, snapshot_time = preprocess.split_in_snapshot(args, train_hyperedge, train_time)
    val_snapshot_data, val_snapshot_time = preprocess.split_in_snapshot(args, valid_hyperedge, valid_time)
    test_snapshot_data, test_snapshot_time = preprocess.split_in_snapshot(args, test_hyperedge, test_time)
    best_roc = 0
    
    init_prev_node_feature = [args.v_feat] * num_layers
    
    for epoch in tqdm(range(args.epochs)):
        
        # [Train]
        prev_node_feature = init_prev_node_feature
        
        for t in (range(len(snapshot_data)-1)):   
            t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
                
            # load t & t+1 snapshot data (t data : st-han input /t+1 data : pos label + neg label)
            # load t snapshot
            data_dict, snapshot_node_index = data_load.gen_data(args, snapshot_data[t], snapshot_time[t])
            data, g = gen_HG_data(args, snapshot_data[t])    
            
            # load t+1 snapshot (for link prediction)
            # next_snapshot_pos_edges, next_snapshot_neg_edges = preprocess.load_hedge_pos_neg(snapshot_data[t+1], args.ns_mode)
            all_pos_hedge = snapshot_data[t+1]
            next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
            train_edges, _, _ = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
            train_pos_loader, train_neg_loader = data_load.get_dataloaders(train_edges, args.batch_size, device , args.ns_mode, args.neg_ratio, label='Train')
                      
            all_v_feat, prev_node_feature = model_train(args, data_dict, data, g, train_pos_loader, train_neg_loader,  
                                                        encoder, updater, decoder, optimizer, scheduler, 
                                                        t_1_prev_node_feature, snapshot_node_index) 
         
        # [Validation]  
        all_val_pred_pos, all_val_label_pos = [], []
        all_val_pred_sns, all_val_label_sns = [], []
        all_val_pred_cns, all_val_label_cns = [], []
        all_val_pred_mns, all_val_label_mns = [], []
        
        prev_node_feature = init_prev_node_feature
        
        for t in (range(len(val_snapshot_data)-1)):
            t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
            
            # load t & t+1 snapshot data (t data : st-han input /t+1 data : pos label + neg label)
            # load t snapshot
            val_data_dict, snapshot_node_index = data_load.gen_data(args, val_snapshot_data[t], val_snapshot_time[t])
            valid_data, val_g = gen_HG_data(args, val_snapshot_data[t])  
            
            # load t+1 snapshot (for link prediction)
            all_pos_hedge = val_snapshot_data[t+1]
            next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
            _, valid_edges, _ = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
            valid_pos_loader, valid_neg_sns_loader, valid_neg_mns_loader, valid_neg_cns_loader = data_load.get_dataloaders(valid_edges, args.batch_size, device, None,  args.neg_ratio, label='Valid')
                 
            # [Validation]     
            pos_list, neg_list = model_eval(args, val_data_dict, valid_data, val_g, 
                                            valid_pos_loader, valid_neg_sns_loader, valid_neg_cns_loader, valid_neg_mns_loader,
                                            encoder, updater, decoder, 
                                            t_1_prev_node_feature, snapshot_node_index)                      
            val_pred_pos, val_label_pos = pos_list
            val_pred_sns, val_pred_cns, val_pred_mns, val_label_ns = neg_list
            
            all_val_pred_pos.extend(val_pred_pos)
            all_val_label_pos.extend(val_label_pos)        
            all_val_pred_sns.extend(val_pred_sns)
            all_val_label_sns.extend(val_label_ns)           
            all_val_pred_cns.extend(val_pred_cns)
            all_val_label_cns.extend(val_label_ns)          
            all_val_pred_mns.extend(val_pred_mns)
            all_val_label_mns.extend(val_label_ns)               
                
        # SNS validation set
        roc_sns, ap_sns = utils.measure(all_val_label_pos+all_val_label_sns, all_val_pred_pos+all_val_pred_sns) 
        
        # MNS validation set
        roc_mns, ap_mns = utils.measure(all_val_label_pos+all_val_label_mns, all_val_pred_pos+all_val_pred_mns) 

        # CNS validation set
        roc_cns, ap_cns = utils.measure(all_val_label_pos+all_val_label_cns, all_val_pred_pos+all_val_pred_cns) 

        # Mixed validation set
        d = len(val_pred_pos) // 3
        if d < 1 :
            d = 1
        all_val_label_mixed = all_val_label_pos + all_val_label_sns[0:d]+all_val_label_mns[0:d]+all_val_label_cns[0:d]
        all_val_pred_mixed = all_val_pred_pos + all_val_pred_sns[0:d]+all_val_pred_mns[0:d]+all_val_pred_cns[0:d]
        roc_mixed, ap_mixed = utils.measure(all_val_label_mixed, all_val_pred_mixed) 
            
        roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
        ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4 
        
        #print(f' {epoch}: \t {roc_sns:.4f} {roc_mns:.4f} {roc_cns:.4f} {roc_mixed:.4f} {roc_average:.4f} \t {ap_sns:.4f} {ap_mns:.4f} {ap_cns:.4f} {ap_mixed:.4f} {ap_average:.4f}')
          
        # early stop    
        total_roc_average = roc_average
        total_ap_average = ap_average
        if best_roc < roc_average or epoch==0:
            best_roc = roc_average         
            best_ap = ap_average
            best_prev_node_feature = prev_node_feature
            torch.save(encoder.state_dict(), f"{args.folder_name}/encoder_{j}.pkt")
            torch.save(updater.state_dict(), f"{args.folder_name}/updater_{j}.pkt")
            torch.save(decoder.state_dict(), f"{args.folder_name}/decoder_{j}.pkt")
            no_improvement_count = 0          
            if epoch == args.epochs:
                break
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience: 
                break 
                
    encoder.load_state_dict(torch.load(f"{args.folder_name}/encoder_{j}.pkt"))
    updater.load_state_dict(torch.load(f"{args.folder_name}/updater_{j}.pkt"))
    decoder.load_state_dict(torch.load(f"{args.folder_name}/decoder_{j}.pkt"))
                
    # [Test]   
    all_test_pred_pos, all_test_label_pos = [], []
    all_test_pred_sns, all_test_label_sns = [], []
    all_test_pred_cns, all_test_label_cns = [], []
    all_test_pred_mns, all_test_label_mns = [], []
        
    prev_node_feature = init_prev_node_feature       
    for t in (range(len(test_snapshot_data)-1)):
        t_1_prev_node_feature = [i.clone().detach() for i in prev_node_feature]
                
        # load t & t+1 snapshot data (t data : st-han input /t+1 data : pos label + neg label)
        # load t snapshot 
        test_data_dict, snapshot_node_index = data_load.gen_data(args, test_snapshot_data[t], test_snapshot_time[t] )
        test_data, test_g = gen_HG_data(args, test_snapshot_data[t])  
        
        # load t+1 snapshot (for link prediction)
        all_pos_hedge = test_snapshot_data[t+1]
        next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge = preprocess.get_next_samples(all_pos_hedge, all_sns_neg_hedge, all_cns_neg_hedge, all_mns_neg_hedge, args.neg_ratio)
        _, _, test_edges = preprocess.get_data_dict(next_pos_edges, sns_neg_hedge, cns_neg_hedge, mns_neg_hedge, args.eval_mode)
        test_pos_loader, test_neg_sns_loader, test_neg_mns_loader, test_neg_cns_loader = data_load.get_dataloaders(test_edges, args.batch_size, device, None,  args.neg_ratio, label='Test')
        
        pos_list, neg_list = model_eval(args, test_data_dict, test_data, test_g, 
                                        test_pos_loader, test_neg_sns_loader, test_neg_cns_loader, test_neg_mns_loader,
                                        encoder, updater, decoder, 
                                        t_1_prev_node_feature, snapshot_node_index)                      
        test_pred_pos, test_label_pos = pos_list
        test_pred_sns, test_pred_cns, test_pred_mns, test_label_ns = neg_list  
    
        all_test_pred_pos.extend(test_pred_pos)
        all_test_label_pos.extend(test_label_pos)     
        all_test_pred_sns.extend(test_pred_sns)
        all_test_label_sns.extend(test_label_ns) 
        all_test_pred_cns.extend(test_pred_cns)
        all_test_label_cns.extend(test_label_ns)   
        all_test_pred_mns.extend(test_pred_mns)
        all_test_label_mns.extend(test_label_ns)   
    
    # SNS validation set
    roc_sns, ap_sns = utils.measure(all_test_label_pos+all_test_label_sns, all_test_pred_pos+all_test_pred_sns) 
    
    # MNS validation set
    roc_mns, ap_mns = utils.measure(all_test_label_pos+all_test_label_mns, all_test_pred_pos+all_test_pred_mns) 

    # CNS validation set
    roc_cns, ap_cns = utils.measure(all_test_label_pos+all_test_label_cns, all_test_pred_pos+all_test_pred_cns) 

    # Mixed validation set
    d = len(all_test_pred_pos) // 3
    if d < 1 :
        d = 1
    all_test_label_mixed = all_test_label_pos + all_test_label_sns[0:d]+all_test_label_mns[0:d]+all_test_label_cns[0:d]
    all_test_pred_mixed = all_test_pred_pos + all_test_pred_sns[0:d]+all_test_pred_mns[0:d]+all_test_pred_cns[0:d]
    roc_mixed, ap_mixed = utils.measure(all_test_label_mixed, all_test_pred_mixed)    
    
    # Avg score
    roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
    ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4    
               
    total_roc_average = roc_average
    total_ap_average = ap_average
    
    return total_roc_average, total_ap_average 
        