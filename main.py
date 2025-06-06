# import wandb
import os
import numpy as np
import statistics
import warnings

import torch
import utils, data_load
import train_msg_att, train_msg_mlp, train_msg_mlp_moving
from models.decoder import *
from models.updater import *
from models.Ours_msg_att import *
from models.Ours_msg_mlp import *

warnings.simplefilter("ignore")
    
def run(args):
    #---------------- get args -------------------#
    print()
    print('============================================ New Data Start =================================================')
    DATA = args.dataset_name
    print(f'[ ARGS ]')

    print(args)

    test_env = f'seed:{args.seed}_model:{args.model}_dim:{args.dim_vertex}_bs:{args.batch_size}_neg:{args.ns_mode},{args.neg_ratio}_updater:{args.updater}'
    test_env2 = f'lr:{args.lr}_agg:{args.aggregator}_ss:{args.snapshot_size}_updater:{args.updater}'
    test_env3 = f'cs:{args.contrast_ratio}_layer:{args.num_layers},{args.time_layers}'
    module_env = f"time_concat:{args.time_concat}_edge_concat:{args.edge_concat}"
    if args.time_concat == 'true':
        if args.edge_concat == 'true':
            module_env = f"time_concat:O  edge_concat:O"
            concat_env = f"time:{args.time_concat_mode} edge:{args.edge_concat_mode}"
        if args.edge_concat == 'false':
            module_env = f"time_concat:O  edge_concat:X"
            concat_env = f"time:{args.time_concat_mode} edge:X"
    elif args.time_concat == 'false':
        if args.edge_concat == 'true':
            module_env = f"time_concat:X  edge_concat:O"
            concat_env = f"time:X  edge:{args.edge_concat_mode}"
        if args.edge_concat == 'false':
            module_env = f"time_concat:X  edge_concat:X"
            concat_env = f"time:X  edge: X"
            
    args.folder_name = f"logs/{DATA}/{args.eval_mode}/{test_env}/{test_env2}/{test_env3}/{module_env}/{concat_env}"
    os.makedirs(args.folder_name, exist_ok=True)    

    print(f"[ SETTING ]")
    print(f"{args.folder_name}\n")
    print(f"MODULE: {module_env}\n")
    print(f"CONCAT: {concat_env}\n")
    
    f_log = open(f"{args.folder_name}.log", "w")
    f_log.write(f"args: {args}\n")
        
        
    roc_list = []
    ap_list = []   

    for i in range(args.exp_num): # number of splits (default: 5)

        # change seed
        utils.set_random_seeds(i)
        
        if args.eval_mode == 'live_update':
            snapshot_data, snapshot_time, num_node = data_load.load_snapshot(args, DATA)
        elif args.eval_mode == 'fixed_split':
            full_data, full_time, num_node = data_load.load_fulldata(args, DATA)
            
        args = data_load.gen_init_data(args, num_node)
        device = args.device

        f_log.write(f'============================================ Experiments {i} ==================================================')
        # Initialize models
        # 0. Args needed
        time_layer = args.time_layers
        time_concat = args.time_concat
        time_concat_mode = args.time_concat_mode
        edge_concat = args.edge_concat
        edge_concat_mode = args.edge_concat_mode
        
        # 1. Hypergraph encoder
        if args.model == 'Ours_msg_att':   
            HypergraphEncoder = Ours_msg_att(time_layer, time_concat, edge_concat, time_concat_mode, edge_concat_mode, 
                                        args.dim_vertex, args.dim_hidden, args.dim_edge, args.dim_time, args.num_heads, args.num_inds)  
        elif args.model == 'Ours_msg_mlp':   
            HypergraphEncoder = Ours_msg_mlp(time_layer, time_concat, edge_concat, time_concat_mode, edge_concat_mode, 
                                        args.dim_vertex, args.dim_hidden, args.dim_edge, args.dim_time, args.num_heads, args.num_inds)  
        encoder = HypergraphEncoder.to(device)
        # 2. Spatio temporalLayer
        if args.updater == 'gru':
            updater = GRULayer(dim_in= args.dim_vertex, dim_out=args.dim_vertex)
            updater = updater.to(device)
        elif args.updater == 'lstm':
            updater = LSTMLayer(input_size= args.dim_vertex, hidden_size= args.dim_vertex, num_layers=2)
            updater = updater.to(device)
        elif args.updater == 'mlp':
            updater = MLP(args.dim_vertex * 2, args.dim_hidden *2 ,args.dim_vertex)   
            updater = updater.to(device)            
        elif args.updater == 'moving_avg':
            pass
        
        #3. Decoder (classifier) for hyperedge prediction
        cls_layers = [args.dim_vertex, 128, 8, 1]
        decoder = Decoder(cls_layers)
        decoder = decoder.to(device)
      
        if args.updater == 'moving_avg':  
            optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            if args.eval_mode == 'live_update':
                auc_roc, ap = train_msg_mlp_moving.live_update(args, encoder, decoder, optimizer, scheduler, 
                                                    snapshot_data, snapshot_time, f_log, i)
                
            elif args.eval_mode == 'fixed_split':
                auc_roc, ap = train_msg_mlp_moving.live_update(args, encoder, decoder, optimizer, scheduler, 
                                                        snapshot_data, snapshot_time, f_log, i)
        
        else:  
            optimizer = torch.optim.Adam(list(encoder.parameters())+ list(updater.parameters()) + list(decoder.parameters()), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            if args.eval_mode == 'live_update':
                if args.model == 'Ours_msg_att':  
                    auc_roc, ap = train_msg_att.live_update(args, encoder, updater, decoder, optimizer, scheduler, 
                                                        snapshot_data, snapshot_time, f_log, i)
                elif args.model == 'Ours_msg_mlp':   
                    auc_roc, ap = train_msg_mlp.live_update(args, encoder, updater, decoder, optimizer, scheduler, 
                                                        snapshot_data, snapshot_time, f_log, i)
            elif args.eval_mode == 'fixed_split':
                if args.model == 'Ours_msg_att':  
                    auc_roc, ap = train_msg_att.fixed_split(args, encoder, updater, decoder, optimizer, scheduler, 
                                                    full_data, full_time, f_log, i)
                elif args.model == 'Ours_msg_mlp':   
                    auc_roc, ap = train_msg_mlp.fixed_split(args, encoder, updater, decoder, optimizer, scheduler, 
                                                    full_data, full_time, f_log, i)
        
        roc_list.append(auc_roc)
        ap_list.append(ap)
        
        print()  

    final_roc = sum(roc_list)/len(roc_list)
    final_ap = sum(ap_list)/len(ap_list)

    if args.exp_num > 1:
        std_roc = statistics.stdev(roc_list)
        std_ap = np.std(ap_list)
    else:
        std_roc = 0.0 
        std_ap = 0.0 

    f_log.write(f'============================================ Test End =================================================\n')
    f_log.write(f"[ SETTING ]\n")
    f_log.write(f"logs/{DATA}/{test_env}/\n")
    f_log.write(f"MODULE: {module_env}\n")
    f_log.write(f"CONCAT: {concat_env}\n")
    
    f_log.write('[ BEST ]\n')
    f_log.write('AUROC\t AP\t \n')
    f_log.write(f'{max(roc_list):.4f}\t{max(ap_list):.4f}\n')
    f_log.write('[ AVG ]\n')
    f_log.write('AUROC\t AP\t \n')
    f_log.write(f'{final_roc:.4f}\t{final_ap:.4f}\n')
    f_log.write(f'===============================================================================================================')
    
    f_log.close
    
    
if __name__ == '__main__':
    
    args = utils.parse_args()# Project and sweep configuration

    run(args)