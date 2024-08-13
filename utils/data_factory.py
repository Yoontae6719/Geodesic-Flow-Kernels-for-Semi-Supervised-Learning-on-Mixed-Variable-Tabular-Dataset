import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import SSL_dataloader

def data_provider(args, flag, ii):
    '''
    data provider function : 
    flag (str) : train, val, test 
    '''
    

    if flag == "train":
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    elif flag == "val":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        
    elif flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        
    elif flag == "fine_tune":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    elif flag == "semi_test":
        shuffle_flag = False
        drop_last = False
        batch_size = 1        
        
    elif flag == "semi_train":
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    elif flag == "semi_valid":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        
    elif flag == "semi_train_infer":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        
    data_set = SSL_dataloader(root_path =   args.root_path,
                                         train_csv = args.train_csv,
                                         test_csv = args.test_csv,
                                         cat_names = args.cat_names,
                                         args = args, ii=ii,
                                         mode = flag)
    
    
    

    print(flag, len(data_set), data_set.__getleafnum__())
    leaf_num = data_set.__getleafnum__()[0]
    tree_num = data_set.__getleafnum__()[1]
    
    data_loader = DataLoader(
                            data_set,
                            batch_size=batch_size,
                            shuffle=shuffle_flag,
                            num_workers=args.num_workers,
                            drop_last=drop_last)
    
    return data_set, data_loader, leaf_num, tree_num
