import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from utils.data_utils import MultiColumnLabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
import argparse


class SSL_dataloader(Dataset):
    def __init__(self, root_path, mode, train_csv, test_csv, cat_names, args, ii, scale = True):
        
        self.root_path = root_path
        self.scale = scale
        self.mode = mode
        self.args = args
        self.estimation = args.num_cont + args.num_cat
        self.train = train_csv
        self.test  = test_csv
        self.cat_names = cat_names
        self.label_ratio = args.label_ratio
        self.is_noise = args.is_noise
        
        self.tune_tree = args.tune_tree
        self.reg_alpha = args.reg_alpha
        self.reg_lambda = args.reg_lambda
        self.reg_depth = args.reg_depth

        self.tree_lr = args.tree_lr

        self.ii = ii
                
        self.__read_data__()
        
        # for len function
        if mode == "train" :
            self.data = pd.DataFrame(self.train_cont)
        elif mode == "val" :
            self.data = pd.DataFrame(self.valid_cont)
        elif mode == "test":
            self.data = pd.DataFrame(self.test_cont)
        elif mode == "fine_tune":
            self.data = pd.DataFrame(self.df_semi_cont)
            
        elif mode == "semi_train":
            self.data = pd.DataFrame(self.semi_train_y)
        elif mode == "semi_valid":
            self.data = pd.DataFrame(self.semi_valid_y)
        elif mode == "semi_test":
            self.data = pd.DataFrame(self.test_y)
            
    def __read_data__(self):
        #self.scaler = StandardScaler() #StandardScaler MinMaxScaler, QuantileTransformer(random_state=0), PowerTransformer(method='box-cox', standardize=False)
        #self.scaler = 
        
        # Step 1. Data load
        df_train_raw = pd.read_csv(os.path.join(self.root_path,  self.train))
        df_test_raw  = pd.read_csv(os.path.join(self.root_path,  self.test))
        cat_feat_names = pd.read_csv(os.path.join(self.root_path, self.cat_names)).T.values[0]
        cat_feat_names = np.append(cat_feat_names, "class")

        
        # Step 1.1 Calculate the num of feat.
        cat_start_point = df_train_raw.shape[1] - df_train_raw[cat_feat_names].shape[1]
        unique_class =  len(df_train_raw["class"].unique())

        
        #Quantile 
        quantile_train = np.copy(df_train_raw.iloc[:, :cat_start_point].values).astype(np.float64)
        stds = np.std(quantile_train, axis=0, keepdims=True)
        noise_std = 1e-3 / np.maximum(stds, 1e-3)
        quantile_train += noise_std * np.random.randn(*quantile_train.shape)  
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=1004)
        
        
        # Step 2. Label Encoder
        df_train_raw = MultiColumnLabelEncoder(columns = cat_feat_names).fit_transform(df_train_raw)      #fit_transform
        df_test_raw =  MultiColumnLabelEncoder(columns = cat_feat_names).transform(df_test_raw)    #$transform
        
        # Step 2.1 data set split for train valid
        df_train_raw, df_valid_raw = self.__data_sampling__(df_train_raw, 0.2) # For train valid split
        df_train_raw, df_semi_train_raw = self.__data_sampling__(df_train_raw, self.label_ratio)
        df_valid_raw, df_semi_val_raw   = self.__data_sampling__(df_valid_raw, self.label_ratio) # For get leaves 
        
        ####################  If you want noised setting! use this code. ####################
        if self.is_noise == 1:
            noise_prob = 0.2
            np.random.seed(20205289)
            for i, row in df_semi_train_raw.iterrows():
                random_num = np.random.rand()
                unique_class = df_semi_train_raw["class"].unique()
                if random_num < noise_prob:
                    noised_label = int(np.random.choice(unique_class))
                    while noised_label == row["class"]:
                        noised_label = np.random.choice(unique_class)
                    df_semi_train_raw.at[i, "class"] = noised_label
        else:
            pass
        #####################################################################################
        
        # Step 2.3. Tree embedding
        train_semi_rows = df_semi_train_raw.shape[0]
        valid_semi_rows = train_semi_rows + df_semi_val_raw.shape[0]
        
        if self.tune_tree == False:
            xgb = XGBClassifier(n_estimators=self.estimation,
                            max_depth=4,                
                            n_jobs=-1, seed = self.ii ) # Default setting 
            
        else:
            xgb = XGBClassifier(n_estimators=self.estimation,
                            max_depth=self.reg_depth,      
                            reg_alpha = self.reg_alpha,
                            reg_lambda = self.reg_lambda,
                            learning_rate  = self.tree_lr,
                            n_jobs=-1, seed = self.ii ) # Default setting 
            
        xgb.fit(df_semi_train_raw.iloc[:, :-1], df_semi_train_raw.iloc[:, -1])
        
        X_train_semi_leaves = xgb.apply(df_semi_train_raw.iloc[:, :-1])
        X_val_semi_leaves =  xgb.apply(df_semi_val_raw.iloc[:, :-1])
        X_test_semi_leaves = xgb.apply(df_test_raw.iloc[:, :-1])

        X_leaves = np.concatenate((X_train_semi_leaves,X_val_semi_leaves,X_test_semi_leaves), axis=0)
        
        transformed_leaves, self.leaf_num, new_leaf_index = self.__process_leaf_idx__(X_leaves)
        self.train_leaves, self.valid_leaves, self.test_leaves = transformed_leaves[:train_semi_rows],\
                                                                   transformed_leaves[train_semi_rows:valid_semi_rows],\
                                                                   transformed_leaves[valid_semi_rows:]
        
        self.num_of_tree = transformed_leaves.shape[1]
        
        
        # Step 3. Split continous feat
        df_train_cont = df_train_raw.iloc[:, :cat_start_point]
        df_valid_cont = df_valid_raw.iloc[:, :cat_start_point]
        df_test_cont  = df_test_raw.iloc[:, :cat_start_point]
        df_semi_cont = df_semi_train_raw.iloc[:, :cat_start_point]
                
        # Step 4. Standard scaling for continous variables
        self.scaler.fit(quantile_train)
        #self.scaler.fit(df_train_cont.values)
        self.train_cont = self.scaler.transform(df_train_cont.values)
        self.valid_cont = self.scaler.transform(df_valid_cont.values)
        self.test_cont  = self.scaler.transform(df_test_cont.values)
        self.df_semi_cont = self.scaler.transform(df_semi_cont.values)
        
        # Step 5. Split categorical feat
        self.train_cat = df_train_raw.iloc[:, cat_start_point:-1].values  # Exclude label
        self.valid_cat = df_valid_raw.iloc[:, cat_start_point:-1].values  # Exclude label
        self.test_cat  = df_test_raw.iloc[:,  cat_start_point:-1].values  # Exclude label
        self.df_semi_cat = df_semi_train_raw.iloc[:, cat_start_point:-1].values
        
        # Step 6. Labeling
        self.train_y = df_train_raw.iloc[:,-1:].values
        self.valid_y = df_valid_raw.iloc[:,-1:].values
        self.test_y = df_test_raw.iloc[:,-1:].values  
        self.semi_train_y = df_semi_train_raw.iloc[:,-1:].values  

        self.semi_train_y = df_semi_train_raw.iloc[:,-1:].values  #  
        self.semi_valid_y = df_semi_val_raw.iloc[:,-1:].values    #
        
        
    def __getleafnum__(self):
        return self.leaf_num, self.num_of_tree
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if self.mode == "train":
            cont_variable = self.train_cont[idx]
            cat_variable = self.train_cat[idx]
            labels = self.train_y[idx]
            
            return torch.tensor(cont_variable, dtype = torch.float32),\
                   torch.tensor(cat_variable, dtype = torch.long),\
                   torch.tensor(labels, dtype = torch.long)
            
        elif self.mode == "val":
            cont_variable = self.valid_cont[idx]
            cat_variable = self.valid_cat[idx]
            labels = self.valid_y[idx]
            
            return torch.tensor(cont_variable, dtype = torch.float32),\
                   torch.tensor(cat_variable, dtype = torch.long),\
                   torch.tensor(labels, dtype = torch.long)

        elif self.mode == "test": 
            cont_variable = self.test_cont[idx]
            cat_variable = self.test_cat[idx]
            labels = self.test_y[idx]
            return torch.tensor(cont_variable, dtype = torch.float32),\
                   torch.tensor(cat_variable, dtype = torch.long),\
                   torch.tensor(labels, dtype = torch.long)
        
        elif self.mode == "fine_tune":
            cont_variable = self.df_semi_cont[idx]
            cat_variable = self.df_semi_cat[idx]
            labels = self.semi_train_y[idx]
            
            return torch.tensor(cont_variable, dtype = torch.float32),\
                   torch.tensor(cat_variable, dtype = torch.long),\
                   torch.tensor(labels, dtype = torch.long)

        elif self.mode == "semi_test": 
            test_leaves = self.test_leaves[idx]
            labels = self.test_y[idx]
            return torch.tensor(test_leaves, dtype = torch.long), torch.tensor(labels, dtype = torch.long)        
        
        elif self.mode == "semi_train":
            train_leaves = self.train_leaves[idx]
            labels = self.semi_train_y[idx]
            return torch.tensor(train_leaves, dtype = torch.long), torch.tensor(labels, dtype = torch.long)        

        elif self.mode == "semi_valid":
            valid_leaves = self.valid_leaves[idx]
            labels = self.semi_valid_y[idx]
            return torch.tensor(valid_leaves, dtype = torch.long), torch.tensor(labels, dtype = torch.long)       
    
        
    def __data_sampling__(self, dataset, ratio):
        
        split = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=1004)
        for train_idx, valid_idx in split.split(dataset, dataset["class"]):
            df_train_ssl = dataset.loc[train_idx].reset_index().iloc[:, 1:]
            df_valid_ssl = dataset.loc[valid_idx].reset_index().iloc[:, 1:]
            
        return df_train_ssl, df_valid_ssl
    def __process_leaf_idx__(self, X_leaves): 
        '''
        This function is to convert the output of XGBoost model to the input of DATE model.
        For an individual import, the output of XGBoost model is a list of leaf index of multiple trees.
        eg. [1, 1, 10, 9, 30, 30, 32, ... ]
        How to distinguish "node 1" of the first tree from "node 1" of the second tree?
        How to distinguish "node 30" of the fifth tree from "node 30" of the sixth tree?
        This function is to assign unique index to every leaf node in all the trees. 
        This function returns;
        - lists of unique leaf index;
        - total number of unique leaf nodes; and
        - a reference table (dictionary) composed of "unique leaf index", "tree id", "(previous) leaf index". 
        '''
        leaves = X_leaves.copy()
        new_leaf_index = dict() # dictionary to store leaf index
        total_leaves = 0
        for c in range(X_leaves.shape[1]): # iterate for each column (ie. 100 trees)
            column = X_leaves[:,c]
            unique_vals = list(sorted(set(column)))
            new_idx = {v:(i+total_leaves) for i,v in enumerate(unique_vals)}
            for i,v in enumerate(unique_vals):
                leaf_id = i+total_leaves
                new_leaf_index[leaf_id] = {c:v}
            leaves[:,c] = [new_idx[v] for v in column]
            total_leaves += len(unique_vals)

        assert leaves.ravel().max() == total_leaves - 1
        return leaves,total_leaves,new_leaf_index        
        
