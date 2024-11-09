import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

#from layers.KAN_VSN import GLU, GRN, VariableSelectionNetwork
from layers.VSN import GLU, GRN, VariableSelectionNetwork


from layers.GTF import GTransformer
from einops import rearrange, repeat
import random



# Remark 1
def end_point(l):
    # Find [-s, s]
    return (2+l)**2

end = [end_point(i) for i in range(0, 100)]
end_df = pd.DataFrame(end).T
end_df.columns = end_df.columns+1

def find_interval(value, end_df):
    # Find [-s, s]
    for col in end_df.columns:
        if end_df[col].iloc[0] >= value:
            return col
    return None

# Remark 1
def find_s(value, ratio):
    if ratio == 0.5:
        return int(2* value * (1-0.5)-1)
    else:
        return int((2*value*(1-ratio)-1) )

    

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.LN = nn.LayerNorm(num_numerical_types)
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases  = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = self.LN(x)
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases
    
    
class Model(nn.Module):
    ''' Pretrained model for semi and self-supervised learning for Tabluar dataset'''
    def __init__(self, args, leaf_num, num_of_tree, device):
        super(Model, self).__init__()
        
        self.num_cat = int(args.num_cat)
        self.num_cont = int(args.num_cont)
        self.lambda_value = args.lambda_value
        self.num_of_tree = int(num_of_tree)
        self.max_leaf = int(leaf_num)
        self.cat_mask_ratio = args.cat_mask_ratio
        self.fcn_hidden = args.fcn_hidden
        self.fcn_n_sin = args.fcn_n_sin
        self.dim_head = args.dim_head
        self.device = device
        
        self.new_unique_value = [find_s(value, self.cat_mask_ratio) for value in args.cat_unique_list]
        self.unique_value = torch.tensor(args.cat_unique_list) #[find_s(value, self.cat_mask_ratio) for value in ]

        self.g_input_size =   int((args.num_cont + args.num_cat) * args.embed_dim)

        # Step 0. Leaf embedding
        self.leaf_embedding = nn.Embedding(self.max_leaf, args.embed_dim)
        self.tree_embedding = nn.Linear(self.num_of_tree, int(args.num_cont + args.num_cat))
        
        # Step 1. Continuous variable embeddings
        self.cont_emb_layers = nn.ModuleList()
        for i in range(args.num_cont):
            cont_emb = nn.Linear(1, args.cont_emb, bias = False)
            self.cont_emb_layers.append(cont_emb)
        
        self.numerical_embedder = NumericalEmbedder(args.cont_emb, args.num_cont) # batch * num_cont -> batch * num_cont *cont_emb
        
        # KAN BASED VSN
        #self.vsn_f = VariableSelectionNetwork(input_size = args.cont_emb,
        #                                       num_cont_inputs = args.num_cont,
        #                                       hidden_size = args.num_cont,
        #                                       dropout = 0.1, #0.2
        #                                      device = self.device,
        #                                      fcn_hidden=self.fcn_hidden,
        #                                      fcn_n_sin = self.fcn_n_sin
        #                                      )
        # NORMAL VSN
        self.vsn_f = VariableSelectionNetwork(input_size = args.cont_emb,
                                               num_cont_inputs = args.num_cont,
                                               hidden_size = args.num_cont ,
                                               dropout = 0.1)        
        
        # Step 2. representation 
        self.f_target = GTransformer(categories = tuple(args.cat_unique_list), # tuple containing the number of unique values within each category
                                       num_continuous = args.num_cont,
                                       dim = args.embed_dim,                         
                                       depth = args.depth,
                                       heads = args.heads,
                                       dim_head = self.dim_head,
                                       attn_dropout = 0.1,
                                       ff_dropout = 0.1,
                                      dim_out = 100
                                      )
        self.f_source = GTransformer(categories = tuple(args.cat_unique_list), # tuple containing the number of unique values within each category
                                       num_continuous = args.num_cont,
                                       dim = args.embed_dim,                         
                                       depth = args.depth,
                                       heads = args.heads,
                                       dim_head = self.dim_head,
                                       attn_dropout = 0.1,
                                       ff_dropout = 0.1,
                                      dim_out = 100
                                      )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(args.embed_dim),
            nn.ReLU(),
            nn.Linear(args.embed_dim, args.out_class)
        )
        
    def forward(self, x_cont, x_cat, leaf, mode):   
        
        if mode == "semi":
            # Step 1. Get leaf embedding 
            leaf_emb = self.leaf_embedding(leaf) # B * n_estimation * cat_dim
            if leaf_emb.shape[1] != (self.num_cat + self.num_cont):
                leaf_emb= self.tree_embedding(leaf_emb.permute(0,2,1)).permute(0,2,1)
                
            leaf_emb = self.f_target(None, None, leaf_emb, mode = "semi",return_attn = False) # batch * dim
            logit = self.to_logits(leaf_emb)
            
            return logit, leaf_emb
        
        elif mode == "self":
            # Step 1. Get importance continouse variable using VSN 
            if self.num_cont != 1: 
                x_cont = self.numerical_embedder(x_cont)
                x_cont = x_cont.view(x_cont.shape[0], -1) #x_cont = self.apply_cont_embedding(x_cont)
                x_cont, f_ori_weights = self.vsn_f(x_cont)
            else:
                f_ori_weights = None
                pass

            # Step 2. Tabular Corruption
            positive_mix_cont, negative_mix_cont, cat_nh_pos, cat_nh_1  = self.tab_corruption(x_cont,
                                                                                                 x_cat,
                                                                                                self.new_unique_value, self.unique_value,
                                                                                                lambda_value =self.lambda_value)

            # step 3. Representation f_{\theta}
            # We want match between neg_representation and pos_representation.
            target_representation = self.f_target(positive_mix_cont, cat_nh_pos, None, mode = "self", return_attn = False)   # old one; 
            source_representation = self.f_source(negative_mix_cont, cat_nh_1,  None, mode = "self", return_attn = False)    # New one;  

            return source_representation, target_representation, target_representation, f_ori_weights
      
        
    def random_non_zero_int(self, value, batch_size, cat):
        rand_int = torch.randint(-value, value+1, size=(batch_size, cat)).cuda()
        replacement_values = list(range(-value, 0)) + list(range(1, value+1))
        zero_positions = (rand_int == 0)
        rand_int[zero_positions] = torch.tensor([random.choice(replacement_values) for _ in range(zero_positions.sum())], dtype=rand_int.dtype).cuda()
        return rand_int
                                 
    def tab_corruption(self, x1, x2, new_unique_value, unique_value, lambda_value=1.0):
        '''Returns mixed inputs (categorical and continuos variables).
           x1 : continous variables
           x2 : categorical variables
           new_unique_value : [-s,s]
           
           output :
           mixed_x1 : Soft dimension corruption for cont
           mixed_x2 : Hard dimension corruption for cont
           mixed_x3 : Corruption nh_s_max approach for cat
           mixed_x3 : Corruption nh_s_1   approach for cat
           
        '''
        batch_size,feature_dim = x1.size()
        cat_size = x2.size()[1]   
        nh1_copy = x2.clone() # Tensor object has no attribute 'copy'
        
        # Continous feature corruption 
        index1 = torch.randperm(feature_dim).cuda()
        index2 = torch.randperm(feature_dim).cuda()
        
        index_row = torch.randperm(batch_size).cuda()
        
        x11 = x1.clone()
        x11 = x11[index_row,:]
        
        mixed_x1 = lambda_value * x1 + (1 - lambda_value) * x11[:,  index1]
        mixed_x2 = (1 - lambda_value) * x1 + lambda_value * x11[:,  index2]

        unique_value = unique_value.cuda()
        # categorical corruption nh[r_k%] # positive        
        #rand_bins = [torch.randint(-value, value + 1, size=(batch_size, 1)).cuda() for value in new_unique_value] 
        rand_bins = [self.random_non_zero_int(value, batch_size, 1) for value in new_unique_value]
        rand_bin = torch.cat(rand_bins, dim=1)
        mixed_x3 = rand_bin + x2
        condition = torch.logical_or(mixed_x3 < 0, mixed_x3 >= torch.tensor(unique_value))
        mixed_x3 = torch.where(condition, x2, mixed_x3)
        
        # categorical corruption nh[s_{1}] (That is a maximize version)
        rand_nh1 = self.random_non_zero_int(1, batch_size, cat_size)  #torch.randint(-1, 2, size = (batch_size, cat_size)).cuda()
        mixed_x4 = rand_nh1 + nh1_copy
        condition2 = torch.logical_or(mixed_x4 < 0, mixed_x4 >= torch.tensor(unique_value))
        mixed_x4 = torch.where(condition2, nh1_copy, mixed_x4)
        
        return mixed_x1, mixed_x2, mixed_x3, mixed_x4
    
    
    def apply_cont_embedding(self, x):
        '''
        Apply continous variable to embedding space 
        '''
        
        if len(x.size()) == 3:
            conti_vectors = []
            for i in range(self.num_cont):
                conti_emb = self.cont_emb_layers[i](x[:, :, i:i+1])
                cont_vectors.append(conti_emb)  
            conti_emb = torch.cat(cont_vectors, dim = 2)
        else:
            conti_vectors = []
            for i in range(self.num_cont):
                conti_emb = self.cont_emb_layers[i](x[:, i:i+1])
                conti_vectors.append(conti_emb)
            conti_emb = torch.cat(conti_vectors, dim = 1)
            
        return conti_emb

    
    
        
