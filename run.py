import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from scipy.special import softmax

#import umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def min_max_normalize(df):
    result = df.copy()
    for i in df.columns:
        min_val = df[i].min()
        max_val = df[i].max()
        result[i] = (df[i] - min_val) / (max_val - min_val)
    return result



def main():
    
    # 1. define random seed
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 2. parser
    parser = argparse.ArgumentParser(description='Semi and Self-supervised learning for Tabluar data')
    
    # 2.1. basic configs
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='GKSMT', help='model name, options: [will be updated]')
    parser.add_argument('--label_ratio', type = float, default = 0.2, help = "tranining lable ratio")

    # 2.2. data configs
    parser.add_argument('--data', type=str, required=True, default='AD', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./OpenML-CC18-cat/adult/', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--is_noise', type=int, default=0, help='injection label noise in the traninig data')
    parser.add_argument('--train_csv', type=str, default='trainIMP.csv', help='location of train csv')
    parser.add_argument('--test_csv', type=str, default='testIMP.csv', help='location of test csv')
    parser.add_argument('--cat_names', type=str, default='cat_feat_names.csv', help='location of cat_feat_names')
    
    # 2.3 model configs - Self supervised learning task
        # 2.3.1. embedding configs
    parser.add_argument("--num_cont", type = int, default = 6, help = "number of continouse variables")
    parser.add_argument("--num_cat", type = int, default = 8, help = "number of categorical variables")
    parser.add_argument("--cat_unique_list", nargs='+', type=int, default = [9, 17, 8, 16, 6, 5, 3, 43],
                                              help = "unique characteristics of categorical variables e.g., --cat_unique_list 12 5 3 4") 
    parser.add_argument("--fcn_hidden", type = int, default = 6, help = "number of hidden variable for KAN-FCL")
    parser.add_argument("--fcn_n_sin", type = int, default = 3, help = "number of Sin")

    #conti
    parser.add_argument("--cont_emb", type = int, default = 24, help = "embedding dimension of continous variable")
    
    # Transforemr
    parser.add_argument("--cat_mask_ratio", type = float, default = 0.3, help = "dimension of categorical variable")
    parser.add_argument("--embed_dim", type = int, default = 24, help = "dimension of embedding dimension for transformer")
    parser.add_argument("--heads", type = int, default = 8, help = "heads of transformer")
    parser.add_argument("--depth", type = int, default = 8, help = "depth of transformer")
    parser.add_argument("--dim_head", type = int, default = 64, help = "dimension of heads")

    # non
    parser.add_argument("--lambda_value", type = float, default = 0.7, help = "lambda_value for cutmix") #0.1, 0.3, 0.5, 0.7, 0.9
    
    # 2.3.2 modeling f configs
    parser.add_argument('--output_attention', type = bool, default=False, help='whether to output attention in encoder')
    parser.add_argument("--mult", type = int, default = 3, help = "expention multipler of MLP") #3 4
    parser.add_argument('--dropout', type=float, default=0.05, help = 'dropout rate')
    parser.add_argument('--loss_weight', type=float, default=1.0, help = 'loss_weight 0.1 to 1')
    
    # 2.3.3 modeling g configs
    parser.add_argument("--out_dim", type = int, default = 12, help = "out_dim")

    # 2.4 Model configs - Classification task
    parser.add_argument("--out_class", type = int, default = 2, help = "classification class of output dimension")
    
    # 2.5. Tree tune 
    parser.add_argument("--tune_tree", type = bool, default = False, help = "Tree Tuen")
    parser.add_argument("--reg_depth", type = int, default = 0.001, help = "depths")    
    parser.add_argument("--reg_alpha", type = float, default = 0.001, help = "L1 regularization term")
    parser.add_argument("--reg_lambda", type = float, default = 0.001, help = "L2 regularization term")
    parser.add_argument("--tree_lr", type = float, default = 0.001, help = "tree lr")
    
    # 2.5. Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs') #epoch
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--sim_loss', type=str, default='GKP', help='loss function')

    # 2.6. GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
   # print("*******************ARG.SGPU      ",args.devices)

    print('Args in experiment:')
    print(args)
    print("==")
    print(args.loss_weight)
    
    Exp = Exp_Main
    
    if args.is_training:
        for ii in range(3): # for robustness
        
            setting = 'noise_{}_label_{}_{}_{}_ii_{}'.format(
                args.is_noise,
                args.label_ratio,
                args.data,
                int(args.loss_weight) * 10,
                ii)
            
            exp = Exp(args,ii) # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            
            folder_path = './results/' + "GKP_IMP/" + setting + '/' #GKP_Semiabl, GKP_Semiloss, GKP_Semi(will do)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)            
            
            results_list = []
            total_test_logit, test_y_list, total_test_repre = exp.test_repre_leaf(setting, "semi_test") # Directly 
            test_logit = [v.cpu().detach().numpy() for v in total_test_logit]
            test_logit_feats = np.concatenate([v for v in test_logit])

            batch_test_y = [v.cpu().detach().numpy() for v in test_y_list ]
            batch_test_y = np.array(batch_test_y).flatten()
            batch_test_y = batch_test_y.reshape(-1)  

            predicted_classes = np.argmax(test_logit_feats, axis=1)
            probabilities = softmax(test_logit_feats, axis=1)
            
            leaf_acc = accuracy_score(batch_test_y, predicted_classes)
            leaf_f1 = f1_score(batch_test_y, predicted_classes, average='macro')
            
            if args.out_class ==2:
                pos_prob = probabilities[:, 1]
                leaf_auc = roc_auc_score(batch_test_y, pos_prob)
            else:
                leaf_auc = roc_auc_score(batch_test_y, probabilities, multi_class  ="ovr")

            results_list.append({"data" : args.data, "model": "Leaf",
                                         "test_acc": leaf_acc,
                                         "roc_auc_score": leaf_auc,
                                         "f1_score": leaf_f1, "ii" : ii})

            # If we use the representation, then we can use logit regression for fine tune.
            total_train_logit, train_y_list, total_train_repre = exp.test_repre_leaf(setting, "semi_train") 
            
            train_logit = [v.cpu().detach().numpy() for v in total_train_logit ]
            train_logit = np.concatenate([v for v in train_logit])

            batch_train_y = [v.cpu().detach().numpy() for v in train_y_list ]
            batch_train_y = np.array(batch_train_y).flatten()
            batch_train_y = batch_train_y.reshape(-1)  

            train_repre = [v.cpu().detach().numpy() for v in total_train_repre ]
            train_repre = np.concatenate([v for v in train_repre])

            test_repre = [v.cpu().detach().numpy() for v in total_test_repre ]
            test_repre = np.concatenate([v for v in test_repre])
            
            #imp_variable = exp.test_repre(setting, "semi_train")
            #print(imp_variable.shape)
           
            for c in [0.01, 0.1, 1, 10]:
                clf = LogisticRegression(max_iter=1200, solver='lbfgs', C=c, multi_class='multinomial')
                clf.fit(train_repre, batch_train_y)

                te_acc = clf.score(test_repre , batch_test_y)
                y_pred_proba = clf.predict_proba(test_repre)
                if y_pred_proba.shape[1] ==2:
                    pos_prob = y_pred_proba[:, 1]
                    auc = roc_auc_score(batch_test_y, pos_prob)
                else:
                    auc = roc_auc_score(batch_test_y, y_pred_proba, multi_class  ="ovr")

                f1 = f1_score(batch_test_y, clf.predict(test_repre), average='macro')
                results_list.append({"data" : args.data, "model": "LogReg_" + str(c),
                                                     "test_acc": te_acc,
                                                     "roc_auc_score": auc,
                                                     "f1_score": f1, "ii" : ii})
            
            

            di = args.data + "{}.csv".format(ii)
            pd.DataFrame(results_list).to_csv(os.path.join(folder_path, di), index= False)
            
            print(">>>>> Leaf data")
            torch.cuda.empty_cache()
        
    
        
if __name__ == "__main__":
    main()
    
    
