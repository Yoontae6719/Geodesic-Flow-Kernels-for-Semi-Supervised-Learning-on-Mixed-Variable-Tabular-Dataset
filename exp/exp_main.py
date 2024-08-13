from exp.exp_basic import Exp_Basic
from models import GKSMT
from utils.tools import EarlyStopping
from utils.metrics import InfoNCE

from sklearn.metrics import roc_auc_score
from utils.data_factory import data_provider

from algorithms.GKF_solver import GFK

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

import warnings
import matplotlib.pyplot as plt
from torchmetrics.regression import CosineSimilarity



#MSE loss
warnings.filterwarnings('ignore')

def barlow(x, y, lmbd = 5e-3):
    bs = x.size(0)
    emb = x.size(1)

    xNorm = (x - x.mean(0)) / x.std(0)
    yNorm = (y - y.mean(0)) / y.std(0)
    crossCorMat = (xNorm.T@yNorm) / bs
    loss = (crossCorMat*lmbd - torch.eye(emb, device=torch.device('cuda'))*lmbd).pow(2)
    
    return loss.sum()


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


__all__ = ['align_loss', 'uniform_loss']


def init_func(m, init_type = "xavier", init_gain = 0.02):
    ### https://github.com/vaseline555/Federated-Averaging-PyTorch/blob/main/src/utils.py
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        else:
            raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    
    elif classname.find('BatchNorm1d') != -1 or classname.find('InstanceNorm1d') != -1 or classname.find('LayerNorm') != -1:
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)  
        


class Exp_Main(Exp_Basic):
    def __init__(self, args, ii):
        super(Exp_Main, self).__init__(args, ii)
        
        self.label_ratio = args.label_ratio
        self.embed_dim = args.embed_dim
        self.cont_emb = args.cont_emb
        self.depth = args.depth
        self.heads = args.heads
        self.data = args.data
        self.itr = args.itr
        self.out_class = args.out_class
        self.lambda_value = args.lambda_value
        self.loss_weight = args.loss_weight
        self.sim_loss = args.sim_loss
        
        self.num_variable = args.num_cont + args.num_cat
        self.cont_ratio = args.num_cont / self.num_variable
        self.cat_ratio  = args.num_cat / self.num_variable
        
        self.ii = ii
                
        
    def _build_model(self, leaf_num, num_of_tree, device):
        model_dict = {
            'GKSMT': GKSMT,
        }
        model = model_dict[self.args.model].Model(self.args, leaf_num, num_of_tree, self.device).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            init_func(model, init_type = "xavier", init_gain = 0.02)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model

    
    def _get_data(self, flag):
        data_set, data_loader, leaf_num, num_of_tree = data_provider(self.args, flag, self.ii)
        #self.args.leaf_num = leaf_num
        return data_set, data_loader, leaf_num, num_of_tree
    
    def _select_optimizer(self):
        
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim
    
    def _select_criterion(self, task_mode):
        '''
        CrossEntropy loss + weight probability
        '''
        if task_mode == "Classification":
            criterion = nn.CrossEntropyLoss(label_smoothing=0.15) 
        elif task_mode == "GKP":
            criterion = GFK().fit
        elif task_mode == "barlow":
            criterion = barlow
        elif task_mode == "align":
            criterion = align_loss
        elif task_mode == "info":
            criterion = InfoNCE()
        elif task_mode == "Cosine":
            criterion = CosineSimilarity(reduction = 'mean')
        return criterion
    
    
    def accuracy_function(self, real, pred):    
        real = real.cpu()
        pred = torch.argmax(pred, dim=1).cpu()
        score = f1_score(real, pred, average = "macro")
        acc = accuracy_score(real, pred)
        return score, acc

    
    def accuracy_function_test(self, real, pred):    
        score = f1_score(real, pred, average = "macro")
        acc = accuracy_score(real, pred)
        return score, acc

    def auroc_function_test(self, real, pred):
        row_sums = torch.sum(pred, 1) 
        row_sums = row_sums.repeat(1, self.out_class)
        pred = torch.div( pred , row_sums )
        pred = pred.view(-1, 1)

        if self.out_class == 2:
            roauc = roc_auc_score(real, pred, average = "micro")
        else:
            roauc = roc_auc_score(real, pred, multi_class = "ovr")

        return roauc

    def vali(self, vali_data, vali_loader, semi_val_loader, CLS_criterion, GKP_criterion):
        vali_CLS_loss = []
        vali_GKP_loss = []
        total_val_score = []
        total_val_acc = [] 
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_cont, batch_cat, batch_y) in enumerate(vali_loader): 
                
                # Feed input to the model
                batch_cont = batch_cont.float().to(self.device)                
                batch_cat  = batch_cat.to(self.device)
                batch_y = batch_y.to(self.device)
                
                neg_feat, pos_feat, pos_repre  = self.model(batch_cont, batch_cat, None, mode = "self")
                
                # Compute loss
                if self.sim_loss == "GKP":
                    loss = GKP_criterion(neg_feat.detach(), pos_feat) 
                else:
                    loss = GKP_criterion(neg_feat, pos_feat) 
                vali_GKP_loss.append(loss.item())
                
            for j, (batch_leaves, batch_yy) in enumerate(semi_val_loader): 
                batch_leaves = batch_leaves.long().to(self.device)                
                batch_yy = batch_yy.to(self.device)
                
                y_hat,_  = self.model(None, None, batch_leaves, mode = "semi")
                cls_loss = CLS_criterion(y_hat, batch_yy.squeeze(dim=-1)) * self.loss_weight
                score, acc = self.accuracy_function(batch_yy, y_hat)
                total_val_score.append(score)
                total_val_acc.append(acc)
                vali_CLS_loss.append(cls_loss.item())         
                

                
       
            
        valid_GKP_loss = np.average(vali_GKP_loss)
        valid_CLS_loss = np.average(vali_CLS_loss)
        vliad_TOT_loss = valid_GKP_loss + valid_CLS_loss

        total_val_score = np.average(total_val_score)
        total_val_acc = np.average(total_val_acc)   
        self.model.train()
        
        return vliad_TOT_loss, total_val_score, total_val_acc, valid_GKP_loss, valid_CLS_loss
                
    
    def train(self, setting):
        
        train_data, train_loader,_,_ = self._get_data(flag='train')
        vali_data, vali_loader,_,_   = self._get_data(flag='val')
        test_data, test_loader,_,_   = self._get_data(flag='test')

        semi_train_data, semi_train_loader,_,_   = self._get_data(flag='semi_train')
        semi_valid_data, semi_valid_loader,_,_   = self._get_data(flag='semi_valid')
        
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.StepLR(optimizer=model_optim, step_size=10, gamma=0.5)
        
        CLS_criterion = self._select_criterion(task_mode = "Classification")
        GKP_criterion = self._select_criterion(task_mode = self.sim_loss)
                        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_CLS_loss = []
            train_GKP_loss = []
            
            total_score = []
            total_acc = []         
                                
            self.model.train()
            
            epoch_time = time.time()
            
            for i, (batch_cont, batch_cat, batch_y) in enumerate(train_loader): 
                iter_count += 1    
                model_optim.zero_grad()
                
                # Feed input to the model
                batch_cont = batch_cont.float().to(self.device)                
                batch_cat  = batch_cat.to(self.device)
                neg_feat, pos_feat, pos_repre  = self.model(batch_cont, batch_cat, None, mode = "self")
                if self.sim_loss == "GKP":
                    loss = GKP_criterion(neg_feat.detach(), pos_feat) 
                else:
                    loss = GKP_criterion(neg_feat, pos_feat) 
                train_GKP_loss.append(loss.item())
                
                loss.backward(retain_graph=True)
                model_optim.step()
                
            for j, (batch_leaves, batch_yy) in enumerate(semi_train_loader): 
                iter_count += 1
                model_optim.zero_grad()
                
                batch_leaves = batch_leaves.long().to(self.device)                
                batch_yy = batch_yy.to(self.device)
                
                y_hat,_  = self.model(None, None, batch_leaves, mode = "semi")
                cls_loss = CLS_criterion(y_hat, batch_yy.squeeze(dim=-1)) * self.loss_weight
                
                score, acc = self.accuracy_function(batch_yy, y_hat)
                total_score.append(score)
                total_acc.append(acc)
                train_CLS_loss.append(cls_loss.item())
                    
                cls_loss.backward()
                model_optim.step()
                
                
            total_score = np.average(total_score)
            total_acc = np.average(total_acc)                    
                    
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_GKP_loss = np.average(train_GKP_loss)
            train_CLS_loss = np.average(train_CLS_loss)
            train_loss = train_GKP_loss + train_CLS_loss
                                                                                                                                                                                
            vali_loss, val_score, val_acc, vgkp, vcls  = self.vali(vali_data, vali_loader, semi_valid_loader, CLS_criterion, GKP_criterion)
            print("Epoch: {0} | Train Loss: {1:.4f}  Vali Loss: {2:.4f} Vali ACC: {3:.4f} ".format(epoch + 1, train_loss, vali_loss, val_acc))
            
            early_stopping(vali_loss, self.model, path)

            
            
            # result save train-val loss fine_tune_K
            #folder_path = './results/' + "GKP/" + setting + '/'
            #if not os.path.exists(folder_path):
            #    os.makedirs(folder_path)

            #f = open(folder_path + f"result.txt", 'a')
            #f.write(setting + "  \n")
            #f.write('epoch:{}, train_loss:{}, vali_loss:{}, itr : {}'.format(epoch, train_loss, vali_loss, self.itr))
            #f.write('\n')
            #f.write('\n')
            #f.close()

            
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            scheduler.step()

        best_model_path = path + '/' + 'checkpoint.pth'            
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    


    def test_repre(self, setting, flag):
        
        test_data, test_loader,_,_   = self._get_data(flag=flag)
        
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        total_ = []
        batch_y_list = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_cont, batch_cat, batch_y) in enumerate(test_loader): 
                
                # Feed input to the model

                batch_cont = batch_cont.float().to(self.device)                
                batch_cat  = batch_cat.to(self.device)
                batch_y = batch_y.to(self.device)
                
                neg_feat, pos_feat, repre  = self.model(x_cont = batch_cont, x_cat = batch_cat, leaf = None, mode = "self")
                
                total_.append(repre)
                batch_y_list.append(batch_y)
        
     
        
        return total_, batch_y_list


    def test_repre_leaf(self, setting, flag):
        
        test_data, test_loader,_,_   = self._get_data(flag=flag)
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        total_ = []
        batch_y_list = []
        total_repre = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_leaves, batch_y) in enumerate(test_loader): 
                
                # Feed input to the model
                batch_leaves = batch_leaves.long().to(self.device)                
                batch_y = batch_y.to(self.device)
                
                logit,repre  = self.model(x_cont = None, x_cat = None, leaf = batch_leaves, mode = "semi")
                
                total_.append(logit)
                total_repre.append(repre)
                batch_y_list.append(batch_y)
        
     
        
        return total_, batch_y_list, total_repre
