from argparse import Namespace
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from torch_geometric.datasets import PPI, Amazon, Planetoid, WebKB#, HeterophilousGraphDataset
from torch_geometric.data import (InMemoryDataset, Data, DataLoader)
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
import random
import scipy.sparse as sp

def dataloader(config):
    device = list(range(torch.cuda.device_count()))
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    if config.data.data in ['cora', 'citeseer', 'pubmed']:
        data = Planetoid(root = 'data', name = config.data.data)[0]
        nlabel = config.data.nlabel
        y = F.one_hot(data.y, nlabel).float()
        data.x[data.x>0] = 1
        """
        features = np.array(data.x.cpu().detach().numpy())
        rowsum = np.array(features.sum(1))
        rowsum = (rowsum==0)*1+rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        data.x = torch.FloatTensor(features)
        """
        return (data.x.to(device_id), y.to(device_id), data.edge_index.to(device_id), 
                data.train_mask.to(device_id), data.val_mask.to(device_id), data.test_mask.to(device_id))

    if config.data.data in ['Cornell', 'Texas', 'Wisconsin']:
        data = WebKB(root = 'data', name = config.data.data)[0]
        nlabel = config.data.nlabel
        y = F.one_hot(data.y, nlabel).float()
        return (data.x.to(device_id), y.to(device_id), data.edge_index.to(device_id), 
                data.train_mask[:,config.data.fold].to(device_id), data.val_mask[:,config.data.fold].to(device_id), data.test_mask[:,config.data.fold].to(device_id))

    if config.data.data in ['Roman-empire', 'Amazon-ratings']:
        data = HeterophilousGraphDataset(root = 'data', name = config.data.data)[0]
        nlabel = config.data.nlabel
        y = F.one_hot(data.y, nlabel).float() 
        #print(data.train_mask.shape)
        return (data.x.to(device_id), y.to(device_id), data.edge_index.to(device_id), 
                data.train_mask[:,config.seed].to(device_id), data.val_mask[:,config.seed].to(device_id), data.test_mask[:,config.seed].to(device_id))


    if config.data.data in ['Photo', 'Computers']:
        data = Amazon(root = 'data', name = config.data.data)[0]
        nlabel = config.data.nlabel
        y = F.one_hot(data.y, nlabel).float()
        
        train_idx_list = []
        valid_idx_list = []
        for i in range(0, config.data.nlabel):
            temp_idx = torch.where(data.y==i)[0].tolist()
            np.random.shuffle(temp_idx)
            train_idx_list = train_idx_list + temp_idx[-20:]
            valid_idx_list = valid_idx_list + temp_idx[:30]            
        idx = list(range(0,data.x.shape[0]))
        left_idx_list = list(set(idx)-set(train_idx_list)-set(valid_idx_list))
        np.random.shuffle(left_idx_list)        
        
        test_idx = torch.BoolTensor(data.x.shape[0])
        test_idx[:] = False
        test_idx[left_idx_list] = True
        train_idx = torch.BoolTensor(data.x.shape[0])
        train_idx[:] = False
        train_idx[train_idx_list] = True
        val_idx = torch.BoolTensor(data.x.shape[0])
        val_idx[:] = False
        val_idx[valid_idx_list] = True
        nlabel = config.data.nlabel
        
        return (data.x.to(device_id), y.to(device_id), data.edge_index.to(device_id), 
                train_idx.to(device_id), val_idx.to(device_id), test_idx.to(device_id))

    if config.data.data in ['Synthetic-scattered']:

        nn = 1000
        x = torch.zeros((nn*2,1)).float()
        y = torch.zeros((nn*2,2)).float()
        for i in range(0,nn):
            y[i, i%2] = 1

        for i in range(nn,nn*2):
            y[i, 1-i%2] = 1

        edge_list = []
        for i in range(0,nn-1):
            edge_list.append([i,i+1]) 
            edge_list.append([nn+i,nn+i+1])
            edge_list.append([i,nn+i])
            edge_list.append([i+1,i]) 
            edge_list.append([nn+i+1,nn+i])
            edge_list.append([nn+i,i])
        edge_list.append([nn*2-1,nn-1])
        edge_list.append([nn-1,0])
        edge_list.append([nn*2-1,nn])
        edge_list.append([nn-1,nn*2-1])
        edge_list.append([0,nn-1])
        edge_list.append([nn,nn*2-1])
        edge_list = torch.tensor(edge_list).T

        nums = list(range(0, nn*2))
        random.shuffle(nums)    
        idx = list(range(0,x.shape[0]))
        train_idx_list = nums[:int(nn*2/10*3)]
        valid_idx_list = nums[int(nn*2/10*3):int(nn*2/10*3)+int(nn*2/10*3)]
        left_idx_list = list(set(idx)-set(train_idx_list)-set(valid_idx_list))        
        
        test_idx = torch.BoolTensor(x.shape[0]+40)
        test_idx[:] = False
        test_idx[left_idx_list] = True
        train_idx = torch.BoolTensor(x.shape[0])
        train_idx[:] = False
        train_idx[train_idx_list] = True
        val_idx = torch.BoolTensor(x.shape[0]+40)
        val_idx[:] = False
        val_idx[valid_idx_list] = True
        nlabel = config.data.nlabel




        nn = 20
        x_ = torch.zeros((nn*2,1)).float()
        y_ = torch.zeros((nn*2,2)).float()
        for i in range(0,nn):
            y_[i, i%2] = 1

        for i in range(nn,nn*2):
            y_[i, 1-i%2] = 1

        edge_list_ = []
        for i in range(0,nn-1):
            edge_list_.append([i,i+1]) 
            edge_list_.append([nn+i,nn+i+1])
            edge_list_.append([i,nn+i])
            edge_list_.append([i+1,i]) 
            edge_list_.append([nn+i+1,nn+i])
            edge_list_.append([nn+i,i])
        edge_list_.append([nn*2-1,nn-1])
        edge_list_.append([nn-1,0])
        edge_list_.append([nn*2-1,nn])
        edge_list_.append([nn-1,nn*2-1])
        edge_list_.append([0,nn-1])
        edge_list_.append([nn,nn*2-1])
        edge_list_ = torch.tensor(edge_list_).T + 2000

        nums = list(range(0, nn*2))
        random.shuffle(nums)    
        idx = list(range(0,x_.shape[0]))
        train_idx_list = [0, 27, 12, 21, 29, 15, 38, 34]#nums[:int(nn*2/10*2)]
        left_idx_list = torch.tensor(list(set(idx)-set(train_idx_list)))       
        
        add_test_idx = torch.BoolTensor((x.shape[0]+x_.shape[0]))
        add_test_idx[:] = False
        add_test_idx[x.shape[0]+left_idx_list] = True
        train_idx_ = torch.BoolTensor(x_.shape[0])
        train_idx_[:] = False
        train_idx_[train_idx_list] = True



        x = torch.cat([x,x_], dim = 0)
        y = torch.cat([y,y_], dim = 0)
        edge_list = torch.cat([edge_list,edge_list_], dim = 1)
        train_idx = torch.cat([train_idx,train_idx_], dim = 0)


        return (x.to(device_id), y.to(device_id), edge_list.to(device_id), 
                train_idx.to(device_id), val_idx.to(device_id), test_idx.to(device_id))

    if config.data.data in ['Synthetic-localized']:

        nn = 100
        x = torch.zeros((nn*2,1)).float()
        y = torch.zeros((nn*2,2)).float()
        for i in range(0,nn):
            y[i, i%2] = 1

        for i in range(nn,nn*2):
            y[i, 1-i%2] = 1

        edge_list = []
        for i in range(0,nn-1):
            edge_list.append([i,i+1]) 
            edge_list.append([nn+i,nn+i+1])
            edge_list.append([i,nn+i])
            edge_list.append([i+1,i]) 
            edge_list.append([nn+i+1,nn+i])
            edge_list.append([nn+i,i])
        edge_list.append([nn*2-1,nn-1])
        edge_list.append([nn-1,0])
        edge_list.append([nn*2-1,nn])
        edge_list.append([nn-1,nn*2-1])
        edge_list.append([0,nn-1])
        edge_list.append([nn,nn*2-1])
        edge_list = torch.tensor(edge_list).T


        idx = list(range(0,x.shape[0]))
        train_idx_list = [i for i in range(0,30)] + [(i+100) for i in range(0,30)] 
        idx_random = [i for i in range(30,100)] + [(i+100) for i in range(30,100)] 
        random.shuffle(idx_random)    
        valid_idx_list = idx_random[:60]
        left_idx_list = list(set(idx)-set(train_idx_list)-set(valid_idx_list))
        
        test_idx = torch.BoolTensor(x.shape[0])
        test_idx[:] = False
        test_idx[left_idx_list] = True
        train_idx = torch.BoolTensor(x.shape[0])
        train_idx[:] = False
        train_idx[train_idx_list] = True
        val_idx = torch.BoolTensor(x.shape[0])
        val_idx[:] = False
        val_idx[valid_idx_list] = True
        nlabel = config.data.nlabel
        
    return (x.to(device_id), y.to(device_id), edge_list.to(device_id), 
            train_idx.to(device_id), val_idx.to(device_id), test_idx.to(device_id))



