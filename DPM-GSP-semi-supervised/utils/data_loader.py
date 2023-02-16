from argparse import Namespace
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from torch_geometric.datasets import PPI, Amazon, Planetoid, WebKB
from torch_geometric.data import (InMemoryDataset, Data, DataLoader)
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import random

def dataloader(config):
    device = list(range(torch.cuda.device_count()))
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device

    if config.data.data in ['cora', 'citeseer', 'pubmed']:
        data = Planetoid(root = 'data', name = config.data.data)[0]
        nlabel = config.data.nlabel
        y = F.one_hot(data.y, nlabel).float()
        data.x[data.x>0] = 1
        return (data.x.to(device_id), y.to(device_id), data.edge_index.to(device_id), 
                data.train_mask.to(device_id), data.val_mask.to(device_id), data.test_mask.to(device_id))

    if config.data.data in ['Cornell', 'Texas', 'Wisconsin']:
        data = WebKB(root = 'data', name = config.data.data)[0]
        nlabel = config.data.nlabel
        y = F.one_hot(data.y, nlabel).float()
        return (data.x.to(device_id), y.to(device_id), data.edge_index.to(device_id), 
                data.train_mask[:,config.data.fold].to(device_id), data.val_mask[:,config.data.fold].to(device_id), data.test_mask[:,config.data.fold].to(device_id))

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
