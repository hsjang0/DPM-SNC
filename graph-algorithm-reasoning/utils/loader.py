import torch
import random
import numpy as np

from models.denoising_model import Denoising_Model
from method_series.gaussian_dpm_losses import gaussian_dpm_losses
from torch_geometric.data import (InMemoryDataset, Data, DataLoader)
import torch.nn.functional as F

def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model():
    model = Denoising_Model()   
    return model


def load_model_optimizer(config_train, device):
    
    model = load_model()
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, 
                                    weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def worker_init_fn(worker_id):
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))


def load_data(config):
    if config.data.data == 'connected_comp':
        from utils.data_loader import connected_comp
        train_dataloader = DataLoader(connected_comp('train', 10, vary = True), num_workers=8, batch_size=64, shuffle=True, worker_init_fn=worker_init_fn)
        test_dataloader = DataLoader(connected_comp('test', 10), num_workers=8, batch_size=256, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
        gen_dataloader = DataLoader(connected_comp('test-2', 15), num_workers=8, batch_size=256, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
    elif config.data.data == 'shortest':
        from utils.data_loader import shortest
        train_dataloader = DataLoader(shortest('train', 10, vary = True), num_workers=8, batch_size=64, shuffle=True, worker_init_fn=worker_init_fn)
        test_dataloader = DataLoader(shortest('test', 10), num_workers=8, batch_size=256, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
        gen_dataloader = DataLoader(shortest('test-2', 15), num_workers=8, batch_size=256, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
    else:
        from utils.data_loader import identity
        train_dataloader = DataLoader(identity('train', 10, vary = True), num_workers=8, batch_size=64, shuffle=True, worker_init_fn=worker_init_fn)
        test_dataloader = DataLoader(identity('test', 10), num_workers=8, batch_size=256, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
        gen_dataloader = DataLoader(identity('test-2', 15), num_workers=8, batch_size=256, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
    return train_dataloader, test_dataloader, gen_dataloader


def load_batch(batch, device, data):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch.x.to(device_id)
    adj_b = batch.edge_index.to(device_id)
    adj_f = batch.edge_attr.to(device_id)
    adj_l = batch.y.to(device_id)
    return x_b, adj_b, adj_f, adj_l, batch.batch.to(device_id)

def load_loss_fn(config, device):
    return gaussian_dpm_losses(config.diffusion.step, device = device, t_batch = config.train.t_batch, s = config.diffusion.s,)

