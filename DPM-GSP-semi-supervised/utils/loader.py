import torch
import random
import numpy as np
from models.model import Denoising_Model, Simple_Model
from method_series.gaussian_ddpm_losses import gaussian_ddpm_losses
from method_series.gaussian_ddpm_losses import simple_losses
import torch.nn.functional as F


def load_seed(seed):
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


def load_model(params):
    params_ = params.copy()
    model = Denoising_Model(**params_)   
    return model


def load_simple_model(params):
    params_ = params.copy()
    model = Simple_Model(**params_)   
    return model



def load_model_optimizer(params, config_train, device):
    model = load_model(params)
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


def load_simple_model_optimizer(params, config_train, device):
    
    model = load_simple_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr_simple, 
                                    weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_data(config):
    from utils.data_loader import dataloader
    return dataloader(config)


def load_loss_fn(config, device):
    if config.diffusion.method == 'Continuous':
        return gaussian_ddpm_losses(config.diffusion.step, device = device)


def load_simple_loss_fn(config, device):
    return simple_losses(device = device)


def load_model_params(config):
    config_m = config.model
    nlabel = config.data.nlabel
    params_ = {'model':config_m.model, 'num_linears': config_m.num_linears, 'nhid': config_m.nhid, 
                'nfeat': config.data.nfeat, 'skip':config_m.skip,'nlabel': nlabel,'num_layers':config_m.num_layers}
    return params_


def load_simple_model_params(config):
    config_m = config.model
    nlabel = config.data.nlabel
    params_ = {'model':config_m.model, 'num_linears': config_m.num_linears, 'nhid': config_m.nhid, 
                'nfeat': config.data.nfeat, 'skip':config_m.skip,'nlabel': nlabel,'num_layers':config_m.num_layers}
    return params_