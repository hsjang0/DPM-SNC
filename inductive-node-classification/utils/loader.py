import torch
import random
import numpy as np
from models.denoising_model import denoising_model
from method_series.gaussian_dpm_losses import gaussian_dpm_losses
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
    model = denoising_model(**params_)
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


def load_data(config):
    from utils.data_loader import dataloader
    return dataloader(config)


def load_batch(batch, device, data):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch.x.to(device_id).view(-1, batch.x.shape[-1]).float()
    adj_b = batch.edge_index.to(device_id).view(2, -1)
    y_b = batch.y.to(device_id)
    batch_b = batch.batch.to(device_id)
    maximum = y_b.max()

    if data == 'cora':
        y_b = F.one_hot(y_b.view(-1), 7).float()
        return x_b, adj_b, y_b, batch_b
    elif data == 'pubmed':
        y_b = F.one_hot(y_b.view(-1), 3).float()
        return x_b, adj_b, y_b, batch_b
    elif data == 'citeseer':
        y_b = F.one_hot(y_b.view(-1), 6).float()
        return x_b, adj_b, y_b, batch_b
    elif data.startswith('ppi'):
        y_b = y_b.view(-1, 121).float()
        return x_b, adj_b, y_b, torch.tensor([0])
    elif data == 'dblp':
        y_b = F.one_hot(y_b.view(-1), 3).float()
        return x_b, adj_b, y_b, torch.tensor([0])


def load_loss_fn(config, device):
    return gaussian_dpm_losses(config.diffusion.step, device = device, time_batch = config.train.time_batch, s = config.diffusion.s)


def load_model_params(config):
    config_m = config.model
    nlabel = config.data.nlabel
    params_ = {'model':config_m.model, 'num_linears': config_m.num_linears, 'nhid': config_m.nhid, 
                'nfeat': config.data.nfeat, 'skip':config_m.skip,
                'nlabel': nlabel,'num_layers':config_m.num_layers}
    return params_
