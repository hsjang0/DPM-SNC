import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import neptune.new as neptune
from utils.loader import load_seed, load_device, load_data, load_model_optimizer, \
                         load_batch, load_loss_fn
from utils.logger import Logger, set_log, start_log, train_log


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.valid_loader, self.test_loader = load_data(self.config)
        self.losses = load_loss_fn(self.config, self.device)

    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.config.train, self.device)
        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)
        self.loss_fn = self.losses.loss_fn
        self.evaluator = self.losses.test

        epoch = 0
        while True:
            for train_block in self.train_loader:
                t_start = time.time()
                x, adj, adj_f, adj_l, batch = load_batch(train_block, self.device, self.config.data.data) 
                loss_subject = (x, adj, adj_f, adj_l, batch)

                self.model.train()
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model, epoch, *loss_subject)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
                self.optimizer.step()
                if self.config.train.lr_schedule:
                    self.scheduler.step()

                if (epoch+1) % self.config.train.print_interval == 0:
                    self.model.eval()

                    test_MSE = []
                    self.model.eval()
                    for e, test_block in enumerate(self.valid_loader):   
                        x, adj, adj_f, adj_l, batch = load_batch(test_block, self.device, self.config.data.data) 
                        with torch.no_grad():
                            test_MSE.append(self.evaluator(self.model, x, adj, adj_f, adj_l, batch, self.config.diffusion.temp))
                        if e*x.shape[0] >= 1024:
                            break
                            
                    test_MSE_huge = []
                    self.model.eval()
                    for e, test_block in enumerate(self.test_loader):   
                        x, adj, adj_f, adj_l, batch = load_batch(test_block, self.device, self.config.data.data) 
                        with torch.no_grad():
                            test_MSE_huge.append(self.evaluator(self.model, x, adj, adj_f, adj_l, batch, self.config.diffusion.temp))
                        if e*x.shape[0] >= 1024:
                            break
                                
                    logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                                f'val acc: {np.mean(test_MSE):.3e} | test acc: {np.mean(test_MSE_huge):.3e} ', verbose=False)

                    tqdm.write(f'[EPOCH {epoch+1:04d}] | val acc: {np.mean(test_MSE):.3e} | '
                                f'test acc: {np.mean(test_MSE_huge):.3e}')

                epoch = epoch+1
                if epoch > 10000000:
                    break
            
        print(' ')
        return self.ckpt
