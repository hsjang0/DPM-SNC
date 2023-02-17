import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
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

        # Prepare model, optimizer, and logger
        self.params = load_model_params(self.config)
        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.params, self.config.train, 
                                                                                        self.device)
        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)
        self.loss_fn = self.losses.loss_fn
        self.evaluator = self.losses.test
        self.monte_eval = self.losses.monte_test

        # Train the model
        for epoch in range(0, (self.config.train.num_epochs)):
            t_start = time.time()

            self.model.train()
            self.optimizer.zero_grad()
            for e, train_block in enumerate(self.train_loader):
                x, adj, y, _ = load_batch(train_block, self.device, self.config.data.data) 
                loss_subject = (x, adj, y)   
                loss = self.loss_fn(self.model, *loss_subject)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.config.train.lr_schedule:
                    self.scheduler.step()
                """
                if loss == None:
                    loss = self.loss_fn(self.model, *loss_subject)
                else:
                    loss = loss + self.loss_fn(self.model, *loss_subject)
                
                if (e+1)%optimize_interval == 0:
                    loss = loss/optimize_interval
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    loss = None
                    if self.config.train.lr_schedule:
                        self.scheduler.step()
                """

            # Evaluate performance
            if epoch % self.config.train.print_interval == 0:
                self.model.eval()

                # Evaluate performance on validation dataset
                result_valid, result_valid_f1 = [], []
                for _, valid_block in enumerate(self.valid_loader):   
                    x_valid, adj_valid, y_valid, _ = load_batch(valid_block, self.device, self.config.data.data) 
                    with torch.no_grad():
                        acc, f1 = self.evaluator(self.model, x_valid, adj_valid, y_valid, self.config.data.data, None)
                        result_valid.append(acc)
                        result_valid_f1.append(f1)

                # Evaluate performance on test dataset
                result_test, result_test_f1 = [], []
                for _, test_block in enumerate(self.test_loader):   
                    x_test, adj_test, y_test, _ = load_batch(test_block, self.device, self.config.data.data) 
                    mask = torch.zeros((y.shape[0])).bool().to(x.device)
                    with torch.no_grad():
                        acc, f1 = self.evaluator(self.model, x_test, adj_test, y_test, self.config.data.data, None)
                        result_test.append(acc)
                        result_test_f1.append(f1)
                            
                # Log intermediate performance
                logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                            f'val acc: {np.mean(result_valid):.3e} | test acc: {np.mean(result_test):.3e} | ' 
                            f'val f1: {np.mean(result_valid_f1):.3e} | test f1: {np.mean(result_test_f1):.3e} ', verbose=False)
                tqdm.write(f'[EPOCH {epoch+1:04d}] | val acc: {np.mean(result_valid):.3e} | '
                            f'test acc: {np.mean(result_test):.3e} | val f1: {np.mean(result_valid_f1):.3e} | test f1: {np.mean(result_test_f1):.3e} ')


                
        print(' ')
