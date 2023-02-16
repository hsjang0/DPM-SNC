import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import torch.nn.functional as F
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, load_loss_fn, \
                         load_simple_model_params, load_simple_model_optimizer, load_simple_loss_fn
from utils.logger import Logger, set_log, start_log, train_log


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.x, self.y, self.adj, self.train_mask, self.valid_mask, self.test_mask = load_data(self.config)
        self.losses = load_loss_fn(self.config, self.device)
        self.simple_losses = load_simple_loss_fn(self.config, self.device)
        
    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # Prepare model, optimizer, and logger
        self.params = load_model_params(self.config)
        self.simple_params = load_simple_model_params(self.config)
        self.simple_model, self.simple_optimizer, self.simple_scheduler = load_simple_model_optimizer(self.simple_params, self.config.train, self.device)
        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.params, self.config.train, self.device)
        self.loss_fn = self.losses.loss_fn
        self.simple_loss_fn = self.simple_losses.loss_fn
        self.estimator = self.losses.estimate
        self.mc_estimator = self.losses.mc_estimate
        self.simple_estimator = self.simple_losses.estimate          

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        # Pre-train mean-field GNN
        best_valid, best_est = 0, None
        print('Pretrain mean-field GNN...')
        for i in range(0,self.config.train.pre_train_epochs):
            self.simple_model.train()
            self.simple_optimizer.zero_grad()
            
            loss_subject = (self.x, self.adj, self.y, self.train_mask)
            loss = self.simple_loss_fn(self.simple_model, *loss_subject)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.simple_model.parameters(), self.config.train.grad_norm)
            self.simple_optimizer.step()
            if self.config.train.lr_schedule:
                self.simple_scheduler.step()
            
            # Evaluate mean-field GNN
            if i%10 == 0:
                self.simple_model.eval()
                y_est = self.simple_estimator(self.simple_model, self.x, self.adj, self.y, self.train_mask)
                pred = torch.argmax(y_est, dim = -1)
                label = torch.argmax(self.y, dim = -1)
                valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                test_acc = torch.mean((pred==label)[self.test_mask].float()).item()
                
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_est = y_est
        print('Done!')

        # Prepare expectation step
        buffer, n_samples, buffer_size = None, 5, 50
        xs, adjs, ys, best_ests, masks = [], [], [], [], []
        for i in range(0,n_samples):
            xs.append(self.x)
            adjs.append(self.adj+self.x.shape[0]*i)
            ys.append(self.y)
            best_ests.append(best_est)
            masks.append(self.train_mask)
        xs, adjs, ys, masks = torch.cat(xs, dim = 0), torch.cat(adjs, dim = 1), torch.cat(ys, dim = 0), torch.cat(masks, dim = 0) # (n_samples*number of data, )
        best_prob = torch.exp(torch.cat(best_ests, dim = 0))

        # Train the model
        for epoch in range(0, self.config.train.num_epochs):
            t_start = time.time()

            # Expectation step
            if epoch % self.config.train.load_interval == 0:
                if epoch > self.config.train.load_start: # Use manifold-constarined sampling of DPM-GSP               
                    expected_y_set = self.mc_estimator(self.model, xs, adjs, ys, masks, temp = self.config.diffusion.temp, coef = self.config.diffusion.coef)
                else: # Use mean-field GNN
                    expected_y_set = torch.distributions.categorical.Categorical(best_prob).sample()
                    expected_y_set = F.one_hot(expected_y_set, best_prob.shape[1]).float()
                
                # Fill the buffer
                expected_y_set = torch.cat(
                        [expected_y_set[i*self.y.shape[0]:(i+1)*self.y.shape[0]].view(1,self.y.shape[0],-1) for i in range(0,n_samples)], dim = 0) # (n_samples, number of data, number of classes)
                if buffer == None:
                    buffer = expected_y_set
                else:
                    buffer = torch.cat([buffer,expected_y_set], dim = 0)

                # Keep the buffer size
                start = buffer.shape[0]-buffer_size
                if start < 0:
                    start = 0
                buffer = buffer[start:]
    
            # Maximization step   
            y_train = buffer[np.random.randint(buffer.shape[0]+1)-1] # Sample from the buffer
            y_train[self.train_mask] = self.y[self.train_mask]   
    
            self.model.train()
            self.optimizer.zero_grad()
            loss_subject = (self.x, self.adj, y_train, self.train_mask, self.config.train.time_batch)
            loss = self.loss_fn(self.model, *loss_subject)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
            self.optimizer.step()
            if self.config.train.lr_schedule:
                self.scheduler.step()

            # Evaluate the model
            if epoch % self.config.train.print_interval == 0 and epoch > 0:
                
                # Manifold-constrained sampling
                y_est = self.mc_estimator(self.model, self.x, self.adj, self.y, self.train_mask, coef = self.config.diffusion.coef)
                pred, label = torch.argmax(y_est, dim = -1), torch.argmax(self.y, dim = -1)
                valid_acc = torch.mean((pred==label)[self.valid_mask].float()).item()
                test_acc = torch.mean((pred==label)[self.test_mask].float()).item()
                
                # N/A Manifold-constrained sampling
                with torch.no_grad():
                    y_est = self.estimator(self.model, self.x, self.adj, self.y, self.train_mask)
                pred = torch.argmax(y_est, dim = -1)
                train_acc = torch.mean((pred==label)[self.train_mask].float()).item()
                
                # Log intermediate performance
                logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                            f'train acc: {train_acc:.3e} | val acc: {valid_acc:.3e} | test acc: {test_acc:.3e}', verbose=False)         
                tqdm.write(f'[EPOCH {epoch+1:04d}] | train acc: {train_acc:.3e} | val acc: {valid_acc:.3e} | test acc: {test_acc:.3e}')
        
        print(' ')
