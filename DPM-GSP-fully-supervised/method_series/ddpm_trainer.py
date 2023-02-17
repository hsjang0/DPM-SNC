import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, load_batch, load_loss_fn
from utils.logger import Logger, set_log, start_log, train_log

def prepare_training_dataset(train_loader, data, time_batch, device):
    x_list, adj_list, y_list, batch_list = [], [], [], []
    for e, train_block in enumerate(train_loader):
        x, adj, y, batch = load_batch(train_block, device, data) 
        x_list.append(x)
        adj_list.append(adj+x.shape[0]*e)
        y_list.append(y)
        batch_list.append(batch)
    x = torch.cat(x_list, dim = 0)
    adj = torch.cat(adj_list, dim = 1)
    y = torch.cat(y_list, dim = 0)
    batch = torch.cat(batch_list, dim = 0)

    x_list, adj_list = [], []
    for i in range(0, time_batch):
        x_list.append(x)
        adj_list.append(adj+x.shape[0]*i)
    x = torch.cat(x_list, dim = 0)
    adj = torch.cat(adj_list, dim = 1)
    return x, adj, y, batch

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

        # Prepare training dataset
        x, adj, y, batch = prepare_training_dataset(self.train_loader, self.config.data.data, self.config.train.time_batch, self.device)
        loss_subject = (x, adj, y)

        # Train the model
        for epoch in range(0, self.config.train.num_epochs):
            t_start = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model, *loss_subject)
            loss.backward()
            self.optimizer.step()
            if self.config.train.lr_schedule:
                self.scheduler.step()

            # Evaluate performance
            if epoch % self.config.train.print_interval == 0:
                self.model.eval() 

                # Evaluate performance on validation dataset
                result_save_train = []
                with torch.no_grad():
                    acc, graph_acc = self.evaluator(
                        self.model, x[:x.shape[0]//self.config.train.time_batch], adj[:,:adj.shape[1]//self.config.train.time_batch], y, self.config.data.data, batch
                        )
                    result_save_train.append(acc)

                # Evaluate performance on validation dataset
                result_valid, result_valid_graph = [], []
                for _, valid_block in enumerate(self.valid_loader):   
                    x_valid, adj_valid, y_valid, batch_valid = load_batch(valid_block, self.device, self.config.data.data) 
                    with torch.no_grad():
                        acc, graph_acc = self.evaluator(self.model, x_valid, adj_valid, y_valid, self.config.data.data, batch_valid)
                        result_valid.append(acc)
                        result_valid_graph.append(graph_acc)
                    
                # Evaluate performance on test dataset
                result_test, result_test_graph = [], []
                for _, test_block in enumerate(self.test_loader):   
                    x_test, adj_test, y_test, batch_test = load_batch(test_block, self.device, self.config.data.data) 
                    with torch.no_grad():
                        acc, graph_acc = self.evaluator(self.model, x_test, adj_test, y_test, self.config.data.data, batch_test)
                        result_test.append(acc)
                        result_test_graph.append(graph_acc)

                # Log intermediate performance
                logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                            f'train acc: {np.mean(result_save_train):.3e} | val acc: {np.mean(result_valid):.3e} | val graph: {np.mean(result_valid_graph):.3e} | '
                            f'test node: {np.mean(result_test):.3e} | test graph: {np.mean(result_test_graph):.3e}', verbose=False)
                                
                tqdm.write(f'[EPOCH {epoch+1:04d}] | train acc: {np.mean(result_save_train):.3e} | val acc: {np.mean(result_valid):.3e} | val graph: {np.mean(result_valid_graph):.3e} | '
                            f'test node: {np.mean(result_test):.3e} | test graph: {np.mean(result_test_graph):.3e}')
                
        print(' ')
