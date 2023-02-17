import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable
import time
from sklearn.metrics import f1_score
from torch_scatter import scatter_mean

def sum_except_batch(x, num_dims=1):
    return torch.sum(x, dim = -1)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s = 0.015):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    timesteps = (
        torch.arange(timesteps + 1, dtype=torch.float64) / timesteps + s
    )
    alphas = timesteps / (1 + s) * math.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = betas.clamp(max=0.999)
    betas = torch.cat(
            (torch.tensor([0], dtype=torch.float64), betas), 0
        )
    betas = betas.clamp(min=0.001)
    return betas


def get_accuracy(data, updated_y, y, batch = None):
    if data.startswith('ppi'):
        pred_label = (updated_y > 0.5).int()
        return torch.mean((pred_label==y).float()).item(), f1_score(y.cpu().detach().numpy(), pred_label.cpu().detach().numpy(), average='micro')
    if data == 'dblp' or batch == None:
        updated_y = torch.argmax(updated_y, dim = -1)
        y = torch.argmax(y, dim = -1)
        return torch.mean((updated_y==y).float()).item(), 0
    else:
        updated_y = torch.argmax(updated_y, dim = -1)
        y = torch.argmax(y, dim = -1)
        graph_acc = scatter_mean((updated_y==y).float(), index=batch, dim=0, dim_size=batch.max().item() + 1)
        graph_acc = (graph_acc == 1.).float().mean().item()
        return torch.mean((updated_y==y).float()).item(), graph_acc


class diffusion_model(torch.nn.Module):
    def __init__(self, device, timesteps, s):
        super(diffusion_model, self).__init__()

        betas = cosine_beta_schedule(timesteps, s)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas
        self.register("posterior_variance", posterior_variance.to(device[0]))
        self.register("betas", betas.to(device[0]))
        self.register("alphas", alphas.to(device[0]))
        self.register("alphas_cumprod", alphas_cumprod.to(device[0]))
        self.register("sqrt_alphas", torch.sqrt(alphas).to(device[0]))
        self.register("alphas_cumprod_prev", alphas_cumprod_prev.to(device[0]))
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device[0]))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).to(device[0]))
        self.register("thresh", (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod)
        self.num_timesteps = timesteps
        self.device = device

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x, t):
        noise = torch.randn_like(x)
        return (
            self.sqrt_alphas_cumprod[t] * x
            + self.sqrt_one_minus_alphas_cumprod[t] * noise, noise
        )

    def q_sample_inter(self, x, t, k):
        noise = torch.randn_like(x)
        var = torch.sqrt(1-self.alphas_cumprod[t+k]/self.alphas_cumprod[t])
        return (
            self.sqrt_alphas_cumprod[t+k] / self.sqrt_alphas_cumprod[t] * x
            + var * noise
        )

    def q_posterior(self, x_0, x_t, t):
        mean = (
            self.posterior_mean_coef1[t] * x_0
            + self.posterior_mean_coef2[t] * x_t
        )
        var = self.posterior_variance[t]*torch.ones_like(x_0)
        log_var_clipped = self.posterior_log_variance_clipped[t]*torch.ones_like(x_0)

        return mean, var, log_var_clipped


class gaussian_ddpm_losses:
    def __init__(self, num_timesteps, device, unweighted_MSE, time_batch = 1, s = 0.008):
        self.diff_Y = diffusion_model(device=device, timesteps = num_timesteps, s = s)
        self.num_timesteps = num_timesteps
        self.device = device
        self.time_batch = time_batch
        self.unweighted_MSE = unweighted_MSE
            
    def loss_fn(self,model, x, adj, y):
        losses = None
        t_list = []        
        y_sample_list = []
        epsilon_list = []
        for i in range(0, self.time_batch):
            t_list.append(self.sample_time(self.device))
            y_sample_temp, epsilon_temp = self.diff_Y.q_sample(y, t_list[-1].item())
            y_sample_list.append(y_sample_temp)
            epsilon_list.append(epsilon_temp)
        t_cat = torch.cat(t_list,dim = 0).view(-1,1)
        t_cat = t_cat.expand(-1,x.shape[0]//self.time_batch)
        t_cat = t_cat.reshape(-1)
        q_Y_sample = torch.cat(y_sample_list, dim = 0)
        epsilon_list = torch.cat(epsilon_list, dim =0)
        orig_shapes = y.shape[0]
        pred_y = model(x, q_Y_sample, adj, t_cat, self.num_timesteps)

        for e, t in enumerate(t_list):    
            if self.unweighted_MSE:
                coef = 1
            else:
                if t == 1:
                    coef = 0.5/self.diff_Y.alphas[1]
                else:
                    coef = 0.5*((self.diff_Y.betas[t]**2)/(self.diff_Y.posterior_variance[t]*self.diff_Y.alphas[t]*(1-self.diff_Y.alphas_cumprod[t-1])))
            if losses == None:
                losses = coef*torch.mean(
                    torch.sum(((pred_y[orig_shapes*e:orig_shapes*(e+1)]-epsilon_list[orig_shapes*e:orig_shapes*(e+1)])**2), dim = 1))
            else:
                losses = losses + coef*torch.mean(
                    torch.sum(((pred_y[orig_shapes*e:orig_shapes*(e+1)]-epsilon_list[orig_shapes*e:orig_shapes*(e+1)])**2), dim = 1))
        return losses/self.time_batch

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device[0]).long()
        return t

    def test(self, model, x, adj, y, data, batch, noise_temp = 0.001):
        updated_y = torch.randn_like(y)*noise_temp
        for i in range(0, self.diff_Y.num_timesteps):
            eps = model(x, updated_y, adj, torch.tensor([self.diff_Y.num_timesteps-i]).to(x.device).expand(x.shape[0]), self.diff_Y.num_timesteps)
            updated_y = (1/self.diff_Y.sqrt_alphas[self.diff_Y.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps)
            updated_y = updated_y + torch.sqrt(self.diff_Y.posterior_variance[self.diff_Y.num_timesteps-i])*torch.randn_like(eps)*noise_temp
        acc, gacc = get_accuracy(data, updated_y, y, batch)
        return acc, gacc

    def monte_test(self, model, x, adj, y, data, batch, noise_temp = 0.001):
        labels_set = torch.zeros(y.shape).float().to(x.device)
        acc = []
        gacc = []
        n_samples = [30,60,125,250,500,1000]
        for k in range(0, 1000):
            updated_y = torch.randn_like(y)*noise_temp
            for i in range(0, self.diff_Y.num_timesteps):
                eps = model(x, updated_y, adj, torch.tensor([self.diff_Y.num_timesteps-i]).to(x.device).expand(x.shape[0]), self.diff_Y.num_timesteps)
                updated_y = (1/self.diff_Y.sqrt_alphas[self.diff_Y.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps)
                updated_y = updated_y + torch.sqrt(self.diff_Y.posterior_variance[self.diff_Y.num_timesteps-i])*torch.randn_like(eps)*noise_temp
            labels = F.one_hot(torch.argmax(updated_y, dim = 1), updated_y.shape[1]).float()
            labels_set = labels_set + labels

            if (k+1) in n_samples:
                acc_, gacc_ = get_accuracy(data, labels_set, y, batch)
                acc.append(acc_)
                gacc.append(gacc_)
        return acc, gacc
        