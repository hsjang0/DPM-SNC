import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable


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
    betas = betas.clamp(max=0.99)
    betas = torch.cat(
            (torch.tensor([0], dtype=torch.float64), betas), 0
        )
    betas = betas.clamp(min=0.01)
    return betas


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
        self.register("betas", betas.to(device[0]))
        self.register("alphas", alphas.to(device[0]))
        self.register("alphas_cumprod", alphas_cumprod.to(device[0]))
        self.register("sqrt_alphas", torch.sqrt(alphas).to(device[0]))
        self.register("alphas_cumprod_prev", alphas_cumprod_prev.to(device[0]))
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device[0]))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).to(device[0]))
        self.register("thresh", (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod)
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod).to(device[0]))
        self.register("posterior_variance", posterior_variance.to(device[0]))
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

    def p_simul(self, model, x, adj, adj_f, adj_l, batch, noise):
        
        real = torch.randn_like(adj_l)*noise
        for i in range(0, self.num_timesteps):
            eps = model(x, adj_f, real, adj, torch.tensor([self.num_timesteps-i]).to(x.device), self.num_timesteps, batch)
            real = (1/self.sqrt_alphas[self.num_timesteps-i])*(real- (self.thresh[self.num_timesteps-i])*eps)
            if i == self.num_timesteps-1:
                real = real
            else:
                real = real + torch.sqrt(self.posterior_variance[self.num_timesteps-i])*torch.randn_like(real)*noise
        return torch.mean(torch.pow(real-adj_l,2)).cpu().detach().numpy()
            

class gaussian_ddpm_losses:
    def __init__(self, num_timesteps, device, t_batch = 1, s = 0.008):
        self.diff_Y = diffusion_model(device=device, timesteps = num_timesteps, s = s)
        self.num_timesteps = num_timesteps
        self.device = device
        self.t_batch = t_batch
        self.scheduled = True
        if s == 0.0101:
            self.scheduled = False
            
    def loss_fn(self,model, epoch, x, adj, adj_f, adj_l, batch):
        
        losses = None
        
        for i in range(0, self.t_batch):
            t = self.sample_time(x.device)
            q_Y_sample, noise = self.diff_Y.q_sample(adj_l, t.item())
            pred_y = model(x, adj_f, q_Y_sample, adj, t, self.num_timesteps, batch)

            if t == 1:
                coef = 0.5/self.diff_Y.alphas[1]
            else:
                coef = 0.5*((self.diff_Y.betas[t]**2)/(self.diff_Y.posterior_variance[t]*self.diff_Y.alphas[t]*(1-self.diff_Y.alphas_cumprod[t-1])))

            if losses == None:
                losses = torch.mean(
                            torch.sum(((pred_y-noise)**2), dim = 1)
                        )
            else:
                losses = losses + torch.mean(
                            torch.sum(((pred_y-noise)**2), dim = 1)
                        )
                
        return losses/self.t_batch

    def test(self, model, x, adj, adj_f, adj_l, batch, noise):
        return self.diff_Y.p_simul(model, x, adj, adj_f, adj_l, batch, noise)

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device).long()
        return t

