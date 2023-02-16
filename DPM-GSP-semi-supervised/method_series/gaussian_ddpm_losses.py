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
    betas = betas.clamp(max=0.999)
    betas = torch.cat(
            (torch.tensor([0], dtype=torch.float64), betas), 0
        )
    betas = betas.clamp(min=0.001)
    return betas


class diffusion_model(torch.nn.Module):
    def __init__(self, device, timesteps):
        super(diffusion_model, self).__init__()

        betas = cosine_beta_schedule(timesteps)
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


class gaussian_ddpm_losses:
    def __init__(self, num_timesteps, device):
        self.diff_Y = diffusion_model(device=device, timesteps = num_timesteps)
        self.num_timesteps = num_timesteps
        self.device = device

    # Loss function
    def loss_fn(self,model, x, adj, y, train_mask, batch = 1):
        losses = None

        for i in range(0, batch):
            t = self.sample_time(self.device)
            q_Y_sample, noise = self.diff_Y.q_sample(y, t)
            pred_y = model(x, q_Y_sample, adj, t, self.num_timesteps, train=True)
            
            # Compute losses for observed nodes
            if losses == None:
                losses = torch.mean(
                    torch.sum(((pred_y[train_mask]-noise[train_mask])**2), dim = -1)
                )
            else:
                losses = losses + torch.mean(
                    torch.sum(((pred_y[train_mask]-noise[train_mask])**2), dim = -1)
                )

            # Compute losses for unobserved nodes
            losses = losses + torch.mean(
                torch.sum(((pred_y[~train_mask]-noise[~train_mask])**2), dim = -1)
            )
                
        return losses/batch

    def estimate(self, model, x, adj, y, mask, temp=0.01):
        updated_y = torch.randn_like(y)*temp
        for i in range(0, self.num_timesteps-1):            
            eps = model(x, updated_y, adj, torch.tensor([self.num_timesteps-i]).to(x.device), self.num_timesteps)
            updated_y = (1/self.diff_Y.sqrt_alphas[self.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps)
        return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()

    # Manifold-constrained sampling
    def mc_estimate(self, model, x, adj, y, mask, temp = 0.01, coef = 1):
        updated_y = torch.randn_like(y)*temp
        for i in range(0, self.num_timesteps):
            updated_y = Variable(updated_y, requires_grad=True)

            # Compute y_prime
            eps = model(x, updated_y, adj, torch.tensor([self.num_timesteps-i]).to(x.device), self.num_timesteps)
            y_prime = (1/self.diff_Y.sqrt_alphas[self.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps)  
            y_prime = y_prime + temp*torch.sqrt(self.diff_Y.posterior_variance[self.num_timesteps-i])*torch.randn_like(y_prime)

            # Compute y_hat
            score = -(1/torch.sqrt(1-self.diff_Y.alphas_cumprod[self.num_timesteps-i]))*eps
            y_hat = (1/torch.sqrt(self.diff_Y.alphas_cumprod[self.num_timesteps-i]))*(updated_y+(1-self.diff_Y.alphas_cumprod[self.num_timesteps-i])*score)

            if self.num_timesteps-i > 1:  
                # Apply manifold-constrained gradient
                imp_loss = torch.sum(torch.sum(((y-y_hat)[mask])**2, dim=1))
                imp_loss.backward()
                alpha = coef/(torch.sum(torch.sum((y-y_hat)[mask]**2, dim = 1)))
                y_update = y_prime - alpha*updated_y.grad.data 

                # Apply consistency step
                y_update[mask] = self.diff_Y.alphas_cumprod[self.num_timesteps-i]*y[mask] + self.diff_Y.sqrt_one_minus_alphas_cumprod[self.num_timesteps-i] *temp* torch.randn_like(y[mask])
                updated_y = y_update
            else:
                updated_y = y_prime

        return F.one_hot(torch.argmax(updated_y, dim = -1), updated_y.shape[1]).float()

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device[0]).long()
        return t


class simple_losses:
    def __init__(self, device):
        self.device = device

    def loss_fn(self,model, x, adj, y, train_mask, batch = 1):
        pred_y = model(x, adj)
        losses = F.nll_loss(pred_y[train_mask], torch.argmax(y[train_mask], dim = -1))
        return losses

    def estimate(self,model, x, adj, y, train_mask):
        return model(x, adj)
    