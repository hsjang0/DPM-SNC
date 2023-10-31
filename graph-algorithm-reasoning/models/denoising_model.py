import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GCN2Conv, GINEConv, global_max_pool
from torch_geometric.utils import to_dense_adj
from models.layers import DenseGCNConv, MLP
import math

def SinusoidalPosEmb(x, num_steps, dim,rescale=4):
    x = x / num_steps * num_steps*rescale
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb



class Denoising_Model(nn.Module):
    def __init__(self):
        super(Denoising_Model, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h // 2)
        self.edge_map_opt = nn.Linear(1, h // 2)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Sequential(
            nn.Linear(257, 257),  
            nn.ReLU(),
            nn.Linear(257, 1)
        )

        self.y_map = nn.Linear(192,1)

        self.time_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x, edge_attr, q_Y_sample, adj, t, num_steps, batch, flags=None, train=False):

        edge_embed = self.edge_map(edge_attr)
        opt_edge_embed = self.edge_map_opt(q_Y_sample)

        t = SinusoidalPosEmb(t, num_steps, 128)
        t = t.reshape(1,-1).expand(x.shape[0],-1)  
        t = self.time_mlp(t)
        
        edge_embed = torch.cat([edge_embed, opt_edge_embed], dim=-1)
        h = self.conv1(x, adj, edge_attr=edge_embed)+t
        h = self.conv2(h, adj, edge_attr=edge_embed)+t
        h = self.conv3(h, adj, edge_attr=edge_embed)+t

        hidden = h
        h1 = hidden[adj[0]]
        h2 = hidden[adj[1]]
        h = torch.cat([h1, h2, q_Y_sample], dim=-1)
        output = self.decode(h)
        return output



