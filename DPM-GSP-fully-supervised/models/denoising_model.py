import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
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


class denoising_model(torch.nn.Module):
    def __init__(self, model, nlabel, nfeat, num_layers, num_linears, nhid, nhead=4, cat_mode = False, skip=False, types = 'continuous'):
        super(denoising_model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.cat_mode = cat_mode
        nhead = 6
        self.nhead = nhead
        self.skip = skip
        self.layers = torch.nn.ModuleList()
        self.time_mlp = nn.Sequential(
            nn.Linear(self.nhid, self.nhid * 2),
            nn.ELU(),
            nn.Linear(self.nhid * 2, self.nhid)
        )
        if self.cat_mode:
            self.time_mlp2 = nn.Linear(self.nhid, self.nhid*self.nhead)

        for i in range(self.depth):
            if i == 0:
                if model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid, normalize = True, improved = True))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    if self.cat_mode:
                        self.layers.append(GATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True))
                    else:
                        self.layers.append(GATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=False))
            else:
                if model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid, normalize = True, improved = True))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    if i==self.depth-1:
                        if self.cat_mode:
                            self.layers.append(GATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=False))
                        else:
                            self.layers.append(GATConv(self.nhid+self.nlabel, self.nhid, nhead, concat=False))        
                    else:
                        if self.cat_mode: 
                            self.layers.append(GATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True))
                        else:
                            self.layers.append(GATConv(self.nhid+self.nlabel, self.nhid, nhead, concat=False))

        self.fdim = self.nhid+self.nlabel
        if model == 'GATConv':
            self.activation = torch.nn.ELU()
        else:
            self.activation = torch.nn.ReLU()
        self.types = types
                
        if self.skip:
            self.lin_list = torch.nn.ModuleList()
            for i in range(0,self.depth):
                if self.cat_mode:
                    if i == 0:
                        self.lin_list.append(nn.Linear(self.nfeat+self.nlabel, self.nhid*self.nhead))
                    elif i == self.depth -1:
                        self.lin_list.append(nn.Linear(self.nhid*self.nhead+self.nlabel, self.nhid))
                    else:
                        self.lin_list.append(nn.Linear(self.nhid*self.nhead+self.nlabel, self.nhid*self.nhead))
                else:
                    if i == 0:
                        self.lin_list.append(nn.Linear(self.nfeat+self.nlabel, self.nhid))
                    else:
                        self.lin_list.append(nn.Linear(self.nhid+self.nlabel, self.nhid))
                    
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel, 
                        use_bn=False, activate_func=F.elu)


    def forward(self, x, q_Y_sample, adj, t, num_steps, flags=None):
        t = SinusoidalPosEmb(t, num_steps, self.nhid)
        t = self.time_mlp(t)
        x = torch.cat([x, q_Y_sample], dim = -1)
        
        x_list = [x]
        for i in range(self.depth):
            if self.cat_mode:
                if i == self.depth -1:
                    x_before_act = self.layers[i](x, adj) + t
                else:
                    x_before_act = self.layers[i](x, adj) + self.time_mlp2(t)
            else:
                x_before_act = self.layers[i](x, adj) + t
            if self.skip:
                x_before_act = x_before_act + self.lin_list[i](x)

            x = self.activation(x_before_act)
            x = torch.cat([x, q_Y_sample], dim = -1)

        updated_y = self.final(x).view(q_Y_sample.shape[0], -1)
        return updated_y

