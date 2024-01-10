import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GCN2Conv, GINConv
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


def weight_init_xavier_uniform(submodule):
    torch.nn.init.xavier_normal_(submodule.weight)


def _one_hot(idx, num_class):
    return torch.zeros(len(idx), num_class).to(idx.device).scatter_(
        1, idx.unsqueeze(1), 1.)


class Simple_Model(torch.nn.Module):
    def __init__(self, model, nfeat, nlabel,num_layers, num_linears, nhid, nhead=8, alpha = 0):
        super(Simple_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.nhead = nhead
        self.layers = torch.nn.ModuleList()

        self.synthetic = (self.nfeat == 1)
        if self.synthetic:
            self.apply_dr = False
            self.improving = False
        else:
            self.apply_dr = True
            self.improving = True

        for i in range(self.depth):
            if i == 0:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nfeat, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(nn.Linear(self.nfeat, self.nhid))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(GCN2Conv(channels = self.nhid, alpha = 0.1, layer = i, theta = 0.5))
                elif self.model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid, out_channels =self.nhid, normalize = True))
                elif self.model == 'GATConv':
                    self.layers.append(GATConv(self.nhead*self.nhid, self.nhid, nhead, concat=True))

        self.activation = torch.nn.ReLU()

        if self.model == 'GATConv':
            self.nhid = self.nhid*self.nhead
            self.activation = torch.nn.ELU()


        self.fdim = self.nhid            
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel, 
                        use_bn=False, activate_func=F.elu, apply_dr = self.apply_dr)
        
        if self.apply_dr:
            self.dr = torch.nn.Dropout(p=0.5)
        else:
            self.dr = torch.nn.Dropout(p=0.0)

    def forward(self, x, adj):

        save_inital = None
        for i in range(self.depth):
            if self.model == 'GCN2Conv':
                if i == 0:
                    x_before_act = self.layers[i](x)
                else:
                    x_before_act = self.layers[i](x, save_inital, edge_index = adj)
            else:
                x_before_act = self.layers[i](x, edge_index = adj)

            x = self.activation(x_before_act)
            x = self.dr(x)

            if i == 0:
                save_inital = x

        pred_y = self.final(x)
        return F.log_softmax(pred_y, dim=1)


class Denoising_Model(torch.nn.Module):
    def __init__(self, model, nlabel, nfeat, num_layers, num_linears, nhid, nhead=8):
        super(Denoising_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.nhead = nhead
        self.layers = torch.nn.ModuleList()

        self.synthetic = (self.nfeat == 1)
        if self.synthetic:
            self.apply_dr = False
            self.improving = False
        else:
            self.apply_dr = True
            self.improving = True

        for i in range(self.depth):
            if i == 0:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(nn.Linear(self.nfeat+self.nlabel, self.nhid))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid, improved = self.improving))
                elif self.model == 'GCN2Conv':
                    self.layers.append(GCN2Conv(channels = self.nhid, alpha = 0.1, layer = i, theta = 0.5))
                elif self.model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid))
                elif self.model == 'GATConv':
                    self.layers.append(GATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True))

        self.activation = torch.nn.ReLU()
        if self.model == 'GATConv':
            self.nhid = self.nhid*self.nhead
            self.activation = torch.nn.ELU()
        
        self.fdim = self.nhid + self.nlabel
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.nhid)
        )
            
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel, 
                        use_bn=False, activate_func=F.elu, apply_dr = self.apply_dr)
                        
        if self.apply_dr:
            self.dr = torch.nn.Dropout(p=0.5)
        else:
            self.dr = torch.nn.Dropout(p=0.0)

    def forward(self, x, q_Y_sample, adj, t, num_steps, train=False):
        t = SinusoidalPosEmb(t, num_steps, 128)
        t = self.time_mlp(t)
        x = torch.cat([x, q_Y_sample], dim = -1)
        
        for i in range(self.depth):

            if self.model == 'GCN2Conv':
                if i == 0:
                    x_before_act = self.layers[i](x)
                else:
                    x_before_act = self.layers[i](x, save_inital, edge_index = adj)
            else:
                x_before_act = self.layers[i](x, adj) +  t

            x = self.activation(x_before_act)
            if train:
                x = self.dr(x)

            if self.model != 'GCN2Conv':
                x = torch.cat([x, q_Y_sample], dim = -1)
            if i == 0:
                save_inital = x

        if self.model == 'GCN2Conv':
             x = torch.cat([x, q_Y_sample], dim = -1)
        pred_y = self.final(x)
        return pred_y




