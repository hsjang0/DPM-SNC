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


class Simple_Model(torch.nn.Module):
    def __init__(self, model, nfeat, nlabel,num_layers, num_linears, nhid, nhead=4, skip=False):
        super(Simple_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        nhead = 8
        self.nhead = nhead
        self.skip = skip
        self.layers = torch.nn.ModuleList()

        for i in range(self.depth):
            if i == 0:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nfeat, out_channels =self.nhid, normalize = True, improved = True))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid, out_channels =self.nhid, normalize = True, improved = True))
                elif self.model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid, out_channels =self.nhid, normalize = True))
                elif self.model == 'GATConv':
                    self.layers.append(GATConv(self.nhead*self.nhid, self.nhid, nhead, concat=True))

        if self.model == 'GATConv':
            self.nhid = self.nhid*self.nhead
            self.activation = torch.nn.ELU()

        self.fdim = self.nhid
        if self.skip:
            self.lin_list = torch.nn.ModuleList()
            for i in range(0,self.depth):
                if i == 0:
                    self.lin_list.append(nn.Linear(self.nfeat, self.nhid))
                else:
                    self.lin_list.append(nn.Linear(self.nhid, self.nhid))
            
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel, 
                        use_bn=False, activate_func=F.elu)
        self.dr = torch.nn.Dropout(p=0.5)

    def forward(self, x, adj):
        
        for i in range(self.depth):
            x_before_act = self.layers[i](x, adj)
            if self.skip:
                x_before_act = x_before_act + self.lin_list[i](x)
            x = self.activation(x_before_act)
            x = self.dr(x)

        pred_y = self.final(x)
        return F.log_softmax(pred_y, dim=1)


class Denoising_Model(torch.nn.Module):
    def __init__(self, model, nlabel, nfeat, num_layers, num_linears, nhid, nhead=4, skip=False):
        super(Denoising_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        nhead = 8
        self.nhead = nhead
        self.skip = skip
        self.layers = torch.nn.ModuleList()

        for i in range(self.depth):
            if i == 0:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid, normalize = True, improved = True))
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat+self.nlabel, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat+self.nlabel, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid, normalize = True, improved = True))
                elif self.model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nhid+self.nlabel, out_channels =self.nhid, normalize = True))
                elif self.model == 'GATConv':
                    self.layers.append(GATConv(self.nhead*self.nhid+self.nlabel, self.nhid, nhead, concat=True))

        if self.model == 'GATConv':
            self.nhid = self.nhid*self.nhead
            self.activation = torch.nn.ELU()

        self.fdim = self.nhid + self.nlabel
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.nhid)
        )
            
        if self.skip:
            self.lin_list = torch.nn.ModuleList()
            for i in range(0,self.depth):
                if i == 0:
                    self.lin_list.append(nn.Linear(self.nfeat+self.nlabel, self.nhid))
                else:
                    self.lin_list.append(nn.Linear(self.nhid+self.nlabel, self.nhid))
            
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel, 
                        use_bn=False, activate_func=F.elu)
        self.dr = torch.nn.Dropout(p=0.5)

    def forward(self, x, q_Y_sample, adj, t, num_steps, train=False):
        t = SinusoidalPosEmb(t, num_steps, 128)
        t = self.time_mlp(t)
        x = torch.cat([x, q_Y_sample], dim = -1)
        
        for i in range(self.depth):
            x_before_act = self.layers[i](x, adj) +  t
            if self.skip:
                x_before_act = x_before_act + self.lin_list[i](x)
            x = self.activation(x_before_act)
            if train:
                x = self.dr(x)
            x = torch.cat([x, q_Y_sample], dim = -1)

        pred_y = self.final(x)
        return pred_y




