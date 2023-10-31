import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GCN2Conv, GINConv
from torch_geometric.utils import to_dense_adj
from models.layers import DenseGCNConv, MLP, LPAConv
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


class LPA_Model(torch.nn.Module):
    def __init__(self, model, nfeat, nlabel,num_layers, num_linears, nhid, num_edges, lpaiters=10, nhead=8):
        super(LPA_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.nhead = nhead
        self.layers = torch.nn.ModuleList()

        self.edge_weight = nn.Parameter(torch.ones(num_edges))
        self.lpa = LPAConv(lpaiters)

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
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid, out_channels =self.nhid, improved = self.improving))
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

    def forward(self, x, y, adj, mask):

        save_inital = None
        for i in range(self.depth):
            x_before_act = self.layers[i](x, adj, self.edge_weight)

            x = self.activation(x_before_act)
            x = self.dr(x)

            if i == 0:
                save_inital = x

        pred_y = self.final(x)
        y_hat = self.lpa(y, adj, mask, self.edge_weight)
        return F.log_softmax(pred_y, dim=1), F.log_softmax(y_hat, dim=1)


class P_Model(torch.nn.Module):
    def __init__(self, model, nfeat, nlabel,num_layers, num_linears, nhid):
        super(P_Model, self).__init__()

        self.nfeat = nfeat
        self.depth = num_layers
        self.nhid = nhid
        self.nlabel = nlabel
        self.model = model
        self.layers = torch.nn.ModuleList()

        for i in range(self.depth):
            if i == 0:
                self.layers.append(GCNConv(in_channels = self.nfeat, out_channels =self.nhid))
            else:
                self.layers.append(GCNConv(in_channels = self.nhid, out_channels =self.nhid))

        self.activation = torch.nn.ReLU()
        self.fdim = self.nhid            
        self.final = MLP(num_layers=num_linears, input_dim=self.fdim, hidden_dim=self.fdim*2, output_dim=self.nlabel, 
                        use_bn=False, activate_func=F.elu)

        if self.nfeat == 1:
            self.dr = torch.nn.Dropout(p=0.0)
        else:
            self.dr = torch.nn.Dropout(p=0.5)

    def forward(self, x, adj):

        for i in range(self.depth):
            x_before_act = self.layers[i](x, edge_index = adj)
            x = self.activation(x_before_act)
            x = self.dr(x)

        pred_y = self.final(x)
        return F.log_softmax(pred_y, dim=1)


class CLGNN_Model(torch.nn.Module):
    def __init__(self, model, nfeat, nlabel,num_layers, num_linears, nhid, nhead=8, alpha = 0):
        super(CLGNN_Model, self).__init__()

        self.nfeat = nfeat+nlabel
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
                elif model == 'SAGEConv':
                    self.layers.append(SAGEConv(in_channels = self.nfeat, out_channels =self.nhid, normalize = True))
                elif model == 'GATConv':
                    self.layers.append(GATConv(self.nfeat, self.nhid, nhead, concat=True))
            else:
                if self.model == 'GCNConv':
                    self.layers.append(GCNConv(in_channels = self.nhid, out_channels =self.nhid, improved = self.improving))
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

    def pad_zeros(self, feat):
        feat_new = torch.zeros(feat.shape[0], self.nlabel)
        feat_new[:, :feat.shape[1]] = feat
        return feat_new


    def forward(self, x, y, predictions, adj, idx_labeled, n_sample, pre_train=True):

        if pre_train:
            feats = torch.zeros(x.shape[0], ).long().to(x.device)
            feats += self.nlabel + 2
            feats[idx_labeled] = y[idx_labeled]
            feats = F.one_hot(feats)[:, :self.nlabel]
            feats = feats.unsqueeze(0)
        else:
            feats = predictions.unsqueeze(0)
            true_feat = F.one_hot(y[idx_labeled])
            if true_feat.shape[1] < self.nlabel:
                true_feat = self.pad_zeros(true_feat)
            feats[:, idx_labeled, :] = true_feat.float()
            
            feats = torch.stack([torch.distributions.categorical.Categorical(predictions).sample() for i in range(n_sample)], dim = 0)
            for i in range(n_sample):
                feats[i, idx_labeled] = y[idx_labeled]
            feats = torch.stack([self.pad_zeros(F.one_hot(feats[i])) for i in range(n_sample)], dim=0)

        feats = feats.to(x.device)
        if pre_train:
            rand_nums = 1
        else:
            rand_nums = n_sample
        
        x_concat = torch.stack([x for i in range(rand_nums)], dim=0)
        x_concat = torch.cat([x_concat, feats.float()], dim=2)

        for i in range(self.depth):
            x_concat = [self.activation(self.layers[i](x_i, adj)) for x_i in x_concat]
            x_concat = [self.dr(x_i) for x_i in x_concat]

        x_concat = sum(x_concat) / len(x_concat)
        x_concat = self.final(x_concat)
        return F.log_softmax(x_concat, dim=1)


class G3NN_Model(torch.nn.Module):
    def __init__(self, nfeat, nlabel, nhid, hid_x=2):
        super(G3NN_Model, self).__init__()
        self.p_y_x = MLP(num_layers=2, input_dim=nfeat, hidden_dim=nhid, output_dim=nlabel, 
                        use_bn=False, activate_func=F.relu)
        
        self.x_enc = torch.nn.Linear(nfeat, hid_x)
        self.p_e_xy = torch.nn.Linear(2 * (hid_x + nlabel), 1)

        self.dropout = 0.5
        self.activation = F.relu
        self.neg_ratio = 1.0

    def forward(self, x, y, adj, train_mask):
        y_log_prob = F.log_softmax(self.p_y_x(x), dim = 1)
        y_prob = torch.exp(y_log_prob)
        y_prob = torch.where(
            train_mask.unsqueeze(1), _one_hot(y, y_prob.size(1)),
            y_prob)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.x_enc(x))

        # Positive edges.
        y_query = F.embedding(adj[0], y_prob)
        y_key = F.embedding(adj[1], y_prob)
        x_query = F.embedding(adj[0], x)
        x_key = F.embedding(adj[1], x)
        xy = torch.cat([x_query, x_key, y_query, y_key], dim=1)
        e_pred_pos = self.p_e_xy(xy)

        # Negative edges.
        e_pred_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = adj.size(1)
            num_nodes = x.size(0)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = torch.randint(num_nodes,
                                           (2, num_edges_neg)).to(x.device)
            y_query = F.embedding(edge_index_neg[0], y_prob)
            y_key = F.embedding(edge_index_neg[1], y_prob)
            x_query = F.embedding(edge_index_neg[0], x)
            x_key = F.embedding(edge_index_neg[1], x)
            xy = torch.cat([x_query, x_key, y_query, y_key], dim=1)
            e_pred_neg = self.p_e_xy(xy)

        return e_pred_pos, e_pred_neg, y_log_prob

    def nll_generative(self, x, y, adj, train_mask, post_y_log_prob):
        e_pred_pos, e_pred_neg, y_log_prob = self.forward(x, y, adj, train_mask)
        # unlabel_mask = data.val_mask + data.test_mask
        unlabel_mask = torch.ones_like(train_mask) ^ train_mask

        # nll of p_g_xy
        nll_p_g_xy = -torch.mean(F.logsigmoid(e_pred_pos))
        if e_pred_neg is not None:
            nll_p_g_xy += -torch.mean(F.logsigmoid(-e_pred_neg))

        # nll of p_y_x
        nll_p_y_x = F.nll_loss(y_log_prob[train_mask], y[train_mask])

        nll_p_y_x += -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            y_log_prob[unlabel_mask])

        # nll of q_y_xg
        nll_q_y_xg = -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            post_y_log_prob[unlabel_mask])

        return nll_p_g_xy + nll_p_y_x + nll_q_y_xg


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




