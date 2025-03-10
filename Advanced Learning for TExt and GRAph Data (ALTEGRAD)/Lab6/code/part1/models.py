"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2*n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        
        ############## Task 1
    
        ##################
        indices = adj.coalesce().indices()

        h_0 = self.fc(x)
        h_i, h_j = h_0[indices[0, :]], h_0[indices[1, :]]

        h = torch.cat((h_i, h_j), dim = 1)
        h = self.a(h)
        h = self.leakyrelu(h)
        ##################

        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0,:])
        t = torch.zeros(unique.size(0), device=x.device)
        h_sum = t.scatter_add(0, indices[0,:], h)
        h_norm = torch.gather(h_sum, 0, indices[0,:])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)
        
        ##################
        out = torch.sparse.mm(adj * adj_att, h_0)
        ##################

        return out, alpha


class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj, return_attn = False):
        
        ############## Tasks 2 and 4
    
        ##################
        x, _ = self.mp1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        x, alpha = self.mp2(x, adj)
        x = self.relu(x)
        x = self.fc(x)

        if return_attn:
            return F.log_softmax(x, dim = 1), alpha
        ##################

        return F.log_softmax(x, dim=1)
