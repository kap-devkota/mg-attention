import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiGraphAttention(nn.Module):
    """
    GAT layer,  https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, no_networks = 1, concat = True):
        super(MultiGraphAttention, self).__init__()
        self.dropout      = dropout
        self.in_features  = in_features
        self.out_features = out_features
        self.alpha        = alpha
        self.concat       = concat

        self.W            = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.414)
        self.no_networks  = no_networks
        self.a            = nn.Parameter(torch.empty(size=(2 * no_networks * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)
        self.nw           = nn.Parameter(torch.empty(size=(no_networks, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)
        self.leakyrelu    = nn.LeakyReLU(self.alpha)
        self.sm           = nn.Softmax(dim = 0)


    def forward(self, h, adjs):
        """
        h is of the form (N, in_features), Wh of the form (in_features, out_features)
        """
        Wh  = torch.mm(h, self.W)
        wts = self.sm(self.nw)

        """
        Compute the total attention by combining the attention from individual networks.
        Perform, softmax afterwards
        """
        attention = torch.stack([wts[i] * self.compute_attention(Wh, adj, i)
                                 for i, adj in enumerate(adjs)]).sum(dim = 0)
        attention = F.softmax(attention, dim = 1)
        attention = F.dropout(attention, self.dropout, training = self.training)
        h_prime   = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        

    def compute_attention(self, Wh, adj, n_index):
        """
        Compute the attention corresponding the network, indexed by `n_index`, with adjacency matrix adj
        Wh1, Wh2 = N x out_features, multiplied by out_features x 1 => N x 1 
        """
        Wh1 = torch.matmul(Wh, self.a[self.out_features * 2 * n_index: self.out_features * (2 * n_index + 1), :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features * (2 * n_index + 1) : self.out_features * (2 * n_index + 2), :])

        e   = self.leakyrelu(Wh1 + Wh2.T)

        # Apply this with adjacency matrix
        return torch.where(adj > 0, e, 0)
        
    
class MGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        Dense version of MGAT:
        """
        self.attentions = [MultiGraphAttention(nfeat,
                                               nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = MultiGraphAttention(nhid * nheads,
                                           nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, x, adjs):
        x  = F.dropout(x, self.dropout, training = self.training) # N x n_feat
        x  = torch.cat([att(x, adjs) for att in self.attentions], dim = 1) # N x (nhid * n_heads) 
        x  = F.dropout(x, self.dropout, training = self.training)  # N x (nhid * n_heads)
        x  = F.elu(self.out_att(x, adj))                           # N x n_classes
        return F.log_softmax(x, dim = 1)                           # N x n_classes
