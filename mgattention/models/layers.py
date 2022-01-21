import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SparseMultiAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, no_networks = 1, concat = True):
        super(SparseMultiAttention, self).__init__()
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
        self.special_spmm = SpecialSpmmFunction.apply
    
    def forward(self, h, adjs):
        wts = self.sm(self.nw)
        Wh = torch.mm(h, self.W)
        N      = Wh.size()[0]
        
        attention = self.compute_attention(Wh, adjs[0], 0, wts)
            
        for i in range(1,self.no_networks):
            attention  += self.compute_attention(Wh, adjs[i], i, wts)
            
        attention = attention.coalesce()    
        edge      = attention.indices()
        vals      = attention.values()
        vals      = F.softmax(vals)
        vals      = F.dropout(vals, self.dropout, training = self.training)
        attention = torch.sparse_coo_tensor(edge, vals, torch.Size([N, N]))
        h_prime   = torch.matmul(attention, Wh)
        print(attention.size())
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def compute_attention(self, Wh, adj, n_index, wts):
        N   = Wh.size()[0]
        Wh1 = torch.matmul(Wh, self.a[self.out_features * 2 * n_index: self.out_features * (2 * n_index + 1), :]).squeeze()
        Wh2 = torch.matmul(Wh, self.a[self.out_features * (2 * n_index + 1) : self.out_features * (2 * n_index + 2), :]).squeeze()
        
        edge    = adj._indices()
        edge_e  = torch.exp(-self.leakyrelu(Wh1[edge[0, :]] + Wh2[edge[1, :]])) 
        return torch.sparse_coo_tensor(edge, wts[0] * edge_e, torch.Size([N, N]))
    
    

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
        torch.cuda.empty_cache()
        if self.concat:
            out = F.elu(h_prime)
        else:
            out = h_prime
        return out
        

    def compute_attention(self, Wh, adj, n_index):
        """
        Compute the attention corresponding the network, indexed by `n_index`, with adjacency matrix adj
        Wh1, Wh2 = N x out_features, multiplied by out_features x 1 => N x 1 
        """
        Wh1 = torch.matmul(Wh, self.a[self.out_features * 2 * n_index: self.out_features * (2 * n_index + 1), :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features * (2 * n_index + 1) : self.out_features * (2 * n_index + 2), :])

        e   = self.leakyrelu(Wh1 + Wh2.T)
        # N x N => 20 x 20
        attention_out = torch.where(adj > 0, e, torch.tensor(0.).cuda())
        return attention_out
        
    
class MGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, no_networks = 1):
        """
        Dense version of MGAT:
        """
        super(MGAT, self).__init__()
        self.dropout = dropout
        
        self.attentions = [MultiGraphAttention(nfeat,
                                               nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               no_networks = no_networks,
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = MultiGraphAttention(nhid * nheads,
                                           nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           no_networks = no_networks,
                                           concat=False)

    def forward(self, x, adjs):
        x  = F.dropout(x, self.dropout, training = self.training) # N x n_feat
        x  = torch.cat([att(x, adjs) for att in self.attentions], dim = 1) # N x (nhid * n_heads) 
        x  = F.dropout(x, self.dropout, training = self.training)  # N x (nhid * n_heads)
        x  = F.elu(self.out_att(x, adjs))                           # N x n_classes
        out = torch.sigmoid(x)
        return out
    
    
class SMGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, no_networks = 1):
        """
        Dense version of MGAT:
        """
        super(SMGAT, self).__init__()
        self.dropout = dropout
        
        self.attentions = [SparseMultiAttention(nfeat,
                                               nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               no_networks = no_networks,
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SparseMultiAttention(nhid * nheads,
                                           nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           no_networks = no_networks,
                                           concat=False)

    def forward(self, x, adjs):
        x  = F.dropout(x, self.dropout, training = self.training) # N x n_feat
        
        print(f"Shape of x is {x.size()}")
        x  = torch.cat([att(x, adjs) for att in self.attentions], dim = 1) # N x (nhid * n_heads) 
        print("here")
        print(f"Shape of x is {x.size()}")
        x  = F.dropout(x, self.dropout, training = self.training)  # N x (nhid * n_heads)
        print(f"Shape of x is {x.size()}")
        x  = F.elu(self.out_att(x, adjs))                          # N x n_classes
        out = F.sigmoid(x, dim = 1)                                         # N x n_classes
        return out
