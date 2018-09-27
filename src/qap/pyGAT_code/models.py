import torch
import torch.nn as nn
import torch.nn.functional as F
from pyGAT_code.layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, bn_or_not=True):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, bn_or_not=bn_or_not) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, bn_or_not=bn_or_not)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class GAT_ml(nn.Module):
    def __init__(self, nlay, nfeat, nclass, dropout, alpha, bn_or_not=True):
        """Dense version of GAT."""
        super(GAT_ml, self).__init__()
        self.nlay = nlay
        self.dropout = dropout

        self.attention_0 = GraphAttentionLayer(1, nfeat, dropout=dropout, alpha=alpha, concat=True, bn_or_not=bn_or_not)

        for i in range(nlay):
            self.add_module('attention_{}'.format(i+1), GraphAttentionLayer(nfeat, nfeat, dropout=dropout, alpha=alpha, concat=True, bn_or_not=bn_or_not))

        self.out_att = GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False, bn_or_not=bn_or_not)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        cur = self.attention_0(x, adj)
        for i in range(self.nlay):
            cur = self._modules['attention_{}'.format(i+1)](cur, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(cur, adj))
        return F.log_softmax(x, dim=1)
        # return x

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
