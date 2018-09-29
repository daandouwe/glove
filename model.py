import torch
import torch.nn as nn


class GloVe(nn.Module):
    """Global Vectors."""
    def __init__(self, vocab_size, emb_dim=300, sparse=True):
        super(GloVe, self).__init__()
        self.embedding =  nn.Embedding(vocab_size, emb_dim, sparse=sparse)
        self.bias = nn.Embedding(vocab_size, 1, sparse=sparse)

        # Xavier initialization
        initrange = (2.0 / (vocab_size + emb_dim))**0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.bias.weight.data.uniform_(-initrange, initrange)

    def forward(self, indices, log_x, weights):
        w = self.embedding(indices)
        b = self.bias(indices)
        out = w @ torch.t(w) + b + torch.t(b)
        loss = torch.mean(weights * (out - log_x)**2)
        return loss
