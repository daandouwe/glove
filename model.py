import torch
import torch.nn as nn


class GloVe(nn.Module):
    """Global Vectors for word embedding."""
    def __init__(self, vocab_size, emb_dim=300, sparse=True):
        super(GloVe, self).__init__()
        # Word embeddings.
        self.embedding = nn.Embedding(vocab_size, emb_dim, sparse=sparse)
        self.bias = nn.Embedding(vocab_size, 1, sparse=sparse)

        # Context embeddings.
        self.embedding_tilde = nn.Embedding(vocab_size, emb_dim, sparse=sparse)
        self.bias_tilde = nn.Embedding(vocab_size, 1, sparse=sparse)

        # Xavier initialization.
        initrange = (2.0 / (vocab_size + emb_dim))**0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_tilde.weight.data.uniform_(-initrange, initrange)
        # Zero initialization.
        self.bias.weight.data.zero_()
        self.bias_tilde.weight.data.zero_()

    def forward(self, indices, logx, weights):
        w, w_tilde = self.embedding(indices), self.embedding_tilde(indices)
        b, b_tilde = self.bias(indices), self.bias_tilde(indices)
        out = w @ torch.t(w_tilde) + b + torch.t(b_tilde)
        loss = torch.mean(weights * (out - logx)**2)
        return loss
