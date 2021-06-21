import torch
from torch import nn
import torch.nn.functional as F

from util import mask_

import random, math, sys

class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, seq_length, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert seq_length % heads == 0, f'Embedding dimension ({seq_length}) should be divisible by nr. of heads ({heads})'

#         self.seq_length = seq_length
        
        self.seq_length = 1024
        self.heads = heads
        self.mask = mask

        s = seq_length // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(self.seq_length, self.seq_length, bias=False)
        self.toqueries = nn.Linear(self.seq_length, self.seq_length, bias=False)
        self.tovalues  = nn.Linear(self.seq_length, self.seq_length, bias=False)

        self.unifyheads = nn.Linear(self.seq_length, self.seq_length)

    def forward(self, x):

        b, t = x.size()
        h = self.heads
#         assert e == self.seq_length, f'Input embedding dim ({e}) should match layer embedding dim ({self.seq_length})'

        s = t // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)
        
#         print('keys shape = ', keys.shape,', queries shape = ', queries.shape, ', values shape = ', values.shape)

        keys    = keys.view(b, h, s)
        queries = queries.view(b, h, s)
        values  = values.view(b, h, s)
        
#         print('keys shape = ', keys.shape,', queries shape = ', queries.shape, ', values shape = ', values.shape)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, s, 1)
        queries = queries.transpose(1, 2).contiguous().view(b * h, s, 1)
        values = values.transpose(1, 2).contiguous().view(b * h, s, 1)

        queries = queries / (self.seq_length ** (1/4))
        keys    = keys / (self.seq_length ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
#         print('keys shape = ', keys.shape,', queries shape = ', queries.shape, ', values shape = ', values.shape)
        dot = torch.bmm(queries, keys.transpose(1, 2))
#         print('dot shape = ', dot.shape)
        
        assert dot.size() == (b*h, s, s)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
#         print('torch.bmm(dot, values) shape', torch.bmm(dot, values).shape)
        out = torch.bmm(dot, values).view(b, h, s)
#         print('torch.bmm(dot, values) shape', out.shape)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, s * h)
#         print('out shape', out.shape)
        
        out = self.unifyheads(out)
#         print('out shape', out.shape)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, seq_length, heads, mask, ff_hidden_mult=4, dropout=0.0, attention_type='default', pos_embedding=None):
        super().__init__()

        self.attention = SelfAttention(seq_length, heads=heads, mask=mask)
        self.mask = mask

        self.seq_length = 1024
        
        self.norm1 = nn.LayerNorm(self.seq_length)
        self.norm2 = nn.LayerNorm(self.seq_length)

        self.ff = nn.Sequential(
            nn.Linear(self.seq_length, ff_hidden_mult * self.seq_length),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * self.seq_length, self.seq_length)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)
#         print('x after norm 1', x.shape)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)
#         print('x after ff and norm 2', x.shape)

        return x