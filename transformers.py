import torch
from torch import nn
import torch.nn.functional as F

from modules import TransformerBlock

class GTransformer(nn.Module):
    """
    Transformer for generating audio.
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default'):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, attention_type=attention_type, pos_embedding=self.pos_embedding))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        
        tokens = self.token_embedding(x) # Input to the model = batch_size X sample_length i.e. 16 X 512
        b, t, e = tokens.size() # Output from 'Word' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128

        trange = torch.arange(t)
        trange = trange.cuda()
        positions = self.pos_embedding(trange)[None, :, :].expand(b, t, e) # Output from 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128

        x = tokens + positions # Output from 'Word' + 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128 
        
        x = self.tblocks(x) # batch_size X sample_length X embedding_size i.e. 16 X 512 X 128

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens) # Output from Linear (in preparation of the softmax layer) = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256

        x = F.log_softmax(x, dim=2) # Output from Log softmax = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256

        return x