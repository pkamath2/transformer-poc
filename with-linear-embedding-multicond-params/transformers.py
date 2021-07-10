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
    
        self.seq_length = seq_length
        self.emb = emb
        
        self.num_tokens = num_tokens
        
        self.token_embedding = nn.Linear(4, emb)
        self.token_embedding_batchnorm = nn.BatchNorm1d(num_features=emb, track_running_stats=False) 
#         self.token_embedding_norm = nn.LayerNorm(emb)
        self.token_embedding_activation = nn.ReLU()
        
        self.pos_embedding = nn.Linear(seq_length, emb)
        self.pos_embedding_batchnorm = nn.BatchNorm1d(num_features=emb, track_running_stats=False)
#         self.pos_embedding_norm = nn.LayerNorm(emb)
        self.pos_embedding_activation = nn.ReLU()
        

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
        
        tokens = self.token_embedding(x) # Input to the model = batch_size X sample_length X 1 i.e. 16 X 512 X 1
        b, t, e = tokens.size() # Output from 'Word' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128
        tokens = self.token_embedding_batchnorm(tokens.view(b, e, t)).view(b, t, e)
        tokens = self.token_embedding_activation(tokens)
        
#         b, t, e = tokens.size() # Output from 'Word' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128
        
        trange = torch.arange(t)
        trange = trange.cuda().float().view(1, -1)
        positions = self.pos_embedding(trange)[None, :, :].expand(b, t, e)# Output from 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128
        positions = self.pos_embedding_batchnorm(positions.contiguous().view(b, e, t)).view(b, t, e)
        positions = self.pos_embedding_activation(positions)
        
        
        x = tokens + positions #+ conditioning # Output from 'Word' + 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128 
        
        
        x = self.tblocks(x) # batch_size X sample_length X embedding_size i.e. 16 X 512 X 128

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens) # Output from Linear (in preparation of the softmax layer) = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256

        x = F.log_softmax(x, dim=2) # Output from Log softmax = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256

        return x
    
#     def __init__(self, seq_length, heads, depth, num_tokens):
#         super().__init__()

#         self.num_tokens = num_tokens
    
#         self.token_embedding = nn.Linear(seq_length, 1024)
#         self.pos_embedding = nn.Linear(seq_length, 1024)
        
# #         self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
# #         self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

#         tblocks = []
#         for i in range(depth):
#             tblocks.append(
#                 TransformerBlock(seq_length=seq_length, heads=heads, mask=True, pos_embedding=self.pos_embedding))

#         self.tblocks = nn.Sequential(*tblocks)

#         self.toprobs = nn.Linear(1024, num_tokens)

#     def forward(self, x):
#         """
#         :param x: A (batch, sequence length) integer tensor of token indices.
#         :return: predicted log-probability vectors for each token based on the preceding tokens.
#         """
        
# #         print('x shape', x.shape)
#         tokens = self.token_embedding(x) # Input to the model = batch_size X sample_length i.e. 16 X 512
#         b, t = tokens.size() # Output from 'Word' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128
        
        
#         trange = torch.arange(x.shape[1]).float()
#         trange = trange.view(1, x.shape[1])
#         trange = trange.cuda()
#         positions = self.pos_embedding(trange)
#         positions = positions.expand(b, t) # Output from 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128

        
        
# #         print('Before add', tokens.shape, positions.shape)
#         x = tokens + positions # Output from 'Word' + 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128 
# #         print('After add', x.shape)
        
#         x = self.tblocks(x) # batch_size X sample_length X embedding_size i.e. 16 X 512 X 128

# #         print('--------------------')
# #         print('After transformer: ', x.shape)
#         x = self.toprobs(x) # Output from Linear (in preparation of the softmax layer) = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256
# #         print('After toprobs: ', x.shape)
#         x = F.log_softmax(x, dim=1) # Output from Log softmax = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256
# #         print('After log_softmax: ', x.shape)
#         return x