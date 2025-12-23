import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder
from PositionalEncoding import PositionalEncoding

import math

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device is {device}")

class TransformerNetwork(nn.Module):
    
    def __init__(self, embedding_matrix, n_x, num_heads, d_model, d_ff, seq_len, vocab_size, device):
        super().__init__()
        self.n_x = n_x
        self.num_heads = num_heads
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.emb_mat = nn.Embedding.from_pretrained(embeddings = embedding_matrix, freeze = False)
        self.positional_encoding = PositionalEncoding(seq_len = seq_len, d_model = d_model, device = device)
        
        self.encoder_stk = nn.ModuleList([
            Encoder(num_heads = num_heads, d_model = d_model, d_ff = d_ff, device = device, mask = None, post_norm = True) for i in range(n_x)
        ])
        
        self.decoder_stk = nn.ModuleList([
            Decoder(num_heads = num_heads, d_model = d_model, d_ff = d_ff, device = device, mask = None, post_norm = True) for i in range(n_x)
        ])
        
        self.linear = nn.Linear(in_features = d_model, out_features = vocab_size)
    
    def forward(self, X, Y, pad_token):
        
        B = X.shape[0]
        
        pe_embs_x = self.positional_encoding(X)
        
        