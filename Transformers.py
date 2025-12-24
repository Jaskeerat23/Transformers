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
    
    def __init__(self, 
                src_embedding_matrix : torch.Tensor,
                tgt_embedding_matrix : torch.Tensor,
                n_x : int,
                num_heads : int,
                d_model : int,
                d_ff : int, 
                src_seq_len : int,
                tgt_seq_len : int,
                vocab_size : int,
                device : str
                ):
        super().__init__()
        self.n_x = n_x
        self.num_heads = num_heads
        self.d_model = d_model
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.device = device
        
        self.src_emb_mat = nn.Embedding.from_pretrained(embeddings = src_embedding_matrix, freeze = False)
        self.tgt_emb_mat = nn.Embedding.from_pretrained(embeddings = tgt_embedding_matrix, freeze = False)
        
        self.src_positional_encoding = PositionalEncoding(seq_len = src_seq_len, d_model = d_model, device = device)
        self.tgt_positional_encoding = PositionalEncoding(seq_len = tgt_seq_len, d_model = d_model, device = device)
        
        self.encoder_stk = nn.ModuleList([
            Encoder(num_heads = num_heads, d_model = d_model, d_ff = d_ff, device = device, post_norm = True) for i in range(n_x)
        ])
        
        self.decoder_stk = nn.ModuleList([
            Decoder(num_heads = num_heads, d_model = d_model, d_ff = d_ff, device = device, post_norm = True) for i in range(n_x)
        ])
        
        self.linear = nn.Linear(in_features = d_model, out_features = vocab_size)
    
    def forward(self, X : torch.Tensor, Y : torch.Tensor, pad_token : int):
        
        pe_embs_x = self.src_positional_encoding(self.src_emb_mat(X))
        pe_embs_y = self.tgt_positional_encoding(self.tgt_emb_mat(Y))
        
        enc_padding_mask = (((X == pad_token).unsqueeze(dim = 1)).unsqueeze(dim = 1) * float('-inf')).to(device)
        
        dec_padding_mask = ((Y == pad_token).unsqueeze(dim = 1)).unsqueeze(dim = 1) * float('-inf')
        
        L = Y.shape[1]
        
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=Y.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        
        if __name__ == "__main__":
            print(f"Dimension of encoder padding mask is {enc_padding_mask.shape}")
            print(f"Dimension of decoder padding mask is {dec_padding_mask.shape}")
            print(f"Dimension of causal mask is {causal_mask.shape}")
        
        enc_out = pe_embs_x
        
        for encoder in self.encoder_stk:
            enc_out = encoder(enc_out, enc_padding_mask)
        
        dec_out = pe_embs_y
        
        for decoder in self.decoder_stk:
            dec_out = decoder(dec_out, enc_out, dec_padding_mask, causal_mask, enc_padding_mask)
        
        return self.linear(dec_out)
        
if __name__ == "__main__":
    
    X = torch.randint(low = 0, high = 20, size = (32, 20)).to(device)
    Y = torch.randint(low = 0, high = 30, size = (32, 30)).to(device)
    emb_mat = torch.randn(size = (10000, 512)).to(device)
    transformer = TransformerNetwork(emb_mat, emb_mat, 6, 8, 512, 2048, 20, 30, 10000, device).to(device)

    ans = transformer(X, Y, 1)
    
    print(f"Transformers outputs tensor of shape {ans.shape}")