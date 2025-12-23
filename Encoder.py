import torch
import torch.nn as nn
import numpy as np
import Attention

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device is {device}")

class Encoder(nn.Module):
    
    def __init__(self, num_heads, d_model, d_ff, device, post_norm = True):
        super().__init__()
        self.post_norm = post_norm
        self.num_heads = num_heads
        
        #Feed forward Network
        self.ffl1 = nn.Linear(in_features = d_model, out_features = d_ff)
        self.ffl2 = nn.Linear(in_features = d_ff, out_features = d_model)
        self.relu = nn.ReLU()
        
        #Layer Norm layers
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        #Multi Head Attention Part
        self.self_attn = Attention.MultiHeadAttention(d_model = d_model, num_heads = self.num_heads, device = device)
        
        #The below projection layers are used to transform X to Q, K, V respectively
        self.w_q_enc = nn.Linear(in_features = d_model, out_features = d_model)
        self.w_k_enc = nn.Linear(in_features = d_model, out_features = d_model)
        self.w_v_enc = nn.Linear(in_features = d_model, out_features = d_model)
        
        #DropOut
        self.dropout_1 = nn.Dropout(p = 0.1)
        self.dropout_2 = nn.Dropout(p = 0.1)
    
    def forward(self, X, mask = None):
        
        Q = self.w_q_enc(X)
        K = self.w_k_enc(X)
        V = self.w_v_enc(X)
        
        if __name__ == "__main__":
            print(f"The shape of Q is {Q.shape}")
            print(f"The shape of K is {K.shape}")
            print(f"The shape of V is {V.shape}")
        
        multi_attn_out = self.self_attn(Q, K, V, mask)
        
        sub_layer_1 = self.layer_norm1(X + self.dropout_1(multi_attn_out))
        
        ffn_out = self.ffl2(self.relu(self.ffl1(sub_layer_1)))
        
        enc_out = self.layer_norm2(sub_layer_1 + self.dropout_2(ffn_out))
        
        return enc_out

if __name__ == "__main__":
    
    X = torch.randn(32, 20, 512).to(device)
    
    encoder_block = Encoder(8, 512, 2048, device).to(device)
    
    enc_out = encoder_block(X)
    
    print(f"Shape of encoder output is {enc_out.shape}")