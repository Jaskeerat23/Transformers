import torch
import torch.nn as nn
import numpy as np
import Attention

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device is {device}")

class Decoder(nn.Module):
    
    def __init__(self, num_heads, d_model, d_ff, device, post_norm = True):
        super().__init__()
        self.post_norm = post_norm
        
        #FeedForward Network
        self.ffl1 = nn.Linear(in_features = d_model, out_features = d_ff)
        self.ffl2 = nn.Linear(in_features = d_ff, out_features = d_model)
        self.relu = nn.ReLU()
        
        #LayerNorms
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
        #Attention Heads
        self.cross_attn = Attention.MultiHeadAttention(d_model = d_model, num_heads = num_heads, device = device)
        self.self_attn = Attention.MultiHeadAttention(d_model = d_model, num_heads = num_heads, device = device)
        
        #Wq, Wk, Wv for first sub-layer of decoder
        self.w_q_dec = nn.Linear(in_features = d_model, out_features = d_model)
        self.w_k_dec = nn.Linear(in_features = d_model, out_features = d_model)
        self.w_v_dec = nn.Linear(in_features = d_model, out_features = d_model)
        
        #Wq, Wk, Wv for second sub-layer of decoder
        self.w_q_dec_ca = nn.Linear(in_features = d_model, out_features = d_model)
        self.w_k_enc = nn.Linear(in_features = d_model, out_features = d_model)
        self.w_v_enc = nn.Linear(in_features = d_model, out_features = d_model)
        
        #Dropout Layer
        self.dropout_self_attn = nn.Dropout(p = 0.1)
        self.dropout_cross_attn = nn.Dropout(p = 0.1)
        self.dropout_ffn = nn.Dropout(p = 0.1)
    
    #Y represents output embeddings since during training we perform teacher forcing
    #according to the original paper
    def forward(self, Y, enc_out, dec_padding_mask = None, causal_mask = None, enc_padding_mask = None):
        
        Q = self.w_q_dec(Y)
        K = self.w_k_dec(Y)
        V = self.w_v_dec(Y)
        
        self_attn_mask = causal_mask
        if dec_padding_mask is not None:
            self_attn_mask = causal_mask + dec_padding_mask
            
        self_attn_out = self.self_attn(Q, K, V, self_attn_mask)
        
        sub_layer1_out = self.layer_norm1(Y + self.dropout_self_attn(self_attn_out))
        
        Q_dec = self.w_q_dec_ca(sub_layer1_out)
        K_enc = self.w_k_enc(enc_out)
        V_enc = self.w_v_enc(enc_out)
        
        cross_attn_out = self.cross_attn(Q_dec, K_enc, V_enc, enc_padding_mask)
        
        sub_layer2_out = self.layer_norm2(sub_layer1_out + self.dropout_cross_attn(cross_attn_out))
        
        ffl1_out = self.relu(self.ffl1(sub_layer2_out))
        ffn_out = self.ffl2(ffl1_out)
        
        sub_layer3_out = self.layer_norm3(sub_layer2_out + self.dropout_ffn(ffn_out))
        
        return sub_layer3_out

if __name__ == "__main__":
    
    Y = torch.randn(32, 30, 512).to(device)
    enc_out = torch.randn(32, 20, 512).to(device)
    
    decoder_blk = Decoder(num_heads = 8, d_model = 512, d_ff = 2048, device = device).to(device)
    
    dec_out = decoder_blk(Y, enc_out)
    
    print(f"Shape of decoder output is {dec_out.shape}")