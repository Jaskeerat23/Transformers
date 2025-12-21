import torch
import torch.nn as nn
import numpy as np
import Attention

class Encoder(nn.Module):
    
    def __init__(self, seq_len, d_model, d_ff, device, mask = None):
        super().__init__()
        
        #Feed forward Network
        self.ffl1 = nn.Linear(in_features = d_model, out_features = d_ff)
        self.ffl2 = nn.Linear(in_features = d_ff, out_features = d_model)
        self.relu = nn.ReLU()
        
        #Layer Norm layers
        self.layer_norm1 = nn.LayerNorm([seq_len, d_model])
        self.layer_norm2 = nn.LayerNorm([seq_len, d_model])
        
        #Multi Head Attention Part
        self.multi_head_attention = Attention.MultiHeadAttention(device = device, mask = mask)
        
        #The below projection layers are used to transform X to Q, K, V respectively
        self.w_q_enc = nn.Linear(in_features = 512, out_features = 512)
        self.w_k_enc = nn.Linear(in_features = 512, out_features = 512)
        self.w_v_enc = nn.Linear(in_features = 512, out_features = 512)
    
    def forward(self, X):
        pass