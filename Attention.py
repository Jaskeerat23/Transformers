import torch
import math
import torch.nn as nn
import numpy as np
import hyperparams

d_model = hyperparams.DMODEL
d_k = hyperparams.DK
d_v = hyperparams.DV
num_heads = hyperparams.NUM_HEADS

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device for file Attention.py is {device}")
    
    print(f"d_model is {d_model}")
    print(f"d_v is {d_v}")
    print(f"d_k is {d_k}")

class ScaleDotProductAttention(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        self.softmax = nn.Softmax(dim = -1)
        self.device = device
    
    def forward(self, 
                Q : torch.tensor, 
                K : torch.tensor, 
                V : torch.tensor,
                mask : torch.tensor = None
                ):
        
        if __name__ == "__main__":
            print(f"The dimension of Q is {Q.shape}")
            print(f"The dimension of K is {K.shape}")
            print(f"The dimension of V is {V.shape}")
            
            print(f"The dimension of K.T is {torch.transpose(K, -1, -2).shape}")
        
        K = torch.transpose(K, -1, -2)
        similarity_q_k = torch.matmul(Q, K)
        scaled_similarity = (similarity_q_k/math.sqrt(d_k))
        
        if mask is not None:
            scaled_similarity+=mask
        
        softmaxed = self.softmax(scaled_similarity)
        
        cross_attention = torch.matmul(softmaxed, V)
        
        return cross_attention

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, device, mask = None):
        super().__init__()
        self.scaled_dot_attention = ScaleDotProductAttention(device = device)
        self.linear_layer = nn.Linear(in_features = d_model, out_features = d_model, device = device, bias = False)
        self.mask = mask
        self.num_heads = num_heads
        self.d_model = d_model
        
        #According to paper
        self.d_k = self.d_v = d_model//num_heads
    
    def forward(self, Q, K, V):
        
        B, seq_len_q, _ = Q.shape
        _, seq_len, _ = K.shape
        
        Q = torch.reshape(Q, shape = (B, seq_len_q, self.num_heads, self.d_k))
        K = torch.reshape(K, shape = (B, seq_len, self.num_heads, self.d_k))
        V = torch.reshape(V, shape = (B, seq_len, self.num_heads, self.d_v))
        
        Q = torch.transpose(Q, dim0 = 2, dim1 = 1)
        K = torch.transpose(K, dim0 = 2, dim1 = 1)
        V = torch.transpose(V, dim0 = 2, dim1 = 1)
        
        if __name__ == "__main__":
            
            print(f"Batch size is {B}")
            print(f"Sequence length is {seq_len}")
        
        assert Q.shape[1] == self.num_heads and K.shape[1] == self.num_heads and V.shape[1] == self.num_heads, print(f"Num Heads are not equal to {num_heads}")
        assert self.num_heads * self.d_k == self.d_model, print(f"Shape mismatch")
        
        scaled_attn_out = self.scaled_dot_attention(Q, K, V, self.mask)
        scaled_attn_out_transposed = torch.transpose(scaled_attn_out, 1, 2)
        
        if __name__ == "__main__":
            print(f"The shape of scaled dot product attention weights after transposing is {scaled_attn_out_transposed.shape}")
        
        attn_out = torch.reshape(scaled_attn_out_transposed, shape = (B, seq_len_q, self.num_heads * self.d_k))

        return self.linear_layer(attn_out)

if __name__ == "__main__":
    
    q = torch.randn(32, 30, d_model).to(device)
    k = torch.randn(32, 20, d_model).to(device)
    v = torch.randn(32, 20, d_model).to(device)
    
    scaled_dot_attn = ScaleDotProductAttention(device)
    multi_head_attn = MultiHeadAttention(d_model = 512, num_heads = 8, device = device)
    
    scaled_dot_attn_ans = scaled_dot_attn(q, k, v)
    multi_head_attn_ans = multi_head_attn(q, k, v)
    
    print(f"Scaled dot product shape is {scaled_dot_attn_ans.shape}")
    print(f"multi head attn shape is {multi_head_attn_ans.shape}")