import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, device):
        super().__init__()
        
        self.PE = torch.zeros(size = (1, seq_len, d_model)).to(device)
        
        seq_mat = torch.Tensor(torch.arange(end = seq_len, dtype = torch.float)).unsqueeze(dim = 1).to(device)
        div_mat = torch.Tensor([1/(10000**((2*i)/d_model)) for i in range(0, d_model//2)]).unsqueeze(dim = 0).to(device)
        
        if __name__ == "__main__":
            print(f"Dimension of sequence matrix is {seq_mat.shape}")
            print(f"Dimension of div_mat is {div_mat.shape}")
        
        angles = torch.matmul(seq_mat, div_mat)
        
        self.PE[:, :, 0::2] = torch.sin(angles)
        self.PE[:, :, 1::2] = torch.cos(angles)
        
        self.pe_dropout = nn.Dropout(p = 0.1)
        
        self.register_buffer("PE_test", self.PE)
        
        if __name__ == "__main__":
            print(f"Dimension of Positional Encoded Matrix is {self.PE.shape}")
    
    def forward(self, embs):
        
        # print(embs.shape[1])
        pe_embs = self.PE[:, :embs.shape[1], :] + embs
        return self.pe_dropout(pe_embs)

if __name__ == "__main__":
    
    pe = PositionalEncoding(seq_len = 20, d_model = 512, device = 'cuda')
    
    X = torch.randn(size = (32, 20, 512)).to('cuda')
    
    print(pe(X).shape)