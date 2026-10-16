from __future__ import annotations
import torch
from torch import nn

class DirectLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, seq_features=7, context_features=6):
        super().__init__(); self.lstm=nn.LSTM(seq_features,hidden_size,num_layers,batch_first=True,dropout=(dropout if num_layers>1 else 0.0)); self.head=nn.Sequential(nn.Linear(hidden_size+24*context_features,hidden_size),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_size,24))
    def forward(self,x,c): return self.head(torch.cat([self.lstm(x)[0][:,-1,:],c.flatten(1)],dim=1))
