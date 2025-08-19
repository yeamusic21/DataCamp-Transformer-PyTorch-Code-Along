# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

from transformer_tutorial import PositionWiseFeedForward
import torch
import numpy as np
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
###------------------
### Review Initialization
###------------------

d_model = 10
max_seq_length = 20 # 

testmod = PositionWiseFeedForward(d_model, max_seq_length)
print(testmod)