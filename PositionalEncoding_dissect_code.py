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

###------------------
### Step Through Code
###------------------

pe = torch.zeros(max_seq_length, d_model)
print("=========================================")
print("Result of torch.zeros(max_seq_length, d_model):")
print(pe)
print(pe.size())
# Result is just a matrix of 0's of size max_seq_length x d_model

position_1 = torch.arange(0, max_seq_length, dtype=torch.float)
position = position_1.unsqueeze(1)
print("=========================================")
print("Result of torch.arange(0, max_seq_length, dtype=torch.float):")
print(position_1)
print(position_1.size())
print("Result of unsqueeze(1):")
print(position)
print(position.size())
# Example output...
# tensor([[ 0.],
#         [ 1.],
#         [ 2.],
#         [ 3.],
#         [ 4.],
#         [ 5.],
#         [ 6.],
#         [ 7.],
#         [ 8.],
#         [ 9.],
#         [10.],
#         [11.],
#         [12.],
#         [13.],
#         [14.],
#         [15.],
#         [16.],
#         [17.],
#         [18.],
#         [19.]])
# torch.Size([20, 1])