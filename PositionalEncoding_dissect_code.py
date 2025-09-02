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

div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
print("=========================================")
print("Result of torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)):")
print(div_term)
print(div_term.size())
# "div_term: A term used to scale the position indices in a specific way."
# Example output...
# tensor([1.0000e+00, 1.5849e-01, 2.5119e-02, 3.9811e-03, 6.3096e-04])
# torch.Size([5])
print("math.log(10000.0): ", math.log(10000.0))
print("-(math.log(10000.0) / d_model): ", -(math.log(10000.0) / d_model))
print("torch.arange(0, d_model, 2).float(): ", torch.arange(0, d_model, 2).float())

print("==========================================================")
print("position * div_term: ", position * div_term)
print("torch.sin(position * div_term): ", torch.sin(position * div_term))
print("torch.cos(position * div_term): ", torch.cos(position * div_term))
# "The sine function is applied to the even indices and the cosine function to the odd indices of pe."
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
print("pe: ", pe)
# Example output...
# tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
#           1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00],
#         [ 8.4147e-01,  5.4030e-01,  1.5783e-01,  9.8747e-01,  2.5116e-02,
#           9.9968e-01,  3.9811e-03,  9.9999e-01,  6.3096e-04,  1.0000e+00],
#         [ 9.0930e-01, -4.1615e-01,  3.1170e-01,  9.5018e-01,  5.0217e-02,
#           9.9874e-01,  7.9621e-03,  9.9997e-01,  1.2619e-03,  1.0000e+00],
#         [ 1.4112e-01, -9.8999e-01,  4.5775e-01,  8.8908e-01,  7.5285e-02,
#           9.9716e-01,  1.1943e-02,  9.9993e-01,  1.8929e-03,  1.0000e+00],
#         [-7.5680e-01, -6.5364e-01,  5.9234e-01,  8.0569e-01,  1.0031e-01,
#           9.9496e-01,  1.5924e-02,  9.9987e-01,  2.5238e-03,  1.0000e+00],
#         [-9.5892e-01,  2.8366e-01,  7.1207e-01,  7.0211e-01,  1.2526e-01,
#           9.9212e-01,  1.9904e-02,  9.9980e-01,  3.1548e-03,  1.0000e+00],
#         [-2.7942e-01,  9.6017e-01,  8.1396e-01,  5.8092e-01,  1.5014e-01,
#           9.8866e-01,  2.3884e-02,  9.9971e-01,  3.7857e-03,  9.9999e-01],
#         [ 6.5699e-01,  7.5390e-01,  8.9544e-01,  4.4518e-01,  1.7493e-01,
#           9.8458e-01,  2.7864e-02,  9.9961e-01,  4.4167e-03,  9.9999e-01],
#         [ 9.8936e-01, -1.4550e-01,  9.5448e-01,  2.9827e-01,  1.9960e-01,
#           9.7988e-01,  3.1843e-02,  9.9949e-01,  5.0476e-03,  9.9999e-01],
#         [ 4.1212e-01, -9.1113e-01,  9.8959e-01,  1.4389e-01,  2.2415e-01,
#           9.7455e-01,  3.5822e-02,  9.9936e-01,  5.6786e-03,  9.9998e-01],
#         [-5.4402e-01, -8.3907e-01,  9.9990e-01, -1.4096e-02,  2.4856e-01,
#           9.6862e-01,  3.9800e-02,  9.9921e-01,  6.3095e-03,  9.9998e-01],
#         [-9.9999e-01,  4.4257e-03,  9.8514e-01, -1.7173e-01,  2.7281e-01,
#           9.6207e-01,  4.3778e-02,  9.9904e-01,  6.9405e-03,  9.9998e-01],
#         [-5.3657e-01,  8.4385e-01,  9.4569e-01, -3.2506e-01,  2.9688e-01,
#           9.5491e-01,  4.7755e-02,  9.9886e-01,  7.5714e-03,  9.9997e-01],
#         [ 4.2017e-01,  9.0745e-01,  8.8254e-01, -4.7024e-01,  3.2077e-01,
#           9.4716e-01,  5.1731e-02,  9.9866e-01,  8.2024e-03,  9.9997e-01],
#         [ 9.9061e-01,  1.3674e-01,  7.9726e-01, -6.0364e-01,  3.4446e-01,
#           9.3880e-01,  5.5706e-02,  9.9845e-01,  8.8333e-03,  9.9996e-01],
#         [ 6.5029e-01, -7.5969e-01,  6.9200e-01, -7.2190e-01,  3.6793e-01,
#           9.2985e-01,  5.9681e-02,  9.9822e-01,  9.4642e-03,  9.9996e-01],
#         [-2.8790e-01, -9.5766e-01,  5.6939e-01, -8.2207e-01,  3.9117e-01,
#           9.2032e-01,  6.3654e-02,  9.9797e-01,  1.0095e-02,  9.9995e-01],
#         [-9.6140e-01, -2.7516e-01,  4.3251e-01, -9.0163e-01,  4.1416e-01,
#           9.1020e-01,  6.7627e-02,  9.9771e-01,  1.0726e-02,  9.9994e-01],
#         [-7.5099e-01,  6.6032e-01,  2.8479e-01, -9.5859e-01,  4.3689e-01,
#           8.9951e-01,  7.1598e-02,  9.9743e-01,  1.1357e-02,  9.9994e-01],
#         [ 1.4988e-01,  9.8870e-01,  1.2993e-01, -9.9152e-01,  4.5935e-01,
#           8.8826e-01,  7.5568e-02,  9.9714e-01,  1.1988e-02,  9.9993e-01]])

print("==========================================================")
print("pe.unsqueeze(0): ", pe.unsqueeze(0))
# Example output...
# ***No change in pe from before.***
# register_buffer('pe', pe.unsqueeze(0))


#### Per ChatGPT #### 
# In the original "Attention Is All You Need" (Vaswani et al., 2017) paper, the positional encoding is not trainable.
# They used fixed sinusoidal positional encodings that are computed once and added to the input embeddings. Quoting from the paper (Section 3.5, Positional Encoding):
# "We use sine and cosine functions of different frequencies... We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions..."
# So in the original Transformer, the positional encoding layer is frozen—just a deterministic function, not learned.
# However, in follow-up work and implementations, there are two common alternatives:
# Fixed sinusoidal (frozen) — as in the paper.
# Learned positional embeddings — trainable vectors, like word embeddings.
# Both variants are widely used, and modern architectures (like BERT, GPT, etc.) typically use trainable learned positional embeddings instead of sinusoidal.