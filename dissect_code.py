from transformer_tutorial import MultiHeadAttention
import torch
import numpy as np
import torch.nn as nn
import math

###------------------
### Review Initialization
###------------------

d_model = 10
num_heads = 2

testmod = MultiHeadAttention(d_model,num_heads)
print(testmod)

###------------------
### Step Through Code
###------------------

# Initialize dimensions
d_k = d_model // num_heads
print("d_k: ", d_k) # Dimension of each head's key, query, and value

# "split_heads" is the first step in "forward"
# to run it, we need some fake data to represent 
# batch_size, seq_length, d_model = x.size()
# lets use the dimensions - 3, 10, 10

# Create some fake data with dimensions 3, 10, 10
fake_data = np.random.rand(3, 10, 10)

print(fake_data)

list_data = fake_data.tolist()

# 3, 10, 10
tensor_from_list = torch.tensor(list_data)

print(tensor_from_list.size())

batch_size, seq_length, d_model = tensor_from_list.size()

res_split_heads = tensor_from_list.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)

print("Result of split heads:")
print(res_split_heads)
print(res_split_heads.size())

# technically, split_heads runs on nn.Linear(d_model, d_model)
#
# --- here is an example of using nn.Linear from PyTorch documentation
# --- https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
# m = nn.Linear(20, 30)
# input = torch.randn(128, 20)
# output = m(input)
# print(output.size()) # torch.Size([128, 30])
# 
# 128x20 (*) 20x30 = 128x30
# so m(input) is just matrix multiplicaiton between input and weights
# so let's try putting it all together...

print("=============================================")
print("Result of split heads (all together now!):")

W_q = nn.Linear(d_model, d_model)
W_q_Q = W_q(tensor_from_list)
print("W_q_Q.size(): ", W_q_Q.size())
batch_size, seq_length, d_model = W_q_Q.size()
res_split_heads = W_q_Q.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)

print(res_split_heads)
print(res_split_heads.size())

###
# scaled_dot_product_attention
###
print("=============================================")

q = torch.tensor(np.random.rand(3, 10, 10).tolist())
k = torch.tensor(np.random.rand(3, 10, 10).tolist())
v = torch.tensor(np.random.rand(3, 10, 10).tolist())

W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)

W_q_Q = W_q(tensor_from_list)
W_k_K = W_k(tensor_from_list)
W_v_V = W_v(tensor_from_list)

batch_size, seq_length, d_model = W_q_Q.size()
Q = W_q_Q.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
batch_size, seq_length, d_model = W_k_K.size()
K = W_k_K.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
batch_size, seq_length, d_model = W_v_V.size()
V = W_v_V.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)

# attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

print("K size:", K.size())
print("Q size:", Q.size())

step1 = K.transpose(-2, -1)
print("step1 size: ", step1.size())
print("step1: ", step1)

step2 = torch.matmul(Q, step1)
print("step2 size: ", step2.size())
print("step2: ", step2)

attn_scores = step2 / math.sqrt(d_k)
print("attn_scores size: ", attn_scores.size())
print("attn_scores: ", attn_scores)

print("")
print("Skipping mask because looks like it's just set to None at this point in the code.")
print("")

attn_probs = torch.softmax(attn_scores, dim=-1)
print("attn_probs size: ", attn_probs.size())
print("attn_probs: ", attn_probs)

print("=============================================")
print("Result of scaled_dot_product_attention:")

output = torch.matmul(attn_probs, V)
print("output size: ", output.size())
print("output: ", output)

