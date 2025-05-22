from transformer_tutorial import MultiHeadAttention
import torch

###------------------
### Review Initialization
###------------------

d_model = 10
num_heads = 1

testmod = MultiHeadAttention(d_model,num_heads)
print(testmod)

###------------------
### Step Through Code
###------------------

# Initialize dimensions
print("d_k: ", d_model // num_heads) # Dimension of each head's key, query, and value

# "split_heads" is the first step in "forward"
# to run it, we need some fake data to represent 
# batch_size, seq_length, d_model = x.size()
# lets use the dimensions - 3, 10, 10

# Create some fake data with dimensions 3, 10, 10
list_data = [
    [],
    [],
    []
]


# 3, 10, 10
tensor_from_list = torch.tensor(list_data)