# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

from transformer_tutorial import PositionWiseFeedForward
import torch
import numpy as np
import torch.nn as nn
import math

###------------------
### Review Initialization
###------------------

d_model = 10
d_ff = 20 # Dimensionality of the inner layer in the feed-forward network.

testmod = PositionWiseFeedForward(d_model,d_ff)
print(testmod)

###------------------
### Step Through Code
###------------------

fc1 = nn.Linear(d_model, d_ff)
fc2 = nn.Linear(d_ff, d_model)
relu = nn.ReLU()

# Create some fake data with dimensions 3, 10, 10
fake_data = np.random.rand(3, 10, 10)
print("Fake data:")
print(fake_data)
list_data = fake_data.tolist()
x = torch.tensor(list_data)

step1 = fc1(x) # 10x10 * 10x20 gives 10x20

print("Step1 results:")
print(step1)
print(step1.size())