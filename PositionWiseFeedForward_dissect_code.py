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