import time
import datetime
import numpy as np
original_weight = [100, 100, 100]
weight_array = [0.2, 0.3, 0.5]
data_heterogeneity = [1 for _ in range(3)]
def softmax(inputs):
    exp_values = np.exp(inputs)
    softmax_values = exp_values / np.sum(exp_values)
    return softmax_values

def test_Fun(x, y):
    x = x+12
x = 10
test_Fun(x,1)
print(x)
