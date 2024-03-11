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
heterogeneity_value = softmax(data_heterogeneity)
new_weight_array = softmax([x*y for x, y in zip(weight_array, heterogeneity_value)])
print(new_weight_array)
