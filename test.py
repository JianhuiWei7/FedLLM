import time
import datetime
import numpy as np
alpha = [1000, 1000, 1000, 1000, 1000]
p_dirichlet = np.random.dirichlet(alpha)
print(p_dirichlet)
print(np.sum(p_dirichlet))