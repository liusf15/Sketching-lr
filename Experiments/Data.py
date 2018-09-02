"""
    Parameters, data, full OLS
"""

import numpy as np
import pandas as pd

# parameters
n = 1000
gamma = 0.05
p = int(n * gamma)

# generate data
np.random.seed(12)
X = np.random.randn(n, p)
beta = np.random.rand(p, 1)
epsilon = np.random.randn(n, 1)
Y = X @ beta + epsilon

# full OLS
beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
v_full = np.linalg.norm(beta - beta_full) ** 2
p_full = np.linalg.norm(X @ beta - X @ beta_full) ** 2
r_full = np.linalg.norm(Y - X @ beta_full) ** 2

# import real data
MSD_data = pd.read_table("YearPredictionMSD.txt", delimiter=',', nrows=n).as_matrix()
