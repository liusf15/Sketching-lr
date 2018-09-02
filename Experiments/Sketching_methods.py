"""
    Sketching methods:
    Gaussian projection, sparse projection, orthogonal projection,
    Randomized Hadamard projection, uniform sampling, leverage based sampling

    Packages required:
    numpy, time
"""

# import
import numpy as np
import time
import Figure_1
import  Data


# INPUT: target dimension r
# RETURN: [VE, PE, RE, time]
# apply signed permutation, fast DCT for each column (scipy.fftpack), and subsampling on X and Y
def hadamard_projection(r):
    data_tilde = data
    data_tilde[: int(n / 2), :] = -data_tilde[: int(n / 2), :]
    np.random.shuffle(data_tilde)
    data_tilde = np.array([dct(data_tilde[:, i]) for i in range(p + 1)]).T
    data_tilde[0, :] = data_tilde[0, :] / np.sqrt(2)
    idx_sample = np.random.choice(n, r, replace=False)
    x_tilde = data_tilde[idx_sample, :p]
    y_tilde = data_tilde[idx_sample, p]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    beta_hat = beta_hat.reshape((p, 1))
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / p_full
    return ve, pe, re
