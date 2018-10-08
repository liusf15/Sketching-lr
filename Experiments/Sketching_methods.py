"""
    This script includes functions for 6 sketching methods:
    Gaussian projection, sparse projection, orthogonal projection,
    Randomized Hadamard projection, uniform sampling, leverage based sampling

    Packages required:
    numpy, time, scipy
"""

# import
import numpy as np
import time
import Data
from scipy.sparse import csr_matrix
from scipy.fftpack import dct


# INPUT: target dimension r
# RETURN: [VE, PE, RE, OE]
# apply Gaussian random projection on X and Y, then do linear regression
def gaussian_projection(data, r):
    n, p, beta, X, Y, v_full, p_full, r_full, beta_full = [data.n, data.p, data.beta, data.X, data.Y, data.v_full,
                                                           data.p_full, data.r_full, data.beta_full]
    S = np.random.randn(r, n)
    x_tilde = S @ X
    y_tilde = S @ Y
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe


# INPUT: target dimension r, density of the projection matrix s
# RETURN: [VE, PE, RE, OE]
# apply sparse matrix multiplication on X and Y, then do linear regression
def sparse_projection(data, r, s=0.1):
    n, p, beta, X, Y, v_full, p_full, r_full, beta_full = [data.n, data.p, data.beta, data.X, data.Y, data.v_full,
                                                           data.p_full, data.r_full, data.beta_full]
    S = csr_matrix(np.random.choice([-1, 0, 1], p=[s / 2, 1 - s, s / 2], size=r * n).reshape((r, n)))
    x_tilde = S @ X
    y_tilde = S @ Y
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    beta_hat = beta_hat.reshape((p, 1))
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe


# INPUT: n, p
# OUTPUT: n by p haar-distributed matrix
def generate_haar_matrix(n, p):
    if n <= p:
        return np.linalg.qr(np.random.randn(p, n))[0].T
    else:
        return np.linalg.qr(np.random.randn(n, p))[0]


# INPUT: target dimension r
# RETURN: [VE, PE, RE, OE]
# apply Gaussian random projection on X and Y, then do linear regression
def haar_projection(data, r):
    n, p, beta, X, Y, v_full, p_full, r_full, beta_full = [data.n, data.p, data.beta, data.X, data.Y, data.v_full,
                                                           data.p_full, data.r_full, data.beta_full]
    S = generate_haar_matrix(r, n)
    x_tilde = S @ X
    y_tilde = S @ Y
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe


# INPUT: target dimension r
# RETURN: [VE, PE, RE, OE]
# apply signed permutation, fast DCT for each column (scipy.fftpack), and subsampling on X and Y, then do linear regression
def hadamard_projection(data, r):
    n, p, beta, X, Y, v_full, p_full, r_full, beta_full = [data.n, data.p, data.beta, data.X, data.Y, data.v_full,
                                                           data.p_full, data.r_full, data.beta_full]
    data_tilde = np.concatenate([X, Y], axis=1)
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
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe


# INPUT: target dimension r
# RETURN: [VE, PE, RE, OE]
# apply uniform sampling without replacement on X and Y, then do linear regression
def uniform_sampling(data, r):
    n, p, beta, X, Y, v_full, p_full, r_full, beta_full = [data.n, data.p, data.beta, data.X, data.Y, data.v_full,
                                                           data.p_full, data.r_full, data.beta_full]
    idx_uniform = np.random.choice(n, r, replace=False)
    x_tilde = X[idx_uniform, :]
    y_tilde = Y[idx_uniform]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe


# INPUT: r_1
# OUTPUT: estimated leverage scores of X
# use algorithms proposed in Drineas et al 2012
# apply fast DCT: n -> r_1
# QR decomposition
def fast_leverage(data, r_1):
    n, p, X = [data.n, data.p, data.X]
    r_1 = min(r_1, p + 10)
    x_tilde = X
    np.random.shuffle(x_tilde)
    x_tilde = np.array([dct(x_tilde[:, i]) for i in range(p)]).T
    idx_sample = np.random.choice(n, r_1, replace=False)
    x_tilde = x_tilde[idx_sample, :]
    rr = np.linalg.qr(x_tilde, mode='r')
    omega = X @ np.linalg.inv(rr)
    return np.array([np.linalg.norm(omega[i, :]) ** 2 for i in range(n)])


# INPUT: target dimension r
# RETURN: [VE, PE, RE, OE]
# sampling each row of X w.r.t. leverage scores without replacement on X and Y, then do linear regression
def leverage_sampling(data, r):
    n, p, beta, X, Y, v_full, p_full, r_full, beta_full = [data.n, data.p, data.beta, data.X, data.Y, data.v_full,
                                                           data.p_full, data.r_full, data.beta_full]
    leverage_estimate = fast_leverage(data, int(np.log(n) * p))
    idx_leverage = np.random.choice(n, r, p=leverage_estimate / sum(leverage_estimate), replace=False)
    x_tilde = X[idx_leverage, :]
    y_tilde = Y[idx_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe


# INPUT: target dimension r
# RETURN: [VE, PE, RE, OE]
# use the r data points with largest leverage scores to do linear regression
def largest_leverage(data, r):
    n, p, beta, X, Y, v_full, p_full, r_full, beta_full = [data.n, data.p, data.beta, data.X, data.Y, data.v_full,
                                                           data.p_full, data.r_full, data.beta_full]
    leverage_estimate = fast_leverage(data, int(np.log(n) * p))
    idx_deter_leverage = np.argsort(leverage_estimate)[(n - r): n]
    x_tilde = X[idx_deter_leverage, :]
    y_tilde = Y[idx_deter_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe
