"""
    This script includes the codes for creating Figure 2 in Section 2.5.2
"""

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
import time
from scipy import stats
import sklearn as sk
import scipy
import scipy.integrate as integrate
import os
import sys
from Sketching import hadamard_projection


# INPUT: target dimension r
# OUTPUT: [VE, PE, RE]
# sample each row of X and Y independently and uniformly, then do linear regression
def uniform_sampling(r):
    idx_uniform = np.random.choice(n, r, replace=False)
    x_tilde = X[idx_uniform, :]
    y_tilde = Y[idx_uniform]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re


# INPUT: target dimension r
# OUTPUT: [VE, PE, RE]
# sample each row of X w.r.t. leverage scores independently, then do linear regression
def leverage_sampling(r):
    leverage_probability = leverage_score / p * r
    leverage_probability[leverage_probability > 1] = 1
    idx = np.random.binomial(1, p=leverage_probability, size=n)
    idx_leverage = [i for i, j in enumerate(idx) if j == 1]
    x_tilde = X[idx_leverage, :]
    y_tilde = Y[idx_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re


# compute how many less leverage sampling samples than uniform sampling does
def leverage_count(r):
    leverage_probability = leverage_score / p * r
    leverage_probability[leverage_probability > 1] = 1
    return r - sum(leverage_probability)


q = np.linspace(0.1, 1, 10)
for xi in c:
    r = int(n * xi)
    print('r=',r, 'less', leverage_count(r))


# INPUT: target dimension r
# OUTPUT: [VE, PE, RE]
# apply greedy leverage sampling, then do linear regression
def deterministic_leverage(r):
    idx_deter_leverage = np.argsort(leverage_score)[(n - r): n]
    x_tilde = X[idx_deter_leverage, :]
    y_tilde = Y[idx_deter_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re


# inverse eta transform for w^2 evaluated at 1-gamma
def eta_w2_inv(d1, d2, gamma):
    return 1 / (2 * d1 ** 2 * d2 ** 2) * (-d1 ** 2 - d2 ** 2 + (d1 ** 2 + d2 ** 2) / 2 / (1 - gamma) + np.sqrt(
        (-d1 ** 2 - d2 ** 2 + (d1 ** 2 + d2 ** 2) / 2 / (1 - gamma)) ** 2 + 4 * d1 ** 2 * d2 ** 2 * gamma / (
                1 - gamma)))


# eta transform for sw^2, w discrete distribution at d1, d2 evaluated at z
def eta_sw2(z, d1, d2, gamma, xi):
    z1 = eta_w2_inv(d1, d2, gamma)
    pi_1 = min(xi / gamma * (1 - 1 / (1 + d1 ** 2 * z1)), 1)
    pi_2 = min(xi / gamma * (1 - 1 / (1 + d2 ** 2 * z1)), 1)
    # pi_1 = xi / gamma * (1 - 1 / (1 + d1 ** 2 * z1))
    # pi_2 = xi / gamma * (1 - 1 / (1 + d2 ** 2 * z1))
    return (1 - pi_1 / 2 - pi_2 / 2) + pi_1 / 2 / (1 + d1 ** 2 * z) + pi_2 / 2 / (1 + d2 ** 2 * z)


# inverse eta transform for sw^2 evaluated at 1-gamma
# use Newton method to find the solution
def eta_sw2_inv(d1, d2, gamma, xi):
    maxiter = 100
    low = 0
    high = 1
    mid = 0.5
    for t in range(maxiter):
        mid = (low + high) / 2
        if abs(eta_sw2(mid, d1, d2, gamma, xi) - (1 - gamma)) < 1e-8:
            print('converge')
            break
        if eta_sw2(mid, d1, d2, gamma, xi) - (1 - gamma) > 0:
            low = mid
        else:
            high = mid
    return mid


# inverse eta transform of \tilde w^2 evaluated at 1-gamma, for greedy leverage sampling
def eta_truncated_w2_inv(d1, d2, gamma, xi):
    if xi < 0.5:
        return gamma / d2 ** 2 / (xi - gamma)
    else:
        b = d1 ** 2 + d2 ** 2 - ((2 * xi - 1) * d2 ** 2 + d1 ** 2) / 2 / (xi - gamma)
        return 1 / (2 * d1 ** 2 * d2 ** 2) * (-b + np.sqrt(b ** 2 + 4 * d1 ** 2 * d2 ** 2 * gamma / (xi - gamma)))


# compute expectation of sw^2
def E_sw2(d1, d2, gamma, xi):
    z1 = eta_w2_inv(d1, d2, gamma)
    pi_1 = min(xi / gamma * (1 - 1 / (1 + d1 ** 2 * z1)), 1)
    pi_2 = min(xi / gamma * (1 - 1 / (1 + d2 ** 2 * z1)), 1)
    return pi_1 / 2 * d1 ** 2 + pi_2 / 2 * d2 ** 2


# compute expectation of the truncated distribution of w^2
def E_truncated_w2(d1, d2, gamma, xi):
    if xi < 0.5:
        return d2 ** 2
    else:
        return d2 ** 2 / 2 / xi + d1 ** 2 * (1 - 1 / 2 / xi)

# parameters and data generation
n = 20000
p = 1000
gamma = p / n
np.random.seed(8230)
X = np.random.randn(n, p)
d1 = 1
d2 = 3
W = np.random.choice([-d2, -d1, d1, d2], size=n)
X = np.array([X[i, :] * W[i] for i in range(n)])
beta = np.random.rand(p, 1)
epsilon = np.random.randn(n, 1)
Y = X @ beta + epsilon
beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
v_full = np.linalg.norm(beta - beta_full) ** 2
p_full = np.linalg.norm(X @ beta - X @ beta_full) ** 2
r_full = np.linalg.norm(Y - X @ beta_full) ** 2
data = np.concatenate([X, Y], axis=1)
hat = X @ np.linalg.inv(X.T @ X) @ X.T
leverage_score = np.diag(hat)
track = np.zeros((20, 6))
c = np.linspace(0.1, 1, 20)
rep = 50

# numerical results
# leverage score sampling
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpr_leverage = np.zeros((rep, 3))
    # vpr_deterministic = np.zeros((rep, 3))
    for k in range(rep):
        vpr_leverage[k, :] = leverage_sampling(r)
    track[i, :3] = np.mean(vpr_leverage, axis=0)
    i = i + 1

# greedy leverage sampling
i = 0
for xi in c:
    r = int(n * xi)
    track[i, 3:] = deterministic_leverage(r)
    i = i + 1

# uniform sampling
track_uniform = np.zeros((20, 3))
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpr_uniform = np.zeros((rep, 3))
    for k in range(rep):
        vpr_uniform[k, :] = uniform_sampling(r)
    track_uniform[i, :] = np.mean(vpr_uniform, axis=0)
    i = i + 1


# compute theoretical line
z_denominator = eta_w2_inv(d1, d2, gamma)
d = np.linspace(0.2, 1, 100)

z_numerator = np.zeros(100)
for i in range(100):
    z_numerator[i] = eta_sw2_inv(d1, d2, gamma, d[i])

z_numerator_uniform = np.zeros(100)
for i in range(100):
    z_numerator_uniform[i] = eta_w2_inv(d1, d2, gamma / d[i])

A0 = np.zeros(100)
A1 = np.zeros(100)
A2 = np.zeros(100)
A3 = np.zeros(100)
for i in range(100):
    A0[i] = E_sw2(d1, d2, gamma, d[i])
    A1[i] = eta_sw2_inv(d1, d2, gamma, d[i])
    A2[i] = d1 ** 2 / 2 + d2 ** 2 / 2 - E_sw2(d1, d2, gamma, d[i])
    A3[i] = d1 ** 2 / 2 + d2 ** 2 / 2 - E_truncated_w2(d1, d2, gamma, d[i]) * d[i]

z_numerator_truncated = np.zeros(100)
for i in range(100):
    z_numerator_truncated[i] = eta_truncated_w2_inv(d1, d2, gamma, d[i])

# Figure 2
# plot VE
plt.figure(0)
plt.plot(d, z_numerator / z_denominator, ls='-', label='Leverage sampling theory')
plt.scatter(c[2:], track[2:, 0], label='Leverage sampling simulation', marker='o')
plt.plot(d, z_numerator_truncated / z_denominator, ls='--', label='Greedy leverage theory')
plt.scatter(c[2:], track[2:, 3], label='Greedy leverage simulation', marker='s')
plt.plot(d, z_numerator_uniform / z_denominator, label='Uniform sampling theory', ls=':')
plt.scatter(c[2:], track_uniform[2:, 0], label='Uniform sampling simulation', marker='v')
plt.xlabel('r/n', fontsize=15)
plt.ylabel('VE', fontsize=15)
plt.title('Elliptical model sampling - VE', fontsize=15)
plt.grid(linestyle='dotted')
plt.legend(fontsize=12)
plt.savefig('example_VE.png')

# plot PE
plt.figure(1)
plt.scatter(c[2:], track[2:, 1], label='Leverage sampling simulation', marker='o')
plt.plot(d, 1 + 1 / gamma * A2 * A1, ls='-', label='Leverage sampling theory')
plt.scatter(c[2:], track[2:, 4], label='Greedy leverage simulation', marker='s')
plt.plot(d, 1 + 1 / gamma * A3 * z_numerator_truncated, ls='--', label='Greedy leverage theory')
plt.plot(d, 1+(1-d)/gamma*(d1**2/2+d2**2/2)*z_numerator_uniform, label='Uniform sampling theory', ls=':')
plt.scatter(c[2:], track_uniform[2:, 1], label='Uniform sampling simulation', marker='v')
plt.xlabel('r/n', fontsize=15)
plt.ylabel('PE', fontsize=15)
plt.title('Elliptical model sampling - PE', fontsize=15)
plt.grid(linestyle='dotted')
plt.legend(fontsize=12)
plt.savefig('example_PE.png')