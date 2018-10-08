"""
    This script includes the codes for creating Figure 2
"""

import matplotlib.pyplot as plt
import numpy as np


# leverage sampling using the exact leverage scores
# INPUT: target dimension r
# RETURN: [VE, PE, RE]
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


# largest leverage method using the exact leverage scores
# INPUT: target dimension r
# RETURN: [VE, PE, RE]
def deterministic_leverage(r):
    idx_deter_leverage = np.argsort(leverage_score)[(n - r): n]
    x_tilde = X[idx_deter_leverage, :]
    y_tilde = Y[idx_deter_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re


# inverse eta transform for w^2 for discrete distribution
# INPUT: d1, d2, aspect ratio gamma
# RETURN: inverse eta transform of w^2 evaluated at (1-gamma)
def eta_w2_inv(d1, d2, gamma):
    return 1 / (2 * d1 ** 2 * d2 ** 2) * (-d1 ** 2 - d2 ** 2 + (d1 ** 2 + d2 ** 2) / 2 / (1 - gamma) + np.sqrt(
        (-d1 ** 2 - d2 ** 2 + (d1 ** 2 + d2 ** 2) / 2 / (1 - gamma)) ** 2 + 4 * d1 ** 2 * d2 ** 2 * gamma / (
                1 - gamma)))


# eta transform for sw^2, w discrete distribution at d1, d2
# INPUT: z, d1, d2, gamma, xi
# RETURN: eta transform of sw^2 evaluated at z
def eta_sw2(z, d1, d2, gamma, xi):
    z1 = eta_w2_inv(d1, d2, gamma)
    pi_1 = min(xi / gamma * (1 - 1 / (1 + d1 ** 2 * z1)), 1)
    pi_2 = min(xi / gamma * (1 - 1 / (1 + d2 ** 2 * z1)), 1)
    # pi_1 = xi / gamma * (1 - 1 / (1 + d1 ** 2 * z1))
    # pi_2 = xi / gamma * (1 - 1 / (1 + d2 ** 2 * z1))
    return (1 - pi_1 / 2 - pi_2 / 2) + pi_1 / 2 / (1 + d1 ** 2 * z) + pi_2 / 2 / (1 + d2 ** 2 * z)


# inverse of eta transform of sw^2 evaluated at 1-gamma, bisection method
# INPUT: d1, d2, gamma, xi
# RETURN: inverse of eta transform of sw^2 evaluated at 1-gamma
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


# inverse eta transform of truncated distribution of w^2
# INPUT: d1, d2, gamma, xi
# OUTPUT: inverse eta transform of truncated w^2 evaluated at 1 - gamma / xi
def eta_truncated_w2_inv(d1, d2, gamma, xi):
    if xi < 0.5:
        return gamma / d2 ** 2 / (xi - gamma)
    else:
        b = d1 ** 2 + d2 ** 2 - ((2 * xi - 1) * d2 ** 2 + d1 ** 2) / 2 / (xi - gamma)
        return 1 / (2 * d1 ** 2 * d2 ** 2) * (-b + np.sqrt(b ** 2 + 4 * d1 ** 2 * d2 ** 2 * gamma / (xi - gamma)))


# parameters, data generation
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

# full OLS
beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
v_full = np.linalg.norm(beta - beta_full) ** 2
p_full = np.linalg.norm(X @ beta - X @ beta_full) ** 2
r_full = np.linalg.norm(Y - X @ beta_full) ** 2
data = np.concatenate([X, Y], axis=1)

# compute exact leverage scores
hat = X @ np.linalg.inv(X.T @ X) @ X.T
leverage_score = np.diag(hat)
# plt.hist(leverage_score, 100)
track = np.zeros((20, 6))
c = np.linspace(0.1, 1, 20)
rep = 50

# leverage sampling
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpr_leverage = np.zeros((rep, 3))
    vpr_deterministic = np.zeros((rep, 3))
    for k in range(rep):
        vpr_leverage[k, :] = leverage_sampling(r)
    track[i, :3] = np.mean(vpr_leverage, axis=0)
    i = i + 1

# largest leverage
i = 0
for xi in c:
    r = int(n * xi)
    track[i, 3:] = deterministic_leverage(r)
    i = i + 1

# compute theoretical results
z_denominator = eta_w2_inv(d1, d2, gamma)
d = np.linspace(0.2, 1, 500)

z_numerator = np.zeros(500)
for i in range(500):
    z_numerator[i] = eta_sw2_inv(d1, d2, gamma, d[i])

z_numerator_truncated = np.zeros(500)
for i in range(500):
    z_numerator_truncated[i] = eta_truncated_w2_inv(d1, d2, gamma, d[i])

# PLOTS
plt.figure(2, figsize=(8, 6))
plt.plot(d, z_numerator / z_denominator, ls='-', label='Leverage sampling theory')
plt.plot(d, z_numerator_truncated / z_denominator, ls='--', label='Largest leverage theory')
plt.scatter(c[2:], track[2:, 0], label='Leverage sampling simulation', marker='o')
plt.scatter(c[2:], track[2:, 3], label='Largest leverage simulation', marker='s')
plt.xlabel(r'$r/n$', fontsize=13)
plt.ylabel('VE', fontsize=13)
# plt.title('n=' + str(n) + ',p=' + str(p) + ',d1=' + str(d1) + ',d2=' + str(d2))
plt.title('Leverage Scores Elliptical Model')
plt.grid(linestyle='dotted')
plt.legend()
plt.savefig('example_leverage_elliptical.png')
