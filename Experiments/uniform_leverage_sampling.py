"""
    Test our results for uniform and iid projection
    Plots in Section 3.3
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from Data import DATA
from Sketching_methods import uniform_sampling

n = 2000
p = 100
gamma = p / n
np.random.seed(130)
X = np.random.randn(n, p)
beta = np.random.rand(p, 1)
rep = 50
c = np.linspace(0.1, 1, 20)
track = np.empty((20, 4))

# uniform sampling
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpro = np.empty((rep, 4))
    for k in range(rep):
        data = DATA(type='Gaussian', n=n, p=p, X=X, beta=beta)
        vpro[k, :] = uniform_sampling(data, r)
    track[i, :] = np.mean(vpro, axis=0)
    i = i + 1

# Figure 7
d = np.linspace(0.15, 1, 500)
plt.figure(0, figsize=(10, 8))
p11 = plt.subplot(221)
p11.cla()
p11.scatter(c[1:], track[1:, 0], label='Simulation')
p11.plot(d, (1-gamma)/(d-gamma), label=r'Theory: $\frac{n-p}{r-p}$')
p11.grid(linestyle='dotted')
p11.set_ylabel('VE', fontsize=13)
p11.legend()

p12 = plt.subplot(222)
p12.scatter(c[1:], track[1:, 1], label='Simulation')
p12.plot(d, (1-gamma)/(d-gamma), label=r'Theory: $\frac{n-p}{r-p}$')
p12.grid(linestyle='dotted')
p12.set_ylabel('PE', fontsize=13)
p12.legend()

p21 = plt.subplot(223)
plt.scatter(c[1:], track[1:, 2], label='Simulation')
p21.plot(d, d/(d-gamma)-gamma/(1-gamma), label=r'Theory: $\frac{r}{r-p}-\frac{p}{n-p}$')
p21.grid(linestyle='dotted')
p21.set_ylabel('RE', fontsize=13)
p21.set_xlabel(r'$r/n$', fontsize=13)
p21.legend()

p22 = plt.subplot(224)
p22.scatter(c[1:], track[1:, 3], label='Simulation')
p22.plot(d, d*(1-gamma)/(d-gamma), label=r'Theory: $\frac{r(n-p)}{n(r-p)}$')
p22.grid(linestyle='dotted')
p22.set_ylabel('OE', fontsize=13)
p22.set_xlabel(r'$r/n$', fontsize=13)
p22.legend()
plt.subplots_adjust(hspace=.01)
plt.savefig('/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/uniform_sampling.png')


# leverage sampling
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

    x_test = np.random.randn(n, p)
    epsilon_test = np.random.randn(n, 1)
    y_test = x_test @ beta + epsilon_test
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return ve, pe, re, oe


# generate data
np.random.seed(20398)
epsilon = np.random.randn(n, 1)
Y = X @ beta + epsilon

# full OLS
beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
v_full = np.linalg.norm(beta - beta_full) ** 2
p_full = np.linalg.norm(X @ beta - X @ beta_full) ** 2
r_full = np.linalg.norm(Y - X @ beta_full) ** 2

# estimate leverage scores
r_1 = 150
x_tilde = X
x_tilde = np.array([dct(x_tilde[:, i]) for i in range(p)]).T
idx_sample = np.random.choice(n, r_1, replace=False)
x_tilde = x_tilde[idx_sample, :]
rr = np.linalg.qr(x_tilde, mode='r')
omega = X @ np.linalg.inv(rr)
leverage_score = np.array([np.linalg.norm(omega[i, :]) ** 2 for i in range(n)])
leverage_score = leverage_score / sum(leverage_score) * p

i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpro = np.empty((rep, 4))
    for k in range(rep):
        vpro[k, :] = leverage_sampling(r)
    track[i, :] = np.mean(vpro, axis=0)
    i = i + 1

# Figure 8
d = np.linspace(0.15, 1, 500)
plt.figure(0, figsize=(10, 8))
p11 = plt.subplot(221)
p11.cla()
p11.scatter(c[1:], track[1:, 0], label='Simulation')
p11.plot(d, (1 - gamma) / (d - gamma), label=r'Theory: $\frac{n-p}{r-p}$')
p11.grid(linestyle='dotted')
p11.set_ylabel('VE', fontsize=13)
p11.legend()

p12 = plt.subplot(222)
p12.scatter(c[1:], track[1:, 1], label='Simulation')
p12.plot(d, (1 - gamma) / (d - gamma), label=r'Theory: $\frac{n-p}{r-p}$')
p12.grid(linestyle='dotted')
p12.set_ylabel('PE', fontsize=13)
p12.legend()

p21 = plt.subplot(223)
plt.scatter(c[1:], track[1:, 2], label='Simulation')
p21.plot(d, d / (d - gamma) - gamma / (1 - gamma), label=r'Theory: $\frac{r}{r-p}-\frac{p}{n-p}$')
p21.grid(linestyle='dotted')
p21.set_ylabel('RE', fontsize=13)
p21.set_xlabel(r'$r/n$', fontsize=13)
p21.legend()

p22 = plt.subplot(224)
p22.scatter(c[1:], track[1:, 3], label='Simulation')
p22.plot(d, d * (1 - gamma) / (d - gamma), label=r'Theory: $\frac{r(n-p)}{n(r-p)}$')
p22.grid(linestyle='dotted')
p22.set_ylabel('OE', fontsize=13)
p22.set_xlabel(r'$r/n$', fontsize=13)
p22.legend()
plt.subplots_adjust(hspace=.01)
plt.savefig('/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/leverage_sampling.png')



