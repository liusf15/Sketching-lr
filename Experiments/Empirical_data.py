"""
    Test our results on empirical datasets
    Plots in Section 2.2 and Section 5
"""

import numpy as np
import pandas as pd
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import wget

# import real datasets
m = 100000
# download the MSD data from https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
msd = np.array(pd.read_table("datasets/YearPredictionMSD.txt", delimiter=',', nrows=100000))
flt = np.array(pd.read_csv('datasets/nycflight.csv'))
flt = flt[:, 1:]

m = flt.shape[0]
n = 2000
p = 21
gamma = p / n
np.random.seed(230)
idx = np.random.choice(m, n, replace=False)
X = flt[idx, 1:]
Y = flt[idx, 0].reshape((n, 1))
beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
r_full = np.linalg.norm(Y - X @ beta_full) ** 2
test = flt[pd.Int64Index(np.arange(0, m, 1)).difference(idx), :]
c = np.linspace(0.1, 1, 20)
rep = 50


# INPUT: target dimension r
# OUTPUT: [RE, OE]
# apply SRHT, then do linear regression
def hadamard_projection(r):
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
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    test_idx = np.random.choice(m - n, n)
    x_test = test[test_idx, 1:]
    y_test = test[test_idx, 0]
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return re, oe


# INPUT: target dimension r
# OUTPUT: [RE, OE]
# apply Gaussian projection, then do linear regression
def gaussian_projection(r):
    S = np.random.randn(r, n)
    x_tilde = S @ X
    y_tilde = S @ Y
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    test_idx = np.random.choice(m - n, n)
    x_test = test[test_idx, 1:]
    y_test = test[test_idx, 0]
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return re, oe


# INPUT: target dimension r
# OUTPUT: [RE, OE]
# apply uniform sampling, then do linear regression
def uniform_sampling(r):
    idx_uniform = np.random.choice(n, r, replace=False)
    x_tilde = X[idx_uniform, :]
    y_tilde = Y[idx_uniform]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    test_idx = np.random.choice(m - n, n)
    x_test = test[test_idx, 1:]
    y_test = test[test_idx, 0]
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return re, oe


# INPUT: target dimension r
# OUTPUT: [VE, PE]
# sampling each row of X w.r.t. leverage scores without replacement on X and Y, then do linear regression
def leverage_sampling(r):
    leverage_probability = leverage_score / p * r
    leverage_probability[leverage_probability > 1] = 1
    idx = np.random.binomial(1, p=leverage_probability, size=n)
    idx_leverage = [i for i, j in enumerate(idx) if j == 1]
    x_tilde = X[idx_leverage, :]
    y_tilde = Y[idx_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full

    test_idx = np.random.choice(m - n, n)
    x_test = test[test_idx, 1:]
    y_test = test[test_idx, 0]
    oe = np.linalg.norm(y_test - x_test @ beta_hat) ** 2 / np.linalg.norm(y_test - x_test @ beta_full) ** 2
    return re, oe


# estimate the leverage scores according to Drineas et al
def fast_leverage():
    r_1 = 150
    x_tilde = X
    x_tilde = np.array([dct(x_tilde[:, i]) for i in range(p)]).T
    idx_sample = np.random.choice(n, r_1, replace=False)
    x_tilde = x_tilde[idx_sample, :]
    rr = np.linalg.qr(x_tilde, mode='r')
    omega = X @ np.linalg.inv(rr)
    leverage_score = np.array([np.linalg.norm(omega[i, :]) ** 2 for i in range(n)])
    leverage_score = leverage_score / sum(leverage_score) * p
    return leverage_score


# Figure 1
track_hadamard_flt = np.empty((20, 2))
i = 0
for xi in c:
    r = int(n * xi)
    print(i)
    ro = np.empty((rep, 2))
    for k in range(rep):
        ro[k, :] = hadamard_projection(r)
    track_hadamard_flt[i, :] = np.mean(ro, axis=0)
    i = i + 1

pd.DataFrame(track_hadamard).to_csv('MSD_hadamard.csv')
pd.DataFrame(track_hadamard_flt).to_csv('flight_hadamard.csv')

track_gauss_flt = np.empty((20, 2))
rep = 50
i = 0
for xi in c:
    r = int(n * xi)
    print(i)
    ro = np.empty((rep, 2))
    for k in range(rep):
        ro[k, :] = gaussian_projection(r)
    track_gauss_flt[i, :] = np.mean(ro, axis=0)
    i = i + 1

pd.DataFrame(track_gauss).to_csv('MSD_gauss.csv')
pd.DataFrame(track_gauss_flt).to_csv('flight_gauss.csv')
d = np.linspace(0.1, 1, 500)
plt.figure(0, figsize=(8, 6))
p22 = plt.subplot(111)
gamma = 90 / 5000
p22.scatter(c[1:], track_gauss[1:, 1], label='Gaussian Simulation')
p22.scatter(c[1:], track_hadamard[1:, 1], label='Hadamard Simulation')
p22.plot(d, (d - gamma ** 2) / (d - gamma), label=r'Theory: $\frac{nr-p^2}{n(r-p)}$', ls='--')
p22.plot(d, d * (1 - gamma) / (d - gamma), label=r'Theory: $\frac{r(n-p)}{n(r-p)}$', ls=':')
p22.grid(linestyle='dotted')
p22.set_ylabel('OE', fontsize=13)
p22.set_xlabel(r'$r/n$', fontsize=13)
p22.set_title('MSD OE', fontsize=13)
p22.legend(fontsize=13)
plt.savefig('plots/front_MSD_OE.png')

p11 = plt.subplot(111)
gamma = 21 / n
p11.scatter(c[1:], track_gauss_flt[1:, 0], label='Gaussian Simulation')
plt.scatter(c[1:], track_hadamard_flt[1:, 1], label='Hadamard Simulation')
plt.plot(d, (d - gamma ** 2) / (d - gamma))
plt.plot(d, d * (1 - gamma) / (d - gamma))

p11.plot(d, d / (d - gamma), label=r'Theory: $\frac{r}{r-p}$', ls='--')
p11.plot(d, d / (d - gamma) - gamma / (1 - gamma), label=r'Theory: $\frac{r}{r-p}-\frac{p}{n-p}$', ls=':')
p11.set_title('nycflights13 RE', fontsize=13)
p11.grid(linestyle='dotted')
p11.set_ylabel('RE', fontsize=13)
p11.set_xlabel(r'$r/n$', fontsize=13)
p11.legend(fontsize=13)
plt.savefig('plots/front_flight_re.png')

p11.plot(d, (d - gamma ** 2) / (d - gamma), label=r'Theory: $\frac{nr-p^2}{n(r-p)}$', ls='--')
p11.plot(d, d * (1 - gamma) / (d - gamma), label=r'Theory: $\frac{r(n-p)}{n(r-p)}$', ls=':')
p11.grid(linestyle='dotted')
p11.set_ylabel('OE')
p11.set_xlabel(r'$r/n$', fontsize=13)
p11.set_title('MSD OE')
p11.legend()

plt.savefig('plots/front_MSD.png')

# Section 5
# MSD
# estimate leverage score
m = msd.shape[0]
n = 5000
p = 90
gamma = p / n
np.random.seed(230)
idx = np.random.choice(m, n, replace=False)
X = msd[idx, 1:]
Y = msd[idx, 0].reshape((n, 1))
beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
r_full = np.linalg.norm(Y - X @ beta_full) ** 2
test = msd[pd.Int64Index(np.arange(0, m, 1)).difference(idx), :]
track_msd = np.empty((4, 20,
                      2))  # track the results for Gaussian, Hadamard, uniform sampling, leverage sampling; 20 dimensions; RE and OE
leverage_score = fast_leverage()
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    ro = np.empty((4, rep, 2))
    for k in range(rep):
        ro[0, k, :] = gaussian_projection(r)
        ro[1, k, :] = hadamard_projection(r)
        ro[2, k, :] = uniform_sampling(r)
        ro[3, k, :] = leverage_sampling(r)
    track_msd[:, i, :] = np.mean(ro, axis=1)
    i = i + 1
pd.DataFrame(track_msd[0, :, :]).to_csv('plots/empirical_msd_gauss.csv')
pd.DataFrame(track_msd[1, :, :]).to_csv('plots/empirical_msd_hadamard.csv')
pd.DataFrame(track_msd[2, :, :]).to_csv('plots/empirical_msd_uniform.csv')
pd.DataFrame(track_msd[3, :, :]).to_csv('plots/empirical_msd_leverage.csv')

d = np.linspace(0.15, 1, 500)
plt.figure(0, figsize=(13, 6))
p22 = plt.subplot(121)
gamma = 90 / 5000
p22.scatter(c[1:], track_msd[0, 1:, 0], label='Gaussian', marker='*', s=60)
p22.scatter(c[1:], track_msd[1, 1:, 0], label='Hadamard', marker='o', s=30)
p22.scatter(c[1:], track_msd[2, 1:, 0], label='Uniform', marker='x', s=60)
p22.scatter(c[1:], track_msd[3, 1:, 0], label='Leverage', marker='+', s=60)
p22.plot(d, d/(d-gamma),label=r'$\frac{r}{r-p}$', ls='--')
p22.plot(d, d/(d-gamma)-gamma/(1-gamma), label=r'$\frac{r}{r-p}-\frac{p}{n-p}$', ls=':')
p22.legend()
p22.grid(linestyle='dotted')
p22.set_xlabel(r'$r/n$', fontsize=13)
p22.set_ylabel('RE', fontsize=13)
p22.set_title('MSD RE', fontsize=13)

p12 = plt.subplot(122)
gamma = 90 / 5000
p12.scatter(c[1:], track_msd[0, 1:, 1], label='Gaussian', marker='*', s=60)
p12.scatter(c[1:], track_msd[1, 1:, 1], label='Hadamard', marker='o', s=30)
p12.scatter(c[1:], track_msd[2, 1:, 1], label='Uniform', marker='x', s=60)
p12.scatter(c[1:], track_msd[3, 1:, 1], label='Leverage', marker='+', s=60)
p12.plot(d, (d-gamma**2)/(d-gamma), label=r'Theory: $\frac{nr-p^2}{n(r-p)}$', ls='--')
p12.plot(d, d * (1 - gamma) / (d - gamma), label=r'Theory: $\frac{r(n-p)}{n(r-p)}$', ls=':')
p12.legend()
p12.grid(linestyle='dotted')
p12.set_xlabel(r'$r/n$', fontsize=13)
p12.set_ylabel('OE', fontsize=13)
p12.set_title('MSD OE', fontsize=13)
plt.savefig('plots/empirical_msd.png')

# flight
# estimate leverage score
m = flt.shape[0]
n = 5000
p = 21
gamma = p / n
np.random.seed(230)
idx = np.random.choice(m, n, replace=False)
X = flt[idx, 1:]
Y = flt[idx, 0].reshape((n, 1))
beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
r_full = np.linalg.norm(Y - X @ beta_full) ** 2
test = flt[pd.Int64Index(np.arange(0, m, 1)).difference(idx), :]
track_flt = np.empty((4, 20,
                      2))  # track the results for Gaussian, Hadamard, uniform sampling, leverage sampling; 20 dimensions; RE and OE
leverage_score = fast_leverage()
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    ro = np.empty((4, rep, 2))
    for k in range(rep):
        ro[0, k, :] = gaussian_projection(r)
        ro[1, k, :] = hadamard_projection(r)
        ro[2, k, :] = uniform_sampling(r)
        ro[3, k, :] = leverage_sampling(r)
    track_flt[:, i, :] = np.mean(ro, axis=1)
    i = i + 1
pd.DataFrame(track_flt[0, :, :]).to_csv('plots/empirical_flt_gauss.csv')
pd.DataFrame(track_flt[1, :, :]).to_csv('plots/empirical_flt_hadamard.csv')
pd.DataFrame(track_flt[2, :, :]).to_csv('plots/empirical_flt_uniform.csv')
pd.DataFrame(track_flt[3, :, :]).to_csv('plots/empirical_flt_leverage.csv')

d = np.linspace(0.2, 1, 500)
plt.figure(0, figsize=(13, 6))
p22 = plt.subplot(121)
gamma = 21 / 5000
p22.scatter(c[2:], track_flt[0, 2:, 0], label='Gaussian', marker='*', s=60)
p22.scatter(c[2:], track_flt[1, 2:, 0], label='Hadamard', marker='o', s=30)
p22.scatter(c[2:], track_flt[2, 2:, 0], label='Uniform', marker='x', s=60)
p22.scatter(c[2:], track_flt[3, 2:, 0], label='Leverage', marker='+', s=60)
p22.plot(d, d/(d-gamma),label=r'$\frac{r}{r-p}$', ls='--')
p22.plot(d, d/(d-gamma)-gamma/(1-gamma), label=r'$\frac{r}{r-p}-\frac{p}{n-p}$', ls=':')
p22.legend()
p22.grid(linestyle='dotted')
p22.set_xlabel(r'$r/n$', fontsize=13)
p22.set_ylabel('RE', fontsize=13)
p22.set_title('nycflights13 RE', fontsize=13)

p12 = plt.subplot(122)
gamma = 21 / 5000
p12.scatter(c[2:], track_msd[0, 2:, 1], label='Gaussian', marker='*', s=60)
p12.scatter(c[2:], track_msd[1, 2:, 1], label='Hadamard', marker='o', s=30)
p12.scatter(c[2:], track_msd[2, 2:, 1], label='Uniform', marker='x', s=60)
p12.scatter(c[2:], track_msd[3, 2:, 1], label='Leverage', marker='+', s=60)
p12.plot(d, (d-gamma**2)/(d-gamma), label=r'Theory: $\frac{nr-p^2}{n(r-p)}$', ls='--')
p12.plot(d, d * (1 - gamma) / (d - gamma), label=r'Theory: $\frac{r(n-p)}{n(r-p)}$', ls=':')
p12.legend()
p12.grid(linestyle='dotted')
p12.set_xlabel(r'$r/n$', fontsize=13)
p12.set_ylabel('OE', fontsize=13)
p12.set_title('nycflights13 OE', fontsize=13)
plt.savefig('plots/empirical_flt.png')
