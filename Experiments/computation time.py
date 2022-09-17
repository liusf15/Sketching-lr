"""
    This script includes the code for Section 5.10.2, Figure 12
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
import time
from scipy import stats
import sklearn as sk
import scipy
import scipy.integrate as integrate
from scipy.fftpack import fft, dct, ifft
import copy
from scipy.sparse import csr_matrix
from scipy import sparse
from scipy import stats
import sys
import os


# generate multivariate t distribution with auto-regressive scale matrix
def generate_t(n, p, df, Sigma):
    X = np.random.multivariate_normal(np.zeros(p), Sigma, n)
    u = np.random.chisquare(df, 1)
    return X / u * np.sqrt(df)


# INPUT: X, Y
# OUTPUT: full OLS time
def linearregression_time(X, Y):
    start = time.time()
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    end = time.time()
    return end - start


# INPUT: target dimension r
# RETURN: [VE, PE, RE, time]
# apply signed permutation, fast DCT for each column (scipy.fftpack), and subsampling on X and Y, then do linear regression
def hadamard_projection(r):
    start_total = time.time()
    data_tilde = data
    start = time.time()
    data_tilde[: int(n / 2), :] = -data_tilde[: int(n / 2), :]
    np.random.shuffle(data_tilde)
    end = time.time()
    sign_permutation_time = end - start

    start = time.time()
    data_tilde = np.array([dct(data_tilde[:, i]) for i in range(p + 1)]).T
    data_tilde[0, :] = data_tilde[0, :] / np.sqrt(2)
    end = time.time()
    dct_time = end - start

    start = time.time()
    idx_sample = np.random.choice(n, r, replace=False)
    x_tilde = data_tilde[idx_sample, :p]
    y_tilde = data_tilde[idx_sample, p]
    end = time.time()
    sample_time = end - start

    start = time.time()
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    end = time.time()
    lr_time = end - start

    end_total = time.time()

    beta_hat = beta_hat.reshape((p, 1))
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re, sign_permutation_time, dct_time, sample_time, lr_time, end_total - start_total


# INPUT: target dimension r, density of the projection matrix s
# RETURN: [VE, PE, RE, time]
# apply sparse matrix multiplication on X and Y, then do linear regression
def sparse_projection(r, s=0.1):
    start = time.time()
    S = csr_matrix(np.random.choice([-1, 0, 1], p=[s / 2, 1 - s, s / 2], size=r * n).reshape((r, n)))
    x_tilde = S @ X
    y_tilde = S @ Y
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    end = time.time()
    beta_hat = beta_hat.reshape((p, 1))
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re, end - start


# INPUT: target dimension r
# RETURN: [VE, PE, RE, time]
# apply Gaussian random projection on X and Y, then do linear regression
def gaussian_projection(r):
    start = time.time()
    S = np.random.randn(r, n)
    x_tilde = S @ X
    y_tilde = S @ Y
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    end = time.time()
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re, end - start


# INPUT: target dimension r
# RETURN: [VE, PE, RE, time]
# apply uniform sampling without replacement on X and Y, then do linear regression
def uniform_sampling(r):
    start = time.time()
    idx_uniform = np.random.choice(n, r, replace=False)
    x_tilde = X[idx_uniform, :]
    y_tilde = Y[idx_uniform]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    end = time.time()
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re, end - start


# INPUT: r_1
# OUTPUT: estimated leverage scores of X
# use algorithms proposed in Drineas et al 2012
# apply fast DCT: n -> r_1
# QR decomposition
def fast_leverage(r_1):
    r_1 = min(r_1, p + 10)
    x_tilde = copy.copy(X)
    np.random.shuffle(x_tilde)
    x_tilde = np.array([dct(x_tilde[:, i]) for i in range(p)]).T
    idx_sample = np.random.choice(n, r_1, replace=False)
    x_tilde = x_tilde[idx_sample, :]
    rr = np.linalg.qr(x_tilde, mode='r')
    omega = X @ np.linalg.inv(rr)
    return np.array([np.linalg.norm(omega[i, :]) ** 2 for i in range(n)])


# INPUT: target dimension r
# RETURN: [VE, PE, RE, time]
# estimate the leverage scores of X
# sampling each row of X w.r.t. leverage scores without replacement on X and Y, then do linear regression
def leverage_sampling(r):
    start = time.time()
    leverage_estimate = fast_leverage(int(np.log(n) * p))
    idx_leverage = np.random.choice(n, r, p=leverage_estimate / sum(leverage_estimate), replace=False)
    x_tilde = X[idx_leverage, :]
    y_tilde = Y[idx_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    end = time.time()
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re, end - start


# INPUT: target dimension r
# RETURN: [VE, PE, RE, time]
# estimate the leverage scores of X
# apply greedy leverage sampling, then do linear regression
def deterministic_leverage(r):
    start = time.time()
    leverage_estimate = fast_leverage(int(np.log(n) * p))
    idx_deter_leverage = np.argsort(leverage_estimate)[(n - r): n]
    x_tilde = X[idx_deter_leverage, :]
    y_tilde = Y[idx_deter_leverage]
    beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
    end = time.time()
    ve = np.linalg.norm(beta - beta_hat) ** 2 / v_full
    pe = np.linalg.norm(X @ beta - X @ beta_hat) ** 2 / p_full
    re = np.linalg.norm(Y - X @ beta_hat) ** 2 / r_full
    return ve, pe, re, end - start


# INPUT: index t, target dimension r
# OUTPUT: [VE, PE, RE, time]
def main_function(t, r):
    ans = np.zeros(4)
    if t == 0:
        a = hadamard_projection(r)
        ans[: len(a)] = a
        return ans
    if t == 1:
        a = sparse_projection(r)
        ans[: len(a)] = a
        return ans
    if t == 2:
        a = uniform_sampling(r)
        ans[: len(a)] = a
        return ans
    if t == 3:
        a = leverage_sampling(r)
        ans[: len(a)] = a
        return ans
    if t == 4:
        a = deterministic_leverage(r)
        ans[: len(a)] = a
        return ans
    if t == 5:
        a = gaussian_projection(r)
        ans[: len(a)] = a
        return ans


# MAIN
os.chdir('plots/computation_time/results')
experiment_design = np.array([[50000, 0.3], [50000, 0.2], [70000, 0.2]])
rep = 1
# column_names = ['hadamard', 'sparse', 'uniform', 'leverage', 'deter_leverage', 'gaussian']
# column_name2 = ['VE', 'PE', 'RE', 'time']
full_time = np.zeros(3)
begin_iteration = time.time()
for m in range(3):
    n = int(experiment_design[m, 0])
    gamma = experiment_design[m, 1]
    print(n, gamma)
    p = int(n * gamma)
    np.random.seed(2803)
    X = np.random.randn(n, p)
    beta = np.random.rand(p, 1)
    epsilon = np.random.randn(n, 1)
    Y = X @ beta + epsilon
    data = np.concatenate([X, Y], axis=1)

    # full OLS
    start = time.time()
    beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
    end = time.time()
    full_time[m] = end - start
    v_full = np.linalg.norm(beta - beta_full) ** 2
    p_full = np.linalg.norm(X @ beta - X @ beta_full) ** 2
    r_full = np.linalg.norm(Y - X @ beta_full) ** 2

    # hadamard projection
    track = np.zeros((10, 8))
    c = np.arange(int(gamma * 10 + 1) / 10, 1.1, 0.1)
    i = 0
    for xi in c:
        r = int(n * xi)
        print(i, r)
        track[i, :] = hadamard_projection(r)
        i = i + 1
    results = pd.DataFrame(track[:, :], columns=['VE', 'PE', 'RE', 'sign-permutation', 'dct', 'sample', 'lr', 'total'])
    results.to_csv('hadamard_time,n=' + str(n) + ',gamma=' + str(gamma) + '.csv')
results = pd.DataFrame(full_time)
results.to_csv('full_ols_time.csv')
end_iteration = time.time()
print(end_iteration - begin_iteration)

# plots
track_1 = pd.read_csv('hadamard_time,n=50000,gamma=0.2.csv')
track_2 = pd.read_csv('hadamard_time,n=50000,gamma=0.3.csv')
track_3 = pd.read_csv('hadamard_time,n=70000,gamma=0.2.csv')

plt.figure(1, figsize=(6, 6))
p11 = plt.subplot(111)
p11.plot(np.linspace(0.3, 1, 8), track_1.loc[:7, 'total'], label=r'$n=5\times10^4,p=10^4$', ls='-')
p11.plot(np.linspace(0.3, 1, 8), full_time[1] * np.linspace(1, 1, 8), label=r'full $n=5\times10^4,p=10^4$',
         ls='-')
p11.plot(np.linspace(0.4, 1, 7), track_2.loc[:6, 'total'], label=r'$n=5\times10^4,p=1.5\times10^4$', ls='--')
p11.plot(np.linspace(0.3, 1, 8), full_time[0] * np.linspace(1, 1, 8), label=r'full $n=5\times10^4,p=1.5\times10^4$',
         ls='--')
p11.plot(np.linspace(0.3, 1, 8), track_3.loc[:7, 'total'], label=r'$n=7\times10^4,p=1.4\times10^4$', ls='-.')
p11.plot(np.linspace(0.3, 1, 8), full_time[2] * np.linspace(1, 1, 8), label=r'full $n=7\times10^4,p=1.4\times10^4$',
         ls='-.')
p11.set_ylabel('time/sec')
p11.set_xlabel(r'$r/n$')
p11.set_title('Comparing time')
p11.legend(fontsize=6)
p11.grid(linestyle='dotted')
plt.savefig('computation_time_different_n.png')
p12 = plt.subplot(222)
p12.cla()
p12.scatter(np.linspace(0.3, 1, 8), track_3.loc[:7, 'VE'])


C_1 = 4 * 1e-11
plt.figure(2)
temp_1 = np.zeros(10)
n = 5 * 1e4
gamma = 0.2
p = int(n * gamma)
for i in range(8):
    r = n * (gamma + (i + 1) * 0.1)
    print(r)
    temp_1[i] = track_1.loc[i, 'total'] - C_1 * r * p ** 2
temp_1 = temp_1 / (n * p * np.log(n))
np.mean(temp_1[:8])
plt.plot(temp_1[:8])

temp_2 = np.zeros(10)
n = 5 * 1e4
gamma = 0.3
p = int(n * gamma)
for i in range(7):
    r = n * (gamma + (i + 1) * 0.1)
    print(r)
    temp_2[i] = track_2.loc[i, 'total'] - C_1 * r * p ** 2
temp_2 = temp_2 / (n * p * np.log(n))
plt.plot(temp_2[:7])

temp_3 = np.zeros(10)
n = 7 * 1e4
gamma = 0.2
p = int(n * gamma)
for i in range(8):
    r = n * (gamma + (i + 1) * 0.1)
    print(r)
    temp_3[i] = track_3.loc[i, 'total'] - C_1 * r * p ** 2
temp_3 = temp_3 / (n * p * np.log(n))
plt.plot(temp_3[:8])

for i in np.arange(4, 7):
    plt.plot(np.linspace(0.4, 1, 7), track_1.iloc[:7, i])
plt.legend()

n = 50000
p = 10000
plt.plot(np.linspace(0.3, 1, 8), track_1.loc[:7, 'lr'] / p ** 2 / n)
n = 50000
p = 15000
plt.plot(np.linspace(0.4, 1, 7), track_2.loc[:6, 'lr'] / p ** 2 / n)
n = 70000
p = 14000
plt.plot(np.linspace(0.4, 1, 7), track_3.loc[:6, 'lr'] / p ** 2 / n)
d = np.linspace(0.4, 1, 7)
plt.plot(d, d * 4 * 1e-11 - 0.5 * 1e-11)
plt.grid()
dct_mean = [np.mean(track_1.loc[:7, 'dct']), np.mean(track_2.loc[:6, 'dct']), np.mean(track_3.loc[:6, 'dct'])]
n = 1


def time_efficiency(n, p, r):
    return (500 * n * np.log(n) + r * p) / (n * p)


def out_efficiency(n, p, r):
    return r * (n - p) / n / (r - p)


n_seq = np.linspace(1e4, 2e6, 100)
p_seq = np.linspace(5000, 50000, 100)
time_hm = np.empty((100, 100))
for i in range(100):
    for j in range(100):
        time_hm[i, j] = time_efficiency(n_seq[i], p_seq[j], n_seq[i] / 2)
plt.imshow(time_hm)
plt.axis()
plt.colorbar()

OE_hm = np.empty((100, 100))
for i in range(100):
    for j in range(100):
        OE_hm[i, j] = out_efficiency(n_seq[i], p_seq[j], min(n_seq[i] / 2, p_seq[j] + 1))
plt.imshow(OE_hm)
plt.colorbar()

gamma = 0.5
line_style = ['-', '--', '-.', ':']
for i in range(4):
    n = int(1e5 * (i + 1))
    p = int(n * gamma)
    c = np.linspace(0.55, 0.9, 100)
    OE_seq = np.empty(100)
    time_seq = np.empty(100)
    for k in range(100):
        r = n * c[k]
        OE_seq[k] = out_efficiency(n, p, r)
        time_seq[k] = time_efficiency(n, p, r)
    plt.plot(OE_seq, time_seq, label='n=' + str(n), ls=line_style[i])
plt.legend()
plt.xlabel('out-of-sample efficiency')
plt.ylabel('time efficiency')
plt.title(r'$\gamma=0.5$')
plt.grid()
os.chdir('plots/computation_time')
plt.savefig('time_efficiency_gamma='+str(gamma)+'.png')
n = int(1e7)
gamma = 0.1
p = int(n * gamma)
c = np.linspace(0.2, 0.5, 100)
OE_seq = np.empty(100)
time_seq = np.empty(100)
for i in range(100):
    r = n * c[i]
    OE_seq[i] = out_efficiency(n, p, r)
    time_seq[i] = time_efficiency(n, p, r)
plt.plot(OE_seq, time_seq, label=r'$n=10^7,p=10^6$')

plt.legend()
plt.plot(OE_seq, np.linspace(1, 1, 100))
plt.plot(OE_seq, 0.5 * np.linspace(1, 1, 100))

plt.legend(fontsize=6)
plt.figure(3)
plt.plot(c, track[:5, 0, 3], label='Hadamard Projection')
d = np.linspace(0.1, 0.5, 10)
plt.plot(d, full_time * np.linspace(1, 1, 10), label='Full OLS')
plt.ylabel('time/sec')
plt.xlabel(r'r/n')
plt.legend()
plt.title('n=' + str(n) + ', p=' + str(p) + ', Computation time')
results = pd.DataFrame(track[:5, 0, :], columns=['VE', 'PE', 'RE', 'time'])
results.to_csv('Hadamard-n=' + str(n) + 'p=' + str(p) + '.csv')

for i in range(6):
    plt.plot(c[1:], track[1:, i, 3], label=labels[i])
plt.legend()
d = np.linspace(0.2, 1, 100)
plt.plot(d, np.mean(time_full) * np.linspace(1, 1, 100), ls='-.')
plt.plot(c, track[:, 0, 5], label='1')
plt.plot(c, track[:, 1, 3], label='2')
plt.plot(c, track[:, 2, 3], label='3')
plt.plot(c, track[:, 3, 4], label='4')
plt.plot(c, track[:, 4, 3], label='4')
plt.plot(c, track[:, 5, 3], label='4')

plt.legend()
plt.plot(c, track[:, 1, 3], label='sparse_reduced')
plt.plot(c, track[:, 1, 5])
plt.plot(c, track[:, 2, 3])
plt.plot(c, track[:, 3, 6])
plt.plot(c, track[:, 3, 5])
plt.plot(c, track[:, 4, 5])
plt.plot(c, track[:, 4, 4])
plt.plot(c, track[:, 5, 5])
plt.plot(c, track[:, 5, 4])

results = pd.DataFrame(track[:, 0, :])
results.to_csv('hadamard_split_time.csv')
plt.figure(0)
plt.plot(c, track[:, 0, 3], label='sign')
plt.plot(c, track[:, 0, 4], label='permutation')
plt.plot(c, track[:, 0, 5], label='dct')
plt.plot(c, track[:, 0, 6], label='sample')
plt.plot(c, track[:, 0, 7], label='linear regression')
plt.plot(c, np.sum(track[:, 0, 3:8], axis=1), label='sum')
plt.plot(c, track[:, 0, 8], label='full')
plt.legend()

plt.figure(1)
plt.plot(c, track[:, 1, 3], label='sparse projection')
plt.plot(c, track[:, 1, 4], label='linear regression')
plt.plot(c, np.sum(track[:, 1, 3:5], axis=1), label='sum')
plt.plot(c, track[:, 1, 5], label='full')
plt.legend()

# PLOTS
d = np.linspace(0.25, 1, 100)
labels = ['Hadamard', 'Sparse', 'Uniform', 'Leverage', 'deterministic leverage', 'Gaussian']
markers = ['o', '*', '+', 'v', 's', 'p']

plt.figure(6, figsize=(6, 6))
p11 = plt.subplot(221)
p11.cla()
for i in range(6):
    plt.scatter(c[1:], track[1:, i, 0], marker=markers[i], label=labels[i])
plt.grid()
p11.set_xlabel('')
p11.set_ylabel('VE', fontsize=10)
plt.title('VE')
plt.plot(d, (1 - gamma) / (d - gamma), ls='-.', label=r'$\frac{n-p}{r-p}$')
plt.plot(d, 1 + (1 - gamma) / (d - gamma), ls='--', label=r'$1+\frac{n-p}{r-p}$')
plt.legend()

p12 = plt.subplot(222)
p12.cla()
for i in range(6):
    plt.scatter(c[1:], track[1:, i, 1], marker=markers[i], label=labels[i])
plt.grid()
p12.set_xlabel('')
plt.ylabel('PE', fontsize=10)
plt.title('PE')
plt.plot(d, (1 - gamma) / (d - gamma), ls='-.', label=r'$\frac{n-p}{r-p}$')
plt.plot(d, 1 + (1 - gamma) / (d - gamma), ls='--', label=r'$1+\frac{n-p}{r-p}$')
p12.legend(bbox_to_anchor=(1, 1), fontsize=7)

p21 = plt.subplot(223)
for i in range(6):
    plt.scatter(c[1:], track[1:, i, 2], marker=markers[i])
plt.grid()
plt.xlabel(r'$r/n$', fontsize=10)
plt.ylabel('RE', fontsize=10)
plt.title('RE')
plt.plot(d, d / (d - gamma) - gamma / (1 - gamma), ls='-.', label=r'$\frac{r}{r-\gamma}-\frac{\gamma}{1-\gamma}$')
plt.plot(d, d / (d - gamma), ls='--', label=r'$\frac{r}{r-\gamma}$')
plt.legend()

p22 = plt.subplot(224)
for i in range(6):
    plt.scatter(c, np.log(track[:, i, 3] / np.mean(time_full)), marker=markers[i], label=labels[i])
plt.grid()
plt.legend()
plt.xlabel(r'$r/n$', fontsize=13)
plt.ylabel('log time efficiency', fontsize=8)
plt.title('Time efficiency,n=10000,p=2000')

plt.savefig('Computation_time,n=10000,p=1000.png')

i = 0
v_track = np.zeros(10)
hadamard = scipy.linalg.hadamard(2048)[:2000, :2000]
data_tilde = hadamard @ data
for xi in c:
    print(i)
    r = int(n * xi)
    ve = 0
    for k in range(rep):
        data_tilde = dct(data)
        idx = np.random.choice(n, r, replace=False)
        x_tilde = data_tilde[idx, :p]
        y_tilde = data_tilde[idx, p]
        beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
        ve = ve + np.linalg.norm(beta - beta_hat.reshape((p, 1))) ** 2 / v_full
    v_track[i] = ve / rep
    i = i + 1
plt.scatter(c, v_track)
plt.plot(d, (1 - gamma) / (d - gamma))

n = 2000
gamma = 0.1
p = int(n * gamma)
X = np.random.randn(n, p)
beta = np.random.rand(p, 1)
epsilon = np.random.randn(n, 1)
Y = X @ beta + epsilon
data = np.concatenate([X, Y], axis=1)
c = np.linspace(0.1, 1, 10)
time_full = []
time_full_dct = []
i = 0
time_track = np.zeros((10, 2))
for xi in c:
    print(i)
    r = int(n * xi)
    time_reduced = []
    time_temp = []
    for k in range(rep):
        start = time.time()
        data_tilde = np.zeros((n, p))
        data_tilde = np.array([dct(data[:, i]) for i in range(p + 1)]).T
        data_tilde[0, :] = data_tilde[0, :] / np.sqrt(2)
        end = time.time()
        dct_time = end - start  # time for DCT
        x_dct = data_tilde[:, :p]
        y_dct = data_tilde[:, p]

        # start = time.time()
        # beta_hat = np.linalg.inv(x_dct.T @ x_dct) @ x_dct.T @ y_dct
        # end = time.time()
        # time_full_dct.append(end - start)

        start = time.time()
        beta_full = np.linalg.inv(X.T @ X) @ X.T @ Y
        end = time.time()
        lr_full = end - start
        time_full.append(lr_full)

        idx_sample = np.random.choice(n, r, replace=False)
        x_tilde = x_dct[idx_sample, :]
        y_tilde = y_dct[idx_sample]
        x_temp = X[idx_sample, :]
        y_temp = Y[idx_sample]

        start = time.time()
        beta_hat = np.linalg.inv(x_temp.T @ x_temp) @ x_temp.T @ y_temp
        end = time.time()
        lr_temp = end - start
        time_temp.append(lr_temp)

        start = time.time()
        beta_hat = np.linalg.inv(x_tilde.T @ x_tilde) @ x_tilde.T @ y_tilde
        end = time.time()
        lr_time = end - start  # time for reduced linear regression
        time_reduced.append(lr_time)

    time_track[i, :] = [np.mean(time_reduced), np.mean(time_temp)]
    i = i + 1
full = np.mean(time_full)
plt.plot(time_full_dct)
plt.plot(time_full)
plt.figure(2)
plt.plot(c, time_track[:, 0] - full)
plt.plot(c, time_track[:, 1] - full)
plt.grid()
plt.plot(c, time_temp - full)
