"""
    Test our results for Haar and Hadamard projection
    Plots in Section 3.2, Figure 4
"""

import matplotlib.pyplot as plt
import numpy as np
from Data import DATA
from Sketching_methods import haar_projection
from Sketching_methods import hadamard_projection

n = 2000
p = 100
gamma = p / n
np.random.seed(130)
X = np.random.randn(n, p)
beta = np.random.rand(p, 1)
rep = 50
c = np.linspace(0.1, 1, 20)
track = np.empty((20, 4))

# Haar projection
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpro = np.empty((rep, 4))
    for k in range(rep):
        data = DATA(type='Gaussian', n=n, p=p, X=X, beta=beta)
        vpro[k, :] = haar_projection(data, r)
    track[i, :] = np.mean(vpro, axis=0)
    i = i + 1

# Figure 5
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
plt.savefig('plots/Haar.png')

# Hadamard projection
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpro = np.empty((rep, 4))
    for k in range(rep):
        data = DATA(type='Gaussian', n=n, p=p, X=X, beta=beta)
        vpro[k, :] = hadamard_projection(data, r)
    track[i, :] = np.mean(vpro, axis=0)
    i = i + 1

# Figure 6
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
plt.savefig('plots/Hadamard.png')
