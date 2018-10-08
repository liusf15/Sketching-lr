"""
    Compare our results with Raskutti & Mahoney' s paper:
    A statistical perspective on randomized sketching for ordinary least-squares
    Plots in Section 5.10.1, Figure 11
"""

import matplotlib.pyplot as plt
import numpy as np
from Data import DATA
from Sketching_methods import gaussian_projection
from Sketching_methods import hadamard_projection

n = 2000
p = 100
gamma = p / n
np.random.seed(130)
X = np.random.randn(n, p)
beta = np.random.rand(p, 1)
rep = 50
c = np.linspace(0.1, 1, 20)

# Gaussian projection
track = np.empty((20, 4))
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpro = np.empty((rep, 4))
    for k in range(rep):
        data = DATA(type='Gaussian', n=n, p=p, X=X, beta=beta)
        vpro[k, :] = gaussian_projection(data, r)
    track[i, :] = np.mean(vpro, axis=0)
    i = i + 1

# Hadamard projction
track_hadamard = np.empty((20, 4))
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    vpro = np.empty((rep, 4))
    for k in range(rep):
        data = DATA(type='Gaussian', n=n, p=p, X=X, beta=beta)
        vpro[k, :] = hadamard_projection(data, r)
    track_hadamard[i, :] = np.mean(vpro, axis=0)
    i = i + 1

# Figure 11
gamma = p / n
d = np.linspace(0.15, 1, 500)
plt.figure(0, figsize=(10, 8))
p11 = plt.subplot(221)
p11.scatter(c[1:], np.log(track[1:, 1]), label='Simulation')
p11.plot(d, np.log(1 + (1 - gamma) / (d - gamma)), label=r'1+$\frac{n-p}{r-p}$', ls='--')
p11.plot(d, np.log(44 * (1 + 1 / d)), label=r'$44(1+\frac{n}{r})$', ls=':')
p11.grid(linestyle='dotted')
p11.set_ylabel('log PE', fontsize=13)
p11.set_title('Gaussian Projection', fontsize=13)
p11.legend()

p12 = plt.subplot(223)
p12.scatter(c[1:], np.log(track[1:, 2]), label='Simulation')
p12.plot(d, np.log(d / (d - gamma)), label=r'$\frac{r}{r-p}$', ls='--')
p12.plot(d, np.log(1 + 44 * gamma / d), label=r'$1+44\frac{p}{r}$', ls=':')
p12.grid(linestyle='dotted')
p12.set_ylabel('log RE', fontsize=13)
p12.legend()

p21 = plt.subplot(222)
p21.scatter(c[1:], np.log(track_hadamard[1:, 1]), label='Simulation')
p21.plot(d, np.log((1 - gamma) / (d - gamma)), label=r'$\frac{n-p}{r-p}$', ls='--')
p21.plot(d, np.log(1+40*np.log(n*p)*(1+gamma/d)), label=r'$1+40\log(np)(1+\frac{p}{r})$', ls=':')
p21.grid(linestyle='dotted')
p21.set_title('Hadamard Projection', fontsize=13)
p21.legend()


p22 = plt.subplot(224)
p22.scatter(c[1:], np.log(track_hadamard[1:, 2]), label='Simulation')
p22.plot(d, np.log(d/ (d - gamma)-gamma/(1-gamma)), label=r'$\frac{r}{r-p}-\frac{p}{n-p}$', ls='--')
p22.plot(d, np.log(40*np.log(n*p)*(1+1/d)), label=r'$40\log(np)(1+\frac{n}{r})$', ls=':')
p22.grid(linestyle='dotted')
p22.set_xlabel(r'$r/n$', fontsize=13)
p22.legend()
plt.subplots_adjust(hspace=.01)
plt.savefig('/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/comparing_results.png')