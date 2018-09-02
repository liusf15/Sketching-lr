import matplotlib.pyplot as plt
import numpy as np

from Data import DATA
from Sketching_methods import gaussian_projection
from Sketching_methods import hadamard_projection

# Parameters
c = np.linspace(0.1, 1, 20)
rep = 50
track_msd = np.zeros((20, 2))
track_flight = np.zeros((20, 2))
# MSD
n = 2000
np.random.seed(13)
data = DATA(type='MSD', n=n, p=90)
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    for k in range(rep):
        track_msd[i, 0] = track_msd[i, 0] + gaussian_projection(data, r)[2]
        track_msd[i, 1] = track_msd[i, 1] + hadamard_projection(data, r)[2]
    i = i + 1
track_msd = track_msd / rep

# nycflight
np.random.seed(130)
data = DATA(type='nycflight', n=2000, p=21)
i = 0
for xi in c:
    print(i)
    r = int(n * xi)
    for k in range(rep):
        track_flight[i, 0] = track_flight[i, 0] + gaussian_projection(data, r)[2]
        track_flight[i, 1] = track_flight[i, 1] + hadamard_projection(data, r)[2]
    i = i + 1
track_flight = track_flight / rep

# PLOTS
gamma = 90 / 2000
d = np.linspace(0.15, 1, 500)
plt.figure(1, figsize=(12, 6))
plt.subplot(121)
plt.scatter(c[1:], track_msd[1:, 0], label='Gaussian projection')
plt.scatter(c[1:], track_msd[1:, 1], marker='s', label='Hadamard projection')
plt.plot(d, d / (d - gamma), ls='--', label=r'Theory: $\frac{r}{r-p}$')
plt.plot(d, d / (d - gamma) - gamma / (1 - gamma), ls=':', label=r'Theory: $\frac{r}{r-p}-\frac{p}{n-p}$')
# plt.legend()
plt.xlabel(r'$r/n$', fontsize=13)
plt.ylabel('Residual Efficiency')
plt.title('MSD Dataset')
plt.grid()

plt.subplot(122)
gamma = 21 / 2000
plt.scatter(c[1:], track_flight[1:, 0], label='Gaussian projection')
plt.scatter(c[1:], track_flight[1:, 1], marker='s', label='Hadamard projection')
plt.plot(d, d / (d - gamma), ls='--', label=r'Theory: $\frac{r}{r-p}$')
plt.plot(d, d / (d - gamma) - gamma / (1 - gamma), ls=':', label=r'Theory: $\frac{r}{r-p}-\frac{p}{n-p}$')
plt.legend()
plt.xlabel(r'$r/n$', fontsize=13)
plt.ylabel('')
plt.title('nycflights13 Dataset')
plt.grid(linestyle='dotted')
plt.subplots_adjust(.02)

plt.savefig('/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/front_figure.png')
