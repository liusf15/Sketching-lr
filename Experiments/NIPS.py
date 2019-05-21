import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import log as log
from numpy import sqrt as sqrt
import scipy
import pandas as pd
import time


def sketching(data, r, sketch_name):
    if sketch_name == 'gaus':
        return gaussian_projection(data, r)
    elif sketch_name == 'iidp':
        return sparse_projection(data, r)
    elif sketch_name == 'haar':
        return haar_projection(data, r)
    elif sketch_name == 'hada':
        return hadamard_projection(data, r)
    elif sketch_name == 'unif':
        return uniform_sampling(data, r)
    else:
        print("invalid sketching method")
        return 0


n = 2000
gamma = 0.05
p = int(n * gamma)
np.random.seed(97013)
X = np.random.randn(n, p)
np.random.seed(130)
beta = np.random.rand(p, 1)
c = np.linspace(0.1, 1, 20)
rep = 10
type_seq = ['gaus', 'iidp', 'haar', 'hada', 'unif']
result = np.zeros((20 * 5, 5))  # columns=['r', 've', 've_low', 've_high', 'type'])
result_sd = np.zeros((20 * 5, 4)) # columns=['r', 've', 'std' 'type'])

for i in range(20):
    print(i)
    xi = c[i]
    r = int(n * xi)
    temp_result = np.zeros((rep, 5))
    for k in range(rep):
        data = DATA(X=X, beta=beta, n=n, p=p)
        for j in range(5):
            sketch_name = type_seq[j]
            temp_result[k, j] = log(sketching(data, r, sketch_name)[0])  ## log or not
    for u in range(5):
        result_sd[i + u * 20, :] = [xi, np.mean(temp_result[:, u]), np.std(temp_result[:, u]), u]

result_df_sd = pd.DataFrame(result_sd, columns=['r', 've', 'sd', 'type'])
result_df_sd.to_csv("/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/NIPS/gamma_{}_log.csv".format(gamma))


### plots
res_1 = np.array(pd.read_csv("/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/NIPS/gamma_0.05.csv"))
res_2 = np.array(pd.read_csv("/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/NIPS/gamma_0.4.csv"))
res_log_1 = np.array(pd.read_csv('/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/NIPS/gamma_0.05_log.csv'))
res_log_2 = np.array(pd.read_csv("/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/NIPS/gamma_0.4_log.csv"))

label_seq = ['Gaussian', "iid", "Haar", "Hadamard", "Sampling"]
linestyle_seq = ['-', '--', ':', '-.', ':']
fmt_seq = ['v', 'o', '^', 'x', '+']
color_seq = ['orange', 'gold', 'cyan', 'aqua', 'lightblue']

plt.figure(0, figsize=(20, 8))
p1 = plt.subplot(121)
p1.cla()
for i in range(5):
    p1.errorbar(res_log_1[20 * i: 20 * (i + 1), 1], res_log_1[20 * i: 20 * (i + 1), 2], res_log_1[20 * i: 20 * (i + 1), 3],
                color=color_seq[i], capsize=3, label=label_seq[i], ls=linestyle_seq[i], fmt=fmt_seq[i], lw=2)
d_1 = np.linspace(0.1, 1, 20)
gamma_1 = 0.05
p1.plot(d_1, log(1 + (1 - gamma_1) / (d_1 - gamma_1)), lw=4, color='orangered')
p1.plot(d_1, log((1 - gamma_1) / (d_1 - gamma_1)), lw=4, color='blue')
p1.set_ylabel(r"$log(VE)$", fontsize=20)
p1.set_xlabel(r"$r/n$", fontsize=20)
p1.set_title(r'$\gamma=0.05$', fontsize=20)
p1.grid(linestyle='dotted')

p2 = plt.subplot(122)
p2.cla()
for i in range(5):
    p2.errorbar(res_log_2[20 * i: 20 * (i + 1), 1], res_log_2[20 * i: 20 * (i + 1), 2], res_log_2[20 * i: 20 * (i + 1), 3],
                color=color_seq[i], capsize=3, label=label_seq[i], ls=linestyle_seq[i], fmt=fmt_seq[i], lw=2)
gamma_2 = 0.4
d_2 = np.linspace(0.45, 1, 20)
p2.plot(d_2, log(1 + (1 - gamma_2) / (d_2 - gamma_2)), lw=4, color='orange', label=r'$1+\frac{n-p}{r-p}$')
p2.plot(d_2, log((1 - gamma_2) / (d_2 - gamma_2)), lw=4, color='blue', label=r'$\frac{n-p}{r-p}$')
p2.set_xlabel(r"$r/n$", fontsize=20)
p2.set_title(r'$\gamma=0.4$', fontsize=20)
p2.grid(linestyle='dotted')
p2.legend(fontsize=20)
plt.subplots_adjust(wspace=0.08, hspace=0)
plt.savefig('/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/NIPS/log_simu.png')

### empirical
msd_gaus_re = np.zeros((10, 4))
msd_gaus_oe = np.zeros((10, 4))
msd_hada_re = np.zeros((10, 4))
msd_hada_oe = np.zeros((10, 4))
msd_unif_re = np.zeros((10, 4))
msd_unif_oe = np.zeros((10, 4))
for i in range(10):
    gaus = np.array(
        pd.read_csv(
            '/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/R1/intermediate_results/msd_gaus_{}.csv'.format(
                i)))[:, 1:]
    msd_gaus_re[i, :] = [np.mean(gaus[:, 0]), np.quantile(gaus[:, 0], 0.05), np.quantile(gaus[:, 0], 0.95), np.std(gaus[:, 0])]
    msd_gaus_oe[i, :] = [np.mean(gaus[:, 1]), np.quantile(gaus[:, 1], 0.05), np.quantile(gaus[:, 1], 0.95), np.std(gaus[:, 1])]

    hada = np.array(
        pd.read_csv(
            '/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/R1/intermediate_results/msd_hada_{}.csv'.format(
                i)))[:, 1:]
    msd_hada_re[i, :] = [np.mean(hada[:, 0]), np.quantile(hada[:, 0], 0.05), np.quantile(hada[:, 0], 0.95), np.std(hada[:, 0])]
    msd_hada_oe[i, :] = [np.mean(hada[:, 1]), np.quantile(hada[:, 1], 0.05), np.quantile(hada[:, 1], 0.95), np.std(hada[:, 1])]

    unif = np.array(
        pd.read_csv(
            '/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/R1/intermediate_results/msd_unif_{}.csv'.format(
                i)))[:, 1:]
    msd_unif_re[i, :] = [np.mean(unif[:, 0]), np.quantile(unif[:, 0], 0.05), np.quantile(unif[:, 0], 0.95), np.std(unif[:, 0])]
    msd_unif_oe[i, :] = [np.mean(unif[:, 1]), np.quantile(unif[:, 1], 0.05), np.quantile(unif[:, 1], 0.95), np.std(unif[:, 1])]

flt_gaus_re = np.zeros((10, 4))
flt_gaus_oe = np.zeros((10, 4))
flt_hada_re = np.zeros((10, 4))
flt_hada_oe = np.zeros((10, 4))
flt_unif_re = np.zeros((10, 4))
flt_unif_oe = np.zeros((10, 4))
for i in range(10):
    gaus = np.array(
        pd.read_csv(
            '/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/R1/intermediate_results/flt_gaus_{}.csv'.format(
                i)))[:, 1:]
    flt_gaus_re[i, :] = [np.mean(gaus[:, 0]), np.quantile(gaus[:, 0], 0.05), np.quantile(gaus[:, 0], 0.95), np.std(gaus[:, 0])]
    flt_gaus_oe[i, :] = [np.mean(gaus[:, 1]), np.quantile(gaus[:, 1], 0.05), np.quantile(gaus[:, 1], 0.95), np.std(gaus[:, 1])]

    hada = np.array(
        pd.read_csv(
            '/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/R1/intermediate_results/flt_hada_{}.csv'.format(
                i)))[:, 1:]
    flt_hada_re[i, :] = [np.mean(hada[:, 0]), np.quantile(hada[:, 0], 0.05), np.quantile(hada[:, 0], 0.95), np.std(hada[:, 0])]
    flt_hada_oe[i, :] = [np.mean(hada[:, 1]), np.quantile(hada[:, 1], 0.05), np.quantile(hada[:, 1], 0.95), np.std(hada[:, 1])]

    unif = np.array(
        pd.read_csv(
            '/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/R1/intermediate_results/flt_unif_{}.csv'.format(
                i)))[:, 1:]
    flt_unif_re[i, :] = [np.mean(unif[:, 0]), np.quantile(unif[:, 0], 0.05), np.quantile(unif[:, 0], 0.95), np.std(unif[:, 0])]
    flt_unif_oe[i, :] = [np.mean(unif[:, 1]), np.quantile(unif[:, 1], 0.05), np.quantile(unif[:, 1], 0.95), np.std(unif[:, 1])]
msd_gaus_re = log(msd_gaus_re)
msd_hada_re = log(msd_hada_re)
msd_unif_re = log(msd_unif_re)
flt_gaus_re = log(flt_gaus_re)
flt_hada_re = log(flt_hada_re)
flt_unif_re = log(flt_unif_re)
### plots
# d = np.linspace(c[1], 0.002, 500)

plt.figure(0, figsize=(20, 8))
p1 = plt.subplot(121)
p1.cla()
m = 515344
n = m - 10000
p = 90
gamma = p / n
c = np.linspace(0.0003, 0.002, 10)
err = [msd_gaus_re[1:, 0] - msd_gaus_re[1:, 1],
       msd_gaus_re[1:, 2] - msd_gaus_re[1:, 0]]
p1.errorbar(c[1:], msd_gaus_re[1:, 0], err, capsize=2, label='Gaussian', ls='--', fmt='v', lw=2, color='orange')
err = [msd_hada_re[1:, 0] - msd_hada_re[1:, 1],
       msd_hada_re[1:, 2] - msd_hada_re[1:, 0]]
p1.errorbar(c[1:], msd_hada_re[1:, 0], err, capsize=2, label='Haar', ls='-.', fmt='o', lw=2, color='aqua')
err = [msd_unif_re[1:, 0] - msd_unif_re[1:, 1],
       msd_unif_re[1:, 2] - msd_unif_re[1:, 0]]
p1.errorbar(c[1:], msd_unif_re[1:, 0], err, capsize=2, label='Sampling', ls=':', fmt='o', lw=2, color='turquoise')
p1.plot(c[1:], log(c[1:] / (c[1:] - gamma)), label=r'$\frac{r}{r-p}$', ls='-', lw=4, color='orangered')
p1.plot(c[1:], log(c[1:] / (c[1:] - gamma) - gamma / (1 - gamma)), label=r'$\frac{r}{r-p}-\frac{p}{n-p}$', ls='--', lw=3, color='blue')
p1.set_title('MSD', fontsize=20)
p1.set_xlabel(r'$r/n$', fontsize=20)
p1.set_ylabel(r'$log(RE)$', fontsize=20)
p1.grid(linestyle='dotted')

p2 = plt.subplot(122)
p2.cla()
m = 60449
n = m - 10000
p = 21
gamma = p / n
c = np.linspace(0.0006, 0.005, 10)
err = [flt_gaus_re[1:, 0] - flt_gaus_re[1:, 1],
       flt_gaus_re[1:, 2] - flt_gaus_re[1:, 0]]
p2.errorbar(c[1:], flt_gaus_re[1:, 0], err, capsize=2, label='Gaussian', ls='--', fmt='v', lw=2, color='orange')
err = [flt_hada_re[1:, 0] - flt_hada_re[1:, 1],
       flt_hada_re[1:, 2] - flt_hada_re[1:, 0]]
p2.errorbar(c[1:], flt_hada_re[1:, 0], err, capsize=2, label='Haar', ls='-.', fmt='o', lw=2, color='aqua')
err = [flt_unif_re[1:, 0] - flt_unif_re[1:, 1],
       flt_unif_re[1:, 2] - flt_unif_re[1:, 0]]
p2.errorbar(c[1:], flt_unif_re[1:, 0], err, capsize=2, label='Sampling', ls=':', fmt='o', lw=2, color='turquoise')
p2.plot(c[1:], log(c[1:] / (c[1:] - gamma)), label=r'$\frac{r}{r-p}$', ls='-', lw=4, color='orangered')
p2.plot(c[1:], log(c[1:] / (c[1:] - gamma) - gamma / (1 - gamma)), label=r'$\frac{r}{r-p}-\frac{p}{n-p}$', ls='--', lw=3, color='blue')
p2.legend(fontsize=20, loc='upper right')
p2.set_title('Flight dataset', fontsize=20)
p2.set_xlabel(r'$r/n$', fontsize=20)
p2.grid(linestyle='dotted')
plt.ylim([0, 2])
plt.subplots_adjust(wspace=0.08, hspace=0)
plt.subplots_adjust(wspace=0.08, hspace=0)
plt.savefig('/Users/sifanliu/Dropbox/Random Projection/Experiments/plots/NIPS/log_empi.png')