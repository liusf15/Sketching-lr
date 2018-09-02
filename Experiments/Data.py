"""
    Parameters, data, full OLS
"""

import numpy as np
import scipy.linalg
import pandas as pd


class DATA:
    n = 1
    p = 1
    gamma = 0.05
    beta = []
    epsilon = []
    X = []
    Y = []
    beta_full = []
    v_full = 1
    p_full = 1
    r_full = 1

    def __init__(self, type, n, X=None, beta=None, p=50, df=1):
        self.n = n
        self.p = p
        self.gamma = p / n
        if beta is not None:
            self.beta = np.random.rand(p, 1)
        self.epsilon = np.random.randn(n, 1)
        if X is not None:
            self.X = X
            self.Y = self.X @ self.beta + self.epsilon
            self.beta_full = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
            self.v_full = np.linalg.norm(self.beta - self.beta_full) ** 2
            self.p_full = np.linalg.norm(self.X @ self.beta - self.X @ self.beta_full) ** 2
            self.r_full = np.linalg.norm(self.Y - self.X @ self.beta_full) ** 2
        elif type == 'Gaussian':
            self.X = np.random.randn(n, p)
            self.Y = self.X @ self.beta + self.epsilon
            self.beta_full = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
            self.v_full = np.linalg.norm(self.beta - self.beta_full) ** 2
            self.p_full = np.linalg.norm(self.X @ self.beta - self.X @ self.beta_full) ** 2
            self.r_full = np.linalg.norm(self.Y - self.X @ self.beta_full) ** 2
        elif type == 'elliptical':
            self.X = np.random.randn(n, p)
            d1 = 1
            d2 = 3
            W = np.random.choice([-d2, -d1, d1, d2], size=n)
            self.X = np.array([self.X[i, :] * W[i] for i in range(n)])
            self.Y = self.X @ self.beta + self.epsilon
            self.beta_full = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
            self.v_full = np.linalg.norm(self.beta - self.beta_full) ** 2
            self.p_full = np.linalg.norm(self.X @ self.beta - self.X @ self.beta_full) ** 2
            self.r_full = np.linalg.norm(self.Y - self.X @ self.beta_full) ** 2
        elif type == 't':
            sigma = 4 * scipy.linalg.toeplitz(0.5 ** np.linspace(1, p, p))
            u = np.random.chisquare(df, n).reshape(n, 1)
            self.X = np.divide(np.random.multivariate_normal(np.zeros(p), sigma, n), u / np.sqrt(df))
            self.Y = self.X @ self.beta + self.epsilon
            self.beta_full = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
            self.v_full = np.linalg.norm(self.beta - self.beta_full) ** 2
            self.p_full = np.linalg.norm(self.X @ self.beta - self.X @ self.beta_full) ** 2
            self.r_full = np.linalg.norm(self.Y - self.X @ self.beta_full) ** 2
        elif type == 'MSD':
            msd = pd.read_table("/Users/sifanliu/Dropbox/Random Projection/Experiments/real_data/YearPredictionMSD.txt",
                                delimiter=',', nrows=10000).as_matrix()
            idx = np.random.choice(10000, n, replace=False)
            self.p = 90
            self.gamma = self.p / self.n
            self.X = msd[idx, 1:]
            self.Y = msd[idx, 0].reshape((n, 1))
            self.beta_full = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
            self.r_full = np.linalg.norm(self.Y - self.X @ self.beta_full) ** 2
        elif type == 'nycflight':
            flt = pd.read_csv(
                '/Users/sifanliu/Dropbox/Random Projection/Experiments/real_data/nycflight/nycflight.csv').as_matrix()
            idx = np.random.choice(10000, n, replace=False)
            self.p = 21
            self.gamma = self.p / self.n
            self.Y = flt[idx, 1].reshape((n, 1))  # the first column is the response
            self.X = flt[idx, 2:]
            self.beta_full = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
            self.r_full = np.linalg.norm(self.Y - self.X @ self.beta_full) ** 2
