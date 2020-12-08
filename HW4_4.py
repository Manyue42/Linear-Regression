# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 00:50:19 2019

@author: Manyue
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as LA
from matplotlib.patches import Ellipse
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import multivariate_normal

K =3
NUM_DATAPTS = 150

X, y = make_blobs(n_samples = NUM_DATAPTS, centers = K, shuffle = False, 
                  random_state = 0, cluster_std = 0.6)
g1 = np.asarray([[2.0, 0], [-0.9, 1]])
g2 = np.asarray([[1.4, 0], [0.5, 0.7]])
mean1 = np.mean(X[:int(NUM_DATAPTS/K)])
mean2 = np.mean(X[int(NUM_DATAPTS/K):2 * int(NUM_DATAPTS/K)])
X[:int(NUM_DATAPTS/K)] = np.einsum('nj, ij -> ni', 
              X[:int(NUM_DATAPTS/K)] - mean1, g1) + mean1
X[int(NUM_DATAPTS/K):2 * int(NUM_DATAPTS/K)] = np.einsum('nj, ij -> ni',
              X[int(NUM_DATAPTS/K):2 * int(NUM_DATAPTS/K)] - mean2, g2) + mean2
X[:,1] -= 4


# (a)
pi = np.array([1/2, 1/2])
mu = np.array([mean1, mean2])
cov = np.cov(X)

n,d = X.shape

def E_step():
    gamma = np.zeros((NUM_DATAPTS, K))
    num_data = len(X)
    num_cluster = len(mu)
    
    for i in range(num_data):
        for k in range(num_cluster):
            gamma[i, k] = pi[i] * multivariate_normal.pdf(X[i], 
                 mean = mu[k], cov = cov[k])
    gamma = gamma / gamma.sum(axis=1)[:, np.newaxis]
    return gamma


def M_step(gamma):
    count = np.sum(gamma, axis=0)
    return count




def plot_result(gamma=None):
    ax = plt.subplot(111, aspect = 'equal')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.scatter(X[:, 0], X[:, 1], c = gamma, s = 50, cmap = None)
    
    for k in range(K):
        l, v = LA.eig(cov[k])
        theta = np.arctan(v[1,0] / v[0, 0])
        
        e = Ellipse((mu[k, 0], mu[k, 1]), 6*l[0], 6*l[1],
                     theta * 180 / np.pi)
        e.set_alpha(0.5)
        ax.add_artist(e)
        
    plt.show()

#
#if __name__ == '__main__':
#    ll = 0
#    n = len(X)
#    k = len(pi)
#    m = len(mu)
#    num_dim = len(X[0])
#    for d in X:
#       Z = np.zeros(m)
#       for k in range(m):
#           dealta = np.array(d) - mu[k]
#           exp_term = np.dot(delta.T, np.dot(np.linalg.inv(cov[k]),
#                                             delta))
#           
#           Z[k] += np.log(pi[k])
#           Z[k] -+ 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(cov[k]))
#           + exp_term)
#    ll += np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))
#    
#    plot_result(gamma)
        