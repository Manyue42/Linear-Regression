# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:34:41 2019

@author: Manyue
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs ,make_circles

def plot_svc_decision(model, ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)        
    ## plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], 
               alpha = 0.5, linestyles=['--', '-', '--'])    
    ##plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
               s = 300, linewidth = 1, edgecolors = 'black', facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
##plot first dataset
X,y = make_circles(100, factor=.1, noise=.1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='seismic')
clf=SVC(kernel='rbf', C=1000)
model1=clf.fit(X,y)
plot_svc_decision(model1, ax1)  

##generate and plot the second dataset
X,y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.0)
fig2 = plt.figure(figsize=(16,6))
fig2.subplots_adjust(left = 0.0625, right = 0.95, wspace = 0.1)

ax2 = fig2.add_subplot(121)
ax2.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='seismic')

ax3 = fig2.add_subplot(122)
ax3.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='seismic')

clf=SVC(kernel='linear', C=0.1)
model2=clf.fit(X,y)
plot_svc_decision(model2, ax2)  

clf=SVC(kernel='linear', C=500)
model3=clf.fit(X,y)
plot_svc_decision(model3, ax3)  


    