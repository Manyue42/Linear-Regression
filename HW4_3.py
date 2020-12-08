# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:13:17 2019

@author: Manyue
"""

import numpy as np
from sklearn import decomposition
from sklearn import datasets

X = datasets.load_diabetes().data

##(a) Write code to print the matrix V that will be used to transform the dataset, and print all the
##singular values.
pca = decomposition.PCA(n_components='mle')
pca.fit(X)
#decomposition.PCA(copy=True, iterated_power = 'auto', n_components = 'mle',
#                  random_state=None, svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_)
print(pca.singular_values_)


#b
pca = decomposition.PCA(n_components=3)
pca.fit(X)
decomposition.PCA(copy=True, iterated_power = 'auto', n_components = 3,
                  random_state=None, svd_solver='auto', tol=0.0, whiten=False)

print(pca.components_)
