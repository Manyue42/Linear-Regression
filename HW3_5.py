# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:16:09 2019

@author: Manyue
"""

import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern , RationalQuadratic, ExpSineSquared , DotProduct

def func(x1, x2):
    return 0.5 * np.exp(-0.5 * ((x1 + 1.25)**2 + (x2 + 1.75)**2)) + np.exp(-0.5 * ((x1 - 2.25)**2 + (x2 - 2.65)**2))
def noisy_func(x1, x2):
    output = func(x1, x2)
    noise = np.random.normal(0, 0.1, np.shape(output))
    return output + noise



# probability of improvement
def ProbilityImprovement(mu, sigma, opt_val):
    gamma = (mu - opt_val)/sigma
    return norm.cdf(gamma)

#expected improvement
def ExpectedImprovement(mu, sigma, opt_val):
    gamma = (mu - opt_val)/sigma
    return (mu - opt_val)*norm.cdf(gamma) + sigma*norm.pdf(gamma)

def UpperCondenceBound(mu, sigma, beta):
    return mu + beta*sigma


def query(opt_val, gp):
    def obj(x):
        #do Gaussian process prediction
        mu_x, sigma_x = gp.predict(x.reshape(1, -1), return_std = True)
        
        return ExpectedImprovement(mu_x, sigma_x, opt_val)
    res = minimize(obj, np.random.randn(2))
    return res.x


res = 50
lin = np.linspace(-5, 5, res)
meshX, meshY = np.meshgrid(lin, lin)
meshpts = np.vstack((meshX.flatten(), meshY.flatten())).T

def add_subplot(gp, subplt):
    mu = gp.predict(meshpts, return_std = False)
    ax = fig.add_subplot(2, 5, subplt, projection='3d')
    ax.plot_surface(meshX, meshY, np.reshape(mu, (50, 50)),
                    rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    
if __name__ == '__main__':
    true_y = func(meshX, meshY)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(meshX, meshY, true_y, rstride=1, cstride=1, cmap=cm.jet,
                    linewidth=0, antialiased=False)
    plt.title('True_function')
    plt.show()
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
 
## part c) Initialize 4 random points and evaluate the noisy function at these points

xi = np.random.randn(4, 2)
yi = noisy_func(xi[:, 0], xi[:, 1])
    
## part d)initialize the Guassian process regressor with a kernel 

gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0))
    
    
for i in range(10):
    gp.fit(xi, yi)
    
    #find the current optimal value and its location
    opt_val = np.max(yi)
    opt_x = np.where(yi==opt_val)
    
    print('Best value: ', opt_val)
    print('at ', opt_x)
    
    next_x = query(opt_val, gp)
    
    #add next_x to the list of data points
    xi = np.append(xi, [next_x], axis=0)
    
    next_y = noisy_func(xi[-1][0], xi[-1][1]).reshape(1)
    
    #add next_y to the list of observations
    yi = np.append(yi, next_y)
    
    add_subplot(gp, i+1)
    
plt.show()
    