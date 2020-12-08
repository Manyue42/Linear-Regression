# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:43:05 2019

@author: Manyue
"""

import numpy as np
import matplotlib.pyplot as plt

T = 1000
K = 10
# Q, an array representing the action-values
# N, an array representing the number of times different actions have been taken,
# t, the total number of actions taken thus far
#  policy-specific parameter
def e_greedy(Q, N, t, e):
    random_n = np.random.random()
    if e > random_n:
        return np.random.randint(K)
    else:
        return np.argmax(Q)
    
    
def UCB(Q, N, t, c):
    max_upper_B = 0
    for i in Q:
        i = int(i)
        if (N[i] > 0):
            upper_B = Q[i] + c*np.sqrt((np.log(t))/N[i])
  
        else:
            upper_B = c
            
        if upper_B > max_upper_B:
            max_upper_B = upper_B
        return np.argmax(max_upper_B)
    
    
def test_run(policy, param):
    true_means = np.random.normal(0, 1, 10)
    reward = np.zeros(T+1)
    Q = np.zeros(10)
    
    # ..
    N = np.zeros(K)
    Q = np.zeros(K)
    t = 0
    # ..
    for i in range (T + 1):
        action = policy(Q, N, t, param)
        r = np.random.normal(true_means[action], 1)
        N[action]  += 1
        Q[action] += (1/N[action]) * (r - Q[action])
        reward[t] = r
        t += 1
    
    return reward


def main():
    ave_g = np.zeros(T+1)
    ave_eg = np.zeros(T+1)
    ave_ucb = np.zeros(T+1)
    for i in range(2000):
        g = test_run(e_greedy, 0.0)
        eg = test_run(e_greedy, 0.05  )
        ucb = test_run(UCB, 0.2  )
        ave_g  += (g - ave_g) / (i +1)
        ave_eg += (eg - ave_eg) / (i +1)
        ave_ucb += (ucb - ave_ucb) / (i +1)
        
    t = np.arange(T+1)
    plt.plot(t, ave_g, 'b-', t, ave_eg, 'r-', t, ave_ucb, 'g-')
    plt.show()
    
if __name__ == '__main__':
    main()
    
    

    
#https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/
#https://www.datahubbs.com/multi_armed_bandits_reinforcement_learning_1/