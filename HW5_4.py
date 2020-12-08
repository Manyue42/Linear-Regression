# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 00:15:36 2019

@author: Manyue
"""

import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

GOAL = 4
START = 20
SNAKE1 = 7
SNAKE2 = 17

eps = 0.25

class Robot_vs_snakes_world(discrete.DiscreteEnv):
    def __init__(self):
        self.shape = [5,5]
        nS = np.prod(self.shape)
        nA = 4
        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]
        
        P = {}
        grid = np.arange(nS).reshape(self.shape)
        it = np.nditer(grid, flags = ['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a : [] for a in range(nA)}
            is_done = lambda s: s == GOAL
            
            if is_done(s):
                reward = 0.0
            elif s == SNAKE1 or s ==SNAKE2:
                reward = -15.0
            else:
                reward = -1.0
            
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y==0 else s-MAX_X
                ns_right = s if x==(MAX_X - 1) else s+1
                ns_down = s if y==(MAX_Y - 1) else s+MAX_X
                ns_left = s if x==0 else s-1
                
                P[s][UP] = [(1 - (2*eps), ns_up, reward, is_done(ns_up)),
                 (eps, ns_right, reward, is_done(ns_right)),
                 (eps, ns_left, reward, is_done(ns_left))]
                
                P[s][RIGHT] = [(1 - (2*eps), ns_right, reward, is_done(ns_right)),
                 (eps, ns_up, reward, is_done(ns_up)),
                 (eps, ns_down, reward, is_done(ns_down))]
                
                P[s][DOWN] = [(1 - (2*eps), ns_down, reward, is_done(ns_down)),
                 (eps, ns_right, reward, is_done(ns_right)),
                 (eps, ns_left, reward, is_done(ns_left))]
                
                P[s][LEFT] = [(1 - (2*eps), ns_left, reward, is_done(ns_left)),
                 (eps, ns_up, reward, is_done(ns_up)),
                 (eps, ns_down, reward, is_done(ns_down))]
                
            it.iternext()
        isd = np.zeros(nS)
        isd[START] = 1.0
        
        self.P = P
        super(Robot_vs_snakes_world, self).__init__(nS, nA, P, isd)
        
    def _render(self):
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags = ['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            if self.s == s:
                output = " R "
            elif s == GOAL:
                output = " G "
            elif s == SNAKE1 or s == SNAKE2:
                output = " S "
            else:
                output = " o "
            
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()
                
            sys.stdout.write(output)
            
            if x == self.shape[1] - 1:
                sys.stdout.write("\n")
                
            it.iternext()
            
        sys.stdout.write("\n")
   
     
#env = Robot_vs_snakes_world()
#env.senv.step(DIR)
#env.p[state][action]


#This function is to return the value function, as well as the greedy policy function. Print out
#both the policy and the value function.           
def value_iteration(env):
    
    V = np.zeros(env.nS)
    #..
    theta = 0.0001
    gama = 1
    def oneStep(state, V):
        Q_sa = np.zeros(env.nA)
        for i in range(env.nA):
            for next_S in env.P[state][i]:
                p, s_, r, _ = next_S
                Q_sa[i] += (p*(r+gama *V[s_]))
        return Q_sa
    
    while True:
        dta = 0
        for a in range(env.nS):
            MY = oneStep(a, V)
            bactval = np.max(MY)
            dta = max(dta,np.abs(bactval-V[a]))
            V[a] = bactval
        if dta < theta:
            break
        
    policy = np.zeros([env.nS, env.nA])
    for a in range(env.nS):
        MY = oneStep(a,V)
        bact = np.argmax(MY)
        policy[a,bact] = 1.0
        
    return policy, V

env = Robot_vs_snakes_world()
while env.s != GOAL:
    policy, V = value_iteration(env)
    env.step(np.argmax(policy[env.s]))
    env._render()
    

                