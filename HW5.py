# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:31:56 2019

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
    def init(self):
        self.shape = [5, 5]
        
        nS = np.prod(self.shape)    # total number of states
        nA = 4  # total numer of actions per state
        
        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]
        
        P = {}
        grid = np.arange(nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            P[s] = {a: [] for a in range(nA)}
            is_done = lambda s: s == GOAL
            
            if is_done(s):
                reward = 0.0
                
            elif s == SNAKE1 or s == SNAKE2:
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
                ns_right = s if x==(MAX_X-1) else s+1
                ns_down = s if y==(MAX_Y-1) else s+MAX_X
                ns_left = s if x==0 else s-1
                
                P[s][UP] = [(1-(2*eps), ns_up, reward, is_done(ns_up)),
                 (eps, ns_right, reward, is_done(ns_right)),
                 (eps, ns_left, reward, is_done(ns_left))]
                
                P[s][RIGHT] =  [(1-(2*eps), ns_right, reward, is_done(ns_right)),
                 (eps, ns_up, reward, is_done(ns_up)),
                 (eps, ns_down, reward, is_done(ns_down))]
                
                P[s][DOWN] =  [(1-(2*eps), ns_down, reward, is_done(ns_down)),
                 (eps, ns_right, reward, is_done(ns_right)),
                 (eps, ns_left, reward, is_done(ns_left))]
                
                P[s][LEFT] =  [(1-(2*eps), ns_left, reward, is_done(ns_left)),
                 (eps, ns_up, reward, is_done(ns_up)),
                 (eps, ns_down, reward, is_done(ns_down))]
                
            it.iternext()
            
        isd = np.zeros(nS)
        isd[START] = 1.0
        
        self.P = P
        
        super(Robot_vs_snakes_world, self).init(nS, nA, P, isd)
        
    def _render(self):
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
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
#env.s #Robot location
#env.step("DIR") #Command to move. DIR is UP, DOWN, LEFT, RIGHT
#env.p[state][action] #Returns a list of tuples
        
        
def value_iteration(env):
    theta = 0.0001
    gamma = 1   #discount factor
    def one_step_lookahead(state, V):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            for next_step in env.P[state][a]:
                p, s_, r, _ = next_step
                q_sa[a] += (p * (r + gamma * V[s_]))
        return q_sa
    
    V = np.zeros(env.nS)
    while True:
        delta = 0 #Stopping condition
        for s in range(env.nS):
            A = one_step_lookahead(s,V)
            best_action_val = np.max(A)
            delta = max(delta, np.abs(best_action_val-V[s]))    #Calculate delta across all states seen so far
            V[s] = best_action_val  #Update the value function
        if delta < theta:
            break
        
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)  #Find best action
        policy[s, best_action] = 1.0    #Take best action
#    print (policy, V)
    return policy, V
#b) Use policy function obtained to navigat robot through the room at each time step
#env._render()
#if name == 'main':
#    env = Robot_vs_snakes_world()
#    steps = 0
#    while env.s != GOAL:    # If position of robot R not at goal G iterate
#        steps += 1
#        print ("Step: " + str(steps))
#        policy, V = value_iteration(env)
#        env.step(np.argmax(policy[env.s]))
#        print (env.step(np.argmax(policy[env.s])))
#        env._render()   #Prints location of robot at each step
#        
        
env = Robot_vs_snakes_world()
while env.s != GOAL:    # If position of robot R not at goal G iterate
    policy, V = value_iteration(env)
    env.step(np.argmax(policy[env.s]))
    env._render()   #Prints location of robot at each step