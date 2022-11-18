#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    greedy_action_index = np.argmax(Q[state])
    prob = [epsilon/nA] * nA
    prob[greedy_action_index] = 1 - epsilon + (epsilon/nA)
    action = np.random.choice(range(nA), p=prob)
    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for _ in range(1, n_episodes + 1):

        # define decaying epsilon
        epsilon = 0.99 * epsilon

        # initialize the environment 
        s = env.reset()     # s --> state
        
        # get an action from policy
        a = epsilon_greedy(Q, s, nA, epsilon)     # a --> action

        # loop for each step of episode
        while True:
            # return a new state, reward and done
            s_next, r, terminated, truncated, info = env.step(a)    # r --> reward
            if terminated or truncated:
                Q[s_next][a] = 0
                break

            # get next action
            a_next = epsilon_greedy(Q, s_next, nA, epsilon)

            # TD update
            td_target = r + gamma * Q[s_next][a_next]   # td_target
            td_error = td_target - Q[s][a]              # td_error

            # new Q
            Q[s][a] = Q[s][a] + alpha * (td_error)
            
            # update state
            s = s_next

            # update action
            a = a_next
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for _ in range(1, n_episodes + 1):
        # print(i)

        # initialize the environment 
        s = env.reset()
        
        # loop for each step of episode
        while True:
            # get an action from policy
            a = epsilon_greedy(Q,s,nA,epsilon)

            # return a new state, reward and done
            s_next, r, terminated, truncated, info = env.step(a)    # r --> reward
            if terminated or truncated:
                Q[s_next][a] = 0
                break

            # TD update
            td_target = r + gamma * max(Q[s_next])   # td_target with best Q
            td_error = td_target - Q[s][a]                    # td_error

            # new Q
            Q[s][a] = Q[s][a] + alpha * (td_error)
            
            # update state
            s = s_next
    ############################
    return Q
