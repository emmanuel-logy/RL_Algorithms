#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    if score >= 20:
        action = 0
    else:
        action = 1
    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an observation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    # a dict to maintain num of times states occured in episode... easy to check if state is first visit
    first_visit = defaultdict(int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for _ in range(1, n_episodes+1):
        # print("New episode . . . ")
        first_visit.clear()
        # initialize the episode
        state = env.reset()

        # generate empty episode list
        episode = []
        episode.append([state, 1, 0])     # action=1 cuz 1st move would be to hit(=1)
                                          # reward=0 cuz reward=1 only after player wins at the end of episode
        first_visit[state] += 1

        # loop until episode generation is done
        while True:
            # select an action
            action = policy(state)    # episode[-1][0] gives obs from latest event inserted in episode

            # return a reward and new state
            state_next, reward, terminated, truncated, info = env.step(action)

            # append state, action, reward to episode
            episode.append( [state,action,reward] )
            first_visit[state] += 1

            # update state to new state
            state = state_next

            # Stop loop if episode ended
            if terminated or truncated:
                break

        # loop for each step of episode, t = T-1, T-2,...,0
        t = len(episode)-1
        G = 0
        while t >= 0:
            state = episode[t][0]
            action = episode[t][1]
            reward =  episode[t][2]

            # compute G
            G = gamma * G + reward

            # unless state_t appears in states
            # current_state = episode[t][0]
            # first_time = True
            # for i in range(0,t):
            #     state = episode[i][0]
            #     if state == current_state:
            #         first_time = False

            # if first_visit:
            if first_visit[state] == 1:
                # update return_count
                returns_count[state] += 1

                # update return_sum
                returns_sum[state] += G

                # calculate average return for this state over all sampled episodes
                V[state] = returns_sum[state] / returns_count[state]

            first_visit[state] -= 1
            t -= 1
    ############################
    return V

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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    greedy_action_index = np.argmax(Q[state])
    prob = [epsilon/nA] * nA
    prob[greedy_action_index] = 1 - epsilon + (epsilon/nA)
    if any(val<0 for val in prob):
        print("Prob negative: ", epsilon)
        print(prob)
    action = np.random.choice(range(nA), p=prob)
    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(lambda: np.zeros(nA))
    returns_count = defaultdict(lambda: np.zeros(nA))
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    # a dict to maintain num of times states occured in episode... easy to check if state is first visit
    first_visit = defaultdict(lambda: np.zeros(nA))

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    decay_epsilon = True
    for _ in range(1, n_episodes+1):
        first_visit.clear()
        # define decaying epsilon
        if decay_epsilon:
            old_epsilon = epsilon
            epsilon = epsilon - (0.1 / n_episodes)
            if (epsilon < 0):
                epsilon = old_epsilon
                decay_epsilon = False

        # initialize the episode
        state = env.reset()

        # generate empty episode list
        episode = []
        action = 1  # action=1 cuz 1st move would be to hit(=1)
        reward = 0  # reward=0 cuz reward=1 only after player wins at the end of episode
        episode.append([state, action, reward])
        first_visit[state][action] += 1

        # loop until episode generation is done
        while True:

            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, state, nA, epsilon)

            # return a reward and new state
            state_next, reward, terminated, truncated, info = env.step(action)

            # append state, action, reward to episode
            episode.append( [state,action,reward] )
            first_visit[state][action] += 1

            # update state to new state
            state = state_next

            # Stop loop if episode ended
            if terminated or truncated:
                break

        # loop for each step of episode, t = T-1, T-2,...,0
        t = len(episode)-1
        G = 0
        while t >= 0:
            state = episode[t][0]
            action = episode[t][1]
            reward = episode[t][2]
            # compute G
            G = reward + gamma * G

            # unless the pair state_t, action_t appears in <state action> pair list
            if first_visit[state][action] == 1:
                # update return_count
                returns_count[state][action] += 1

                # update return_sum
                returns_sum[state][action] += G

                # calculate average return for this state over all sampled episodes
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]

            first_visit[state][action] -= 1
            t -= 1
    ############################
    return Q
