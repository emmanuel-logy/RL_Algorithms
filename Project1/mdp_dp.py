### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np
import time

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.
    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        delta = 0
        vs_old = value_function.copy()

        for s in range(0, nS):
            vs_new = 0
            for a in range(0,nA):
                vs_a = 0
                for tup in P[s][a]:
                    tp_next, s_next, rew_current, EoE = tup
                    vs_a += tp_next * (rew_current + gamma * vs_old[s_next])
                vs_new += policy[s][a] * vs_a

            value_function[s] = vs_new
            delta = max(delta, abs(value_function[s] - vs_old[s]))

        if delta < tol:
            break
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(0, nS):
        old_action = new_policy[s]
        q_sa = [0, 0, 0, 0]

        for a in range(0,nA):
            vs_a = 0
            for tup in P[s][a]:
                tp_next, s_next, rew_current, EoE = tup
                vs_a += tp_next * (rew_current + gamma * value_from_policy[s_next])
            q_sa[a] = vs_a

        argmax_a = q_sa.index(max(q_sa))
        new_policy[s] = np.zeros([1, nA])   # reset policy
        new_policy[s][argmax_a] = 1.0       # update argmax action to 1
    ############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    ############################
    while True:
        policy_stable = True
        value_from_policy = policy_evaluation(P, nS, nA, new_policy, gamma=0.9, tol=1e-8)
        old_policy = new_policy

        new_policy = policy_improvement(P, nS, nA, value_from_policy, gamma=0.9)

        if (old_policy == new_policy).all():
            break

    V = value_from_policy
    return new_policy, V

    ############################

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """

    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #

    ### Finding optimal value without policy
    while True:
        delta = 0
        vs_old = V_new.copy()

        for s in range(0, nS):
            vs_new = 0
            for a in range(0,nA):
                vs_a = 0
                for tup in P[s][a]:
                    tp_next, s_next, rew_current, EoE = tup
                    vs_a += tp_next * (rew_current + gamma * vs_old[s_next])
                vs_new = max(vs_new, vs_a)

            V_new[s] = vs_new
            delta = max(delta, abs(V_new[s] - vs_old[s]))

        if delta < tol:
            break

    ### Extracting optimal policy from the above optimal value function
    policy_new = np.zeros([nS, nA])
    for s in range(0, nS):
        q_sa = [0, 0, 0, 0]

        for a in range(0,nA):
            vs_a = 0
            for tup in P[s][a]:
                tp_next, s_next, rew_current, EoE = tup
                vs_a += tp_next * (rew_current + gamma * V_new[s_next])
            q_sa[a] = vs_a

        argmax_a = q_sa.index(max(q_sa))
        policy_new[s][argmax_a] = 1.0       # update argmax action to 1

    ############################
    return policy_new, V_new


def render_single(env, policy, render=False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [nS, nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game.
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    -----
    Transition can be done using the function env.step(a) below with FIVE output parameters:
    ob, r, done, info, prob = env.step(a)
    """
    total_rewards = 0
    for _ in range(n_episodes):
        obs = env.reset()  # initialize the episode
        done = False

        while not done:
            if render:
                env.render()  # render the game
                time.sleep(0.01)
            ############################
            # YOUR IMPLEMENTATION HERE #
            action = np.argmax(policy[obs])

            obs, reward, terminated, truncated, info = env.step(action)
            total_rewards += reward
            if terminated or truncated:
                done = True

    return total_rewards
