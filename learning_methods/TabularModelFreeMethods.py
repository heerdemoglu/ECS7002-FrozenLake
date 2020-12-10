import numpy as np


def e_greedy(q, actions, epsilon):
    if np.random.uniform(0, 1) < (1 - epsilon):
        return np.random.choice(np.flatnonzero(q == q.max()))
    else:
        return np.random.choice(actions)


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()     # Reset the states at the start of the episode
        a = e_greedy(q[s], env.n_actions, epsilon[i])   # selecting the action a according to e-greedy policy given state s
        terminal = False

        while not terminal:
            next_s, r, terminal = env.step(a)  # Storing the current state, the current reward and if this is the terminal state or not
            next_a = e_greedy(q[s], env.n_actions, epsilon[i]) # selecting the action next_a according to e-greedy policy given state next_s
            q[s, a] = q[s, a] + eta[i] * (r + (gamma * q[next_s, next_a]) - q[s, a])   # Storing the new values in the q
            s = next_s  # storing the next state as the current state for the next episode
            a = next_a  # storing the next action as the current action for the next episode

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()     # Reset the states at the start of the episode
        terminal = False

        while not terminal:
            a = e_greedy(q[s], env.n_actions, epsilon[i])   # selecting the action a according to e-greedy policy given state s
            next_s, r, terminal = env.step(a)   # Storing the current state, the current reward and if this is the terminal state or not
            q[s, a] = q[s, a] + eta[i] * (r + (gamma * max(q[next_s])) - q[s, a])   # Storing the new values in the q
            s = next_s  # storing the next state as the current state for the next episode

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value