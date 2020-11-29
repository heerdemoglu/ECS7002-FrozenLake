import numpy as np


# Additional e-greedy implementation: ==============================
def e_greedy(q, actions, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)
    else:
        return actions[q.argmax(axis=1)]
# ==================================================================


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()

        # SARSA Code: =================================================================
        a = e_greedy(q[s], env.n_actions(s), epsilon)
        while not env.n_states[-1]:
            r = env.r()
            next_s = env.n_states[env.n_states.index(s) + 1]
            next_a = e_greedy(q[next_s], env.n_actions(next_s), epsilon)
            q[s, a] = q[s, a] + eta * (r + (gamma * q[next_s, next_a]) - q[s, a])
            s = next_s
            a = next_a
        # =============================================================================

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()

        # Q learning Code: =================================================================
        while not env.n_states[-1]:
            a = e_greedy(q[s], env.n_actions(s), epsilon)
            r = env.r()
            next_s = env.n_states[env.n_states.index(s) + 1]
            q[s, a] = q[s, a] + eta * (r + (gamma * max(q[next_s])) - q[s, a])
            s = next_s
        # ==================================================================================

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
