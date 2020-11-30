import numpy as np


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)

# Uses Sutton and Barto's Implementation - Ch 8.4 Pg 212 Edition 1
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    raise NotImplementedError()
    # random_state = np.random.RandomState(seed)
    #
    # eta = np.linspace(eta, 0, max_episodes)
    # epsilon = np.linspace(epsilon, 0, max_episodes)
    #
    # theta = np.zeros(env.n_features)
    #
    # for i in range(max_episodes):
    #     features = env.reset()
    #     # q = features.dot(theta)
    #
    #     # TODO:
    #
    # return theta


# Apply Algorithm 1 from the Assignment Document:
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    # Eta and Epsilon is decayed every episode:
    # Eta: learning rate -- Gamma: Discount rate  -- Epsilon: Greedy factor
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()  # set the initial state by random, states are our features here (a, (s,a))
        q = features.dot(theta)  # for each action construct Q using LF Approx. (Line 4->6) - Vectorized for loop

        done = False  # initially false, check in step of the loop
        delta = 0
        while not done:  # Starts line 7

            # Do epsilon greedy approach: (Lines 8-12)
            if np.random.uniform(0, 1) < (1-epsilon[i]):
                a = np.random.choice(np.flatnonzero(q == q.max()))  # does random tie breaking as well
            else:
                a = np.random.choice(env.n_actions)

            # Do step operation to use action to get state and reward: (Lines 13, 14)
            next_features, reward, done = env.step(a)

            # Calculate temporal difference: (Line 15):
            delta = reward - q[a]

            # Calculate new Q values for the next state: (Lines 16-18) - vectorized for loop
            q = next_features.dot(theta)  # TODO: check here as well use of features matrix

            # Lines 19-21:
            delta += gamma * max(q)
            theta += eta[i] * delta * features[a]
            features = next_features  # not being used; how to incorporate state to wrapper
        # Line 22 - While loop ends
    # Line 23: For loop ends
    return theta
