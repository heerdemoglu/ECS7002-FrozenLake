import numpy as np


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states

        # Features are the grid of actions and states. (Dimensionality; Every combination of (s,a))
        # self.n_actions * self.n_states: represents dimensionality all movements combinations
        # this will be used with one-hot encoding to uniquely identify every state-action with a unique sparse
        # identifier.
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))  # For this question every action has separate features
        # ravel_multi_index gives how deep an indexed value is:
        # Here, given (s,a) locate the parameter, return its index - like flattening and returning the result
        # (s,a) decide on the linearized depth; (self.n_states, self_n_actions)
        # is the size of the array that we look from.
        for a in range(self.n_actions):
            # i := index of feature a given state s. Make a matrix of  (self.n_states, self.n_actions):
            # which flattened index would (s, a) would correspond to.
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0  # set the ith value of this action to 1, leave the rest to zero.
            # What it does -- Given a state s: initialize feature vector for all actions.
            # Finds the flattened index of (s,a) if (s,a) was kept in tabular form.
            # One-hot encodes taking of that action (only a single action) to this index.
            # does this to all possible actions of a given state.
            # state-encoding applied on the fly.
        return features  # Dimension 4x (n_states*n_actions)

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        # Reverse the encoding operation to get the policy and the value.
        for s in range(self.n_states):
            features = self.encode_state(s)  # Fetching all encoded features of a given state-action
            q = features.dot(theta)  # Linear combination of states with state parameter (weights-theta)
            # Returns q (vector of 4, for the given state)
            # Multiplication with theta gives linear combination of state s' with the learned weights from all
            # past actions taken.
            #
            # As no tables are present; "q dot theta" actually filters the theta value encoded
            # for that particular state-actions, due to one hot encoding; all unrelated theta values will be
            # suppressed by feature values.

            # Policy via argmax returns index for that particular state - take best action from q
            policy[s] = np.argmax(q)
            # Value by max, returns value of the best q.
            value[s] = np.max(q)
            # Best policy and value is picked for each state.

        return policy, value

    def reset(self):
        # Reset the state and encode for linear wrapping
        return self.encode_state(self.env.reset())

    def step(self, action):
        # take action in the environment
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        # render called from super.
        self.env.render(policy, value)


# Uses Sutton and Barto's Implementation - Ch 8.4 Pg 212 Edition 1
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()  # set the initial state by random, states are our features here (a, (s,a))
        q = features.dot(theta)  # for each action construct Q using LF Approx. (Line 4->6) - Vectorized for loop

        # Choose A from S using policy from Q:
        if np.random.uniform(0, 1) < (1 - epsilon[i]):
            aa = np.random.choice(np.flatnonzero(q == q.max()))  # does random tie breaking as well
        else:
            aa = np.random.choice(env.n_actions)

        done = False  # initially false, check in step of the loop
        while not done:  # continue until done

            # Act on the action observe reward and next LFA state:
            next_features, reward, done = env.step(aa)

            # Calculate temporal difference:
            delta = reward - q[aa]  # R - Q(s,a)

            # Create new Q from the update:
            q = next_features.dot(theta)

            # Choose A' from S' using policy from Q:
            if np.random.uniform(0, 1) < (1 - epsilon[i]):
                a = np.random.choice(np.flatnonzero(q == q.max()))  # does random tie breaking as well
            else:
                a = np.random.choice(env.n_actions)

            # doing the updates
            delta += gamma * q[a]
            theta += eta[i] * delta * features[aa]

            # State update:
            features = next_features
            aa = a
        # While loop ends
    # For loop ends
    return theta


# Apply Algorithm 1 from the Assignment Document:
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    np.random.RandomState(seed)

    # Eta and Epsilon is decayed every episode:
    # Eta: learning rate -- Gamma: Discount rate  -- Epsilon: Greedy factor
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()  # set the initial state by random, states are our features here (a, (s,a))
        q = features.dot(theta)  # for each action construct Q using LF Approx. (Line 4->6) - Vectorized for loop

        done = False  # initially false, check in step of the loop
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
            q = next_features.dot(theta)

            # Lines 19-21:
            delta += gamma * max(q)
            theta += eta[i] * delta * features[a]

            # State update:
            features = next_features
        # Line 22 - While loop ends
    # Line 23: For loop ends
    return theta
