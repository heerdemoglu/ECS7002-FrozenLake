import contextlib
from Environment import Environment
import numpy as np


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        # TODO:

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # TODO:
        raise NotImplementedError()

    def r(self, next_state, state, action):
        # TODO:
        raise NotImplementedError()

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

    def play(env):
        actions = ['w', 'a', 's', 'd']

        state = env.reset()
        env.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid action')

            state, r, done = env.step(actions.index(c))

            env.render()
            print('Reward: {0}.'.format(r))

        ################ Model-based algorithms ################

    def policy_evaluation(env, policy, gamma, theta, max_iterations):
        value = np.zeros(env.n_states, dtype=np.float)

        # TODO:

        return value

    def policy_improvement(env, value, gamma):
        policy = np.zeros(env.n_states, dtype=int)

        # TODO:

        return policy

    def value_iteration(env, gamma, theta, max_iterations, value=None):
        if value is None:
            value = np.zeros(env.n_states)
        else:
            value = np.array(value, dtype=np.float)

        for _ in range(max_iterations):
            delta = 0.

            for s in range(env.n_states):
                v = value[s]
                value[s] = max([sum(
                    [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in
                     range(env.n_states)])
                    for a in range(env.n_actions)])

                delta = max(delta, np.abs(v - value[s]))

            if delta < theta:
                break

        policy = np.zeros(env.n_states, dtype=int)
        for s in range(env.n_states):
            policy[s] = np.argmax([sum(
                [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)])
                for
                a in range(env.n_actions)])

        return policy, value

    def policy_evaluation(env, policy, gamma, theta, max_iterations):
        value = np.zeros(env.n_states, dtype=np.float)

        for _ in range(max_iterations):
            delta = 0
            for s in range(env.n_states):
                v = value[s]
                value[s] = sum(
                    [env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * value[next_s]) for next_s in
                     range(env.n_states)])

                delta = max(delta, abs(v - value[s]))

            if delta < theta:
                break

        return value

    ################ Tabular model-free algorithms ################

    def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
        random_state = np.random.RandomState(seed)

        eta = np.linspace(eta, 0, max_episodes)
        epsilon = np.linspace(epsilon, 0, max_episodes)

        q = np.zeros((env.n_states, env.n_actions))

        for i in range(max_episodes):
            s = env.reset()
            # TODO:

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
            # TODO:

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value
