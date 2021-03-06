from itertools import product

import numpy as np
import contextlib
import Auxilary_Functions as af

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed = None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p = p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed = None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps
        self.random_state = np.random.RandomState(seed)

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p = self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid Action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy = None, value = None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed = None):

        self.random_state = np.random.RandomState(seed)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        self.n_states = n_states
        n_actions = 4
        self.n_actions = n_actions

        pi = np.zeros(n_states, dtype = float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.pi = pi

        self.absorbing_state = n_states - 1
        self.max_steps = max_steps
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Indices to states (coordinates), states (coordinates) to indices
        self.gmap = af.derive_gmap(self.lake) # convert the self.lake into a 'reward' map, 1 at the goal, 0 everywhere else
        self.itos = list(product(range(self.gmap.shape[0]), range(self.gmap.shape[1])))
        self.itos.append(tuple((-1,-1))) # create an absorb state
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}

        # Precomputed transition probabilities
        self._p = np.zeros((self.n_states, self.n_states, self.n_actions))
        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):

                if state_index == self.absorbing_state:
                    next_state == (-1,-1)
                    next_state_index = self.stoi.get(next_state, state_index)
                    self._p[next_state_index, state_index, action_index] = 1.0 # stay in absorb state if already in absorb state
                elif state_index != self.absorbing_state and (self.lake_flat[state_index] == '#' or self.lake_flat[state_index] == '$'):
                    next_state = (-1, -1) #stay in position
                    next_state_index = self.stoi.get(next_state, state_index) # to absorb state if at hole or reward
                    self._p[next_state_index, state_index, action_index] = 1.0
                else: # if not at hole, reward or absorb state
                    next_state = (state[0] + action[0], state[1] + action[1]) # update position
                    next_state_s0 = (state[0] + self.actions[0][0], state[1] + self.actions[0][1]) # consider slip direction 1
                    next_state_s1 = (state[0] + self.actions[1][0], state[1] + self.actions[1][1]) # consider slip direction 2
                    next_state_s2 = (state[0] + self.actions[2][0], state[1] + self.actions[2][1]) # consider slip direction 3
                    next_state_s3 = (state[0] + self.actions[3][0], state[1] + self.actions[3][1]) # consider slip direction 4

                    # If next_state is not valid, default to current state index
                    next_state_index = self.stoi.get(next_state, state_index)
                    next_state_index_s0 = self.stoi.get(next_state_s0, state_index) # next state based on slip direction 1
                    next_state_index_s1 = self.stoi.get(next_state_s1, state_index) # next state based on slip direction 2
                    next_state_index_s2 = self.stoi.get(next_state_s2, state_index) # next state based on slip direction 3
                    next_state_index_s3 = self.stoi.get(next_state_s3, state_index) # next state based on slip direction 4

                    # define a probability
                    self._p[next_state_index_s0, state_index, action_index] += self.slip/4 # probabilty to slip in any of 4 directions
                    self._p[next_state_index_s1, state_index, action_index] += self.slip/4 # probabilty to slip in any of 4 directions
                    self._p[next_state_index_s2, state_index, action_index] += self.slip/4 # probabilty to slip in any of 4 directions
                    self._p[next_state_index_s3, state_index, action_index] += self.slip/4 # probabilty to slip in any of 4 directions
                    self._p[next_state_index, state_index, action_index] += 1.0-self.slip # probabilty to go in intended direction + probability of ending up there due to slip

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    def a(self, state):
        return self.actions

    def p(self, next_state, state, action):
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        if state != self.absorbing_state:
            return self.gmap[self.itos[state]]
        else:
            return 0

    def is_final(self, state):
        if state == self.absorbing_state:
            return True
        else:
            return False

    def render(self, policy = None, value = None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if  self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            actions = ['^', '_', '<', '>']

            print('Lake: ')
            print(self.lake)

            print('Policy: ')
            policy = np.array([actions[a] for a in policy [:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision = 3, suppress = True):
                print(value[:-1].reshape(self.lake.shape))

    def play(env):
        actions = ['w','a','s','d']

        state = env.reset()
        env.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid Action')

            state, r, done = env.step(actions.index(c))

            env.render()
            print('Reward: {0}.'.format(r))
