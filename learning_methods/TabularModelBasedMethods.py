import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float) # create array size of n_states of type numpy float

    iterations = 0 # initialisaton of iterations

    while iterations < max_iterations: # loop criteria
        delta = 0
        for sc in range(env.n_states): # for all the current states in n_states
            v = value[sc] # note the value indexed by the current state
            # sum of probability * (reward+discount_factor*value) for potential values of next state
            value[sc] = sum([env.p(sn,sc,policy[sc]) * (env.r(sn,sc,policy[sc]) + gamma * value[sn]) for sn in range(env.n_states)])
            delta = max(delta, abs(v - value[sc])) #make delta equal to greater value between current delta or magnitude of change in value
        if delta < theta: # if delta smaller than pre-defined threshold break while loop
            break
        iterations = iterations + 1 # track iterations
    return value


def policy_improvement(env, policy, value, gamma):
    if policy is None: #policy initialisation
        policy = np.zeros(env.n_states, dtype=int) #array size of n_states
    else:
        policy = np.array(policy, dtype=int) #sarray size of input policy

    p = env.p # call p function from environment
    r = env.r # call r function from environment

    stable = True # assuming no initial change in policy
    for s in range(env.n_states): # loop through all current states
        b = policy[s] #current policy

        actions = env.a(s) #get all actions
        _actions = []

        for a in range(len(actions)):
            _actions.append(a) # store index of actions

        # take sum of p(r*gamma*value) for all possible next states
        # take the argument of the max value for all possible actions
        # take value of _actions based on index from previous step
        policy[s] = _actions[int(np.argmax([sum([p(sn, s, a) * (r(sn, s, a) + gamma * value[sn]) for sn in range(env.n_states)]) for a in _actions]))]
        if b != policy[s]: # if policy has changed
            stable = False # it is not stable

    return policy, stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    iterations = 0

    stable = False
    while not stable: # while policy continues to change
        values = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy, stable = policy_improvement(env, policy, values, gamma)
        iterations += 1

    print("POLICY_ITERATION iterations: ", iterations)
    return policy, values


def value_iteration(env, gamma, theta, max_iterations, value=None):
    # initialise valye
    if value is None:
        value = np.zeros(env.n_states) # array of zeros size of n_states
    else:
        value = np.array(value, dtype = np.float) # array size of value, type numpy float

    p = env.p #get probability function from environment
    r = env.r #get reward function from environment

    iterations = 0 # initialise iterations

    while True and iterations < max_iterations:  # loop criteria
        delta = 0 # intialise delta
        for s in range(env.n_states): # for all preset states
            actions = env.a(s) #get actions
            _actions = []

            for a in range(len(actions)):
                _actions.append(a) # store index of actions

            v = value[s] # take value of current state
            # for all possible next states calculate sum of p*(r_gamma*value)
            # for all possible actions, select the max and store as value for current state
            value[s] = max([sum([p(sn, s, a) * (r(sn, s, a) + gamma * value[sn])for sn in range(env.n_states)]) for a in _actions])
            delta = max(delta, abs(v - value[s])) # take greater value between current delta and magnitude of change in value
        if delta < theta: # criteria to break loop
            break
        iterations = iterations + 1 # increase iterations

    policy = np.zeros((env.n_states), dtype=int) # array of zeros size of n_states
    for s in range(env.n_states): #  for all current state in n_states
        actions = env.a(s)
        _actions = []

        for a in range(len(actions)):
            _actions.append(a) # store index of actions

        policy[s] = _actions[int(np.argmax([sum([p(sn, s, a) * (r(sn, s, a) + gamma * value[sn]) for sn in range(env.n_states)]) for a in _actions]))]

    print("VALUE_ITERATION iterations: ", iterations)
    return policy, value
