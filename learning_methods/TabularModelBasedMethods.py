import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    iterations = 0

    while iterations < max_iterations:
        delta = 0
        for sc in range(env.n_states):
            v = value[sc]
            value[sc] = sum([env.p(sn, sc, policy[sc]) * (env.r(sn, sc, policy[sc]) + gamma * value[sn]) for
                             sn in range(env.n_states)])
            delta = max(delta, abs(v - value[sc]))
        if delta < theta:
            break
        iterations = iterations + 1
    return value


def policy_improvement(env, policy, value, gamma):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    p = env.p
    r = env.r

    stable = True
    for s in range(env.n_states):
        b = policy[s]

        actions = env.a(s)
        _actions = []

        for a in range(len(actions)):
            _actions.append(a)

        policy[s] = _actions[int(np.argmax([sum([p(sn, s, a) * (r(sn, s, a) + gamma * value[sn]) for
                                                 sn in range(env.n_states)]) for a in _actions]))]
        if b != policy[s]:
            # print(b, policy[s], s)
            stable = False

    return policy, stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    stable = False
    while not stable:
        values = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy, stable = policy_improvement(env, policy, values, gamma)
        print(stable)

    return policy, values


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    p = env.p
    r = env.r
    iterations = 0
    while True and iterations < max_iterations:
        delta = 0
        for s in range(env.n_states):
            actions = env.a(s)
            _actions = []

            for a in range(len(actions)):
                _actions.append(a)

            v = value[s]
            value[s] = max([sum([p(sn, s, a) * (r(sn, s, a) + gamma * value[sn])
                                 for sn in range(env.n_states)]) for a in _actions])
            delta = max(delta, abs(v - value[s]))
        if delta < theta:
            break
        iterations = iterations + 1

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        actions = env.a(s)
        _actions = []

        for a in range(len(actions)):
            _actions.append(a)
        policy[s] = _actions[int(np.argmax([sum([p(sn, s, a) * (r(sn, s, a) + gamma * value[sn])
                                                 for sn in range(env.n_states)]) for a in _actions]))]
    return policy, value
