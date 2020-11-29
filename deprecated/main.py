from environment import FrozenLake as fl
from learning_methods import TabularModelBasedMethods as tbm


def main():

    seed = 0

    # Small Lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = fl.FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    # env.play()
    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    print('')
    print('## Policy Iteration')
    policy = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    policy, value = tbm.policy_iteration(env, gamma, theta, max_iterations, policy)
    env.render(policy, value)

    print('## Value Iteration')
    policy, value = tbm.value_iteration(env, gamma, theta, max_iterations, value=None)
    env.render(policy, value)


main()
