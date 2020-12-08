import Listing_1 as l1
import Listing_2 as l2

def main():

    seed = 0

    #Small Lake

    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
    """
    lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$'],]
    """

    env = l1.FrozenLake(lake, slip = 0.1, max_steps=16, seed = seed)
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    print('')
    print('## Policy Iteration')
    # initial arbitrary policy for small lake
    policy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # initial arbitrary policy for large lake
    #policy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    policy, value = l2.policy_iteration(env, gamma, theta, max_iterations, policy)
    env.render(policy, value)

    print('')
    print('## Value Iteration')
    policy, value = l2.value_iteration(env, gamma, theta, max_iterations, value=None)
    env.render(policy, value)

main()
