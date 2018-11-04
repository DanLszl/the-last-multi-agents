'''
Usage:
  q_learning.py [--Q-learning|--SARSA] [--episodes <value>]
    [--task <gridworld|cliffwalking>] [--epsilon <value>]
'''

from typing import Dict, Callable

from grid import Grid
from grid import grid_def, cliffwalking_def
from grid import Action

from policy import Policy, EpsilonGreedy, EpsilonGreedyGLIE

from plotting import plot


def Q_learning_choose_next_q_value(q_values: Dict[Action, float], policy: Policy):
    return max(q_values.values())


def SARSA_choose_next_q_value(q_values: Dict[Action, float], policy: Policy):
    action = policy.select_action(q_values, new_episode=False)
    return q_values[action]


def TD(gamma: float, alfa: float, grid: Grid, policy: Policy, num_of_episodes: int, choose_next_q_value: Callable[[Dict[Action, float], Policy], float], **kwargs):
    for e in range(num_of_episodes):
        # if the grid_def doesn't contain a starting pos,
        # current_position is going to be random
        current_position = grid.get_starting_position()
        episode_reward = 0
        new_episode_flag = True
        while not current_position.is_end_state:
            action = policy.select_action(current_position.q_values, new_episode_flag)
            q_value = current_position.q_values[action]
            next_position = current_position.take_action(action)
            reward = next_position.reward
            next_q_values = next_position.q_values
            chosen_next_q_value = choose_next_q_value(next_q_values, policy)
            new_q_value = q_value + alfa * (reward + gamma * chosen_next_q_value - q_value)
            current_position.update_q_value(action, new_q_value)

            current_position = next_position
            episode_reward += reward
            new_episode_flag = False
        # print(episode_reward)


def SARSA(gamma: float, alfa: float, grid: Grid, policy: Policy, num_of_episodes: int, **kwargs):
    TD(gamma, alfa, grid, policy, num_of_episodes, SARSA_choose_next_q_value, **kwargs)


def Q_learning(gamma: float, alfa: float, grid: Grid, policy: Policy, num_of_episodes: int, **kwargs):
    TD(gamma, alfa, grid, policy, num_of_episodes, Q_learning_choose_next_q_value, **kwargs)


def parse_parameters():
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['Q-learning', 'SARSA', 'episodes=', 'task=', 'epsilon='])
    opts = dict(opts)

    try:
        episodes = int(opts['--episodes'])
    except KeyError:
        episodes = 10000

    try:
        task = opts['--task']
    except KeyError:
        task = 'gridworld'

    try:
        epsilon = float(opts['--epsilon'])
    except KeyError:
        epsilon = 0.1

    if '--Q-learning' in opts:
        algorithm_name = 'Q-learning'
    elif '--SARSA' in opts:
        algorithm_name = 'SARSA'
    else:
        algorithm_name = 'Q-learning'

    print('Using the following parameters:')
    print('\talgorithm: ', algorithm_name)
    print('\tepisodes: ', episodes)
    print('\ttask: ', task)
    print('\tepsilon: ', epsilon)
    print('\t')

    return algorithm_name, episodes, task, epsilon


def main():
    algorithm_name, episodes, task, epsilon = parse_parameters()

    gamma = 1

    if task == 'gridworld':
        grid = Grid(grid_def)
        plot_orientation='horizontal'
    elif task == 'cliffwalking':
        grid = Grid(cliffwalking_def)
        plot_orientation='vertical'
    else:
        print('The task ', task, ' is not implemented')
        exit(1)

    if algorithm_name == 'Q-learning':
        algorithm = Q_learning
        # Q learning is off policy, and the env is fully deterministic
        # so alfa can be 1
        learning_rate = 1
        policy = EpsilonGreedy(epsilon)
    else:
        algorithm = SARSA
        learning_rate = 0.1
        policy = EpsilonGreedy(epsilon)
        #policy = EpsilonGreedyGLIE()

    algorithm(gamma, learning_rate, grid, policy, episodes)

    q_filename = task + '_' + algorithm_name
    p_filename = q_filename + '_policy'

    plot(grid, 'plots/' + q_filename + '.png', show_plot=False, plot_orientation=plot_orientation)


if __name__ == '__main__':
    print(__doc__)
    main()
