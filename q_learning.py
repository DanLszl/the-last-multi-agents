import random
from typing import Dict, Callable

from grid import Grid
from grid import grid_def
from grid import Action

from policy import Policy, EpsilonGreedy

from plotting import plot_grid


# %load_ext autoreload
# %autoreload 2


def Q_learning_choose_next_q_value(q_values: Dict[Action, float], policy: Policy):
    return max(q_values.values())


def SARSA_choose_next_q_value(q_values: Dict[Action, float], policy: Policy):
    action = policy.select_action(q_values)
    return q_values[action]


def TD(gamma: float, alfa: float, grid: Grid, policy: Policy, num_of_episodes: int, choose_next_q_value: Callable[[Dict[Action, float], Policy], float]):
    for e in range(num_of_episodes):
        current_position = grid.get_initial_random_position()
        episode_reward = 0
        while not current_position.is_end_state:
            action = policy.select_action(current_position.q_values)
            q_value = current_position.q_values[action]
            next_position = current_position.take_action(action)
            reward = next_position.reward
            next_q_values = next_position.q_values
            chosen_next_q_value = choose_next_q_value(next_q_values, policy)
            new_q_value = q_value + alfa * (reward + gamma * chosen_next_q_value - q_value)
            current_position.update_q_value(action, new_q_value)

            current_position = next_position
            episode_reward += reward

        # print(episode_reward)


def SARSA(gamma: float, alfa: float, grid: Grid, policy: Policy, num_of_episodes: int):
    TD(gamma, alfa, grid, policy, num_of_episodes, SARSA_choose_next_q_value)


def Q_learning(gamma: float, alfa: float, grid: Grid, policy: Policy, num_of_episodes: int):
    TD(gamma, alfa, grid, policy, num_of_episodes, Q_learning_choose_next_q_value)


def main():
    gamma = 0.9
    epsilon = 0.8
    alfa = 0.1

    policy = EpsilonGreedy(epsilon)

    grid = Grid(grid_def)

    # Q_learning(gamma, alfa, grid, policy, num_of_episodes=100000)
    SARSA(gamma, alfa, grid, policy, num_of_episodes=100000)

    plot_grid(grid)


if __name__ == '__main__':
    main()
