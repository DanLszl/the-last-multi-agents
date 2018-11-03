from grid import Grid
from grid import grid_def

from grid import Action

#from random import random, randint, choice
import random

from typing import Dict
import operator
from grid import Action


import matplotlib.pyplot as plt
import numpy as np


# %load_ext autoreload
# %autoreload 2


class Policy:
    def select_action(self, q_values: Dict[Action, float]) -> Action:
        raise NotImplementedError()


class EpsilonGreedy(Policy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, q_values: Dict[Action, float]) -> Action:
        if random.random() < self.epsilon:
            # select random action
            return random.choice(list(Action))
        else:
            # select greedy action
            return max(q_values, key=q_values.get)


def Q_learning(gamma: float, alfa: float, grid: Grid, policy: Policy, num_of_episodes: int):
    for e in range(num_of_episodes):
        current_position = grid.get_initial_random_position()
        episode_reward = 0
        while not current_position.is_end_state:
            if not current_position.can_step_on_it:
                print("How am I here?")
            action = policy.select_action(current_position.q_values)
            q_value = current_position.q_values[action]
            next_position = current_position.take_action(action)
            reward = next_position.reward
            next_q_values = next_position.q_values
            best_next_q_value = max(next_q_values.values())
            new_q_value = q_value + alfa * (reward + gamma * best_next_q_value - q_value)
            current_position.update_q_value(action, new_q_value)

            current_position = next_position
            episode_reward += reward

        # print(episode_reward)


def plot(grid):
    left = []
    right = []
    top = []
    bottom = []
    for row in grid.grid[1:-1]:
        left_row = []
        right_row = []
        top_row = []
        bottom_row = []
        for cell in row[1:-1]:
            top_row.append(cell.q_values[Action.N])
            right_row.append(cell.q_values[Action.E])
            bottom_row.append(cell.q_values[Action.S])
            left_row.append(cell.q_values[Action.W])
        left.append(left_row)
        right.append(right_row)
        top.append(top_row)
        bottom.append(bottom_row)

    left = np.array(left)
    right = np.array(right)
    top = np.array(top)
    bottom = np.array(bottom)

    # Plotting Source: https://stackoverflow.com/questions/44666679/something-like-plt-matshow-but-with-triangles

    def quatromatrix(left, bottom, right, top, ax=None, triplotkw={},tripcolorkw={}):
        if not ax: ax=plt.gca()
        n = left.shape[0]; m=left.shape[1]

        a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
        tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

        A = np.zeros((n*m*5,2))
        Tr = np.zeros((n*m*4,3))

        for i in range(n):
            for j in range(m):
                k = i*m+j
                A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
                Tr[k*4:(k+1)*4,:] = tr + k*5

        C = np.c_[ left.flatten(), bottom.flatten(),
                  right.flatten(), top.flatten()   ].flatten()

        triplot = ax.triplot(A[:,0], A[:,1], Tr, **triplotkw)
        tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, **tripcolorkw)
        return tripcolor

    fig, ax=plt.subplots()

    quatromatrix(left, bottom, right, top, ax=ax,
                 triplotkw={"color":"k", "lw":1},
                 tripcolorkw={"cmap": "gray"})

    ax.margins(0)
    ax.set_aspect("equal")
    plt.show()


def main():
    gamma = 0.9
    epsilon = 0.8
    alfa = 0.1

    policy = EpsilonGreedy(epsilon)

    grid = Grid(grid_def)

    Q_learning(gamma, alfa, grid, policy, num_of_episodes=100000)

    plot(grid)

if __name__ == '__main__':
    main()
