import random
from typing import Dict

from grid import Action


class Policy:
    def select_action(self, q_values: Dict[Action, float], new_episode: bool) -> Action:
        raise NotImplementedError()


class EpsilonGreedy(Policy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, q_values: Dict[Action, float], new_episode: bool) -> Action:
        if random.random() < self.epsilon:
            # select random action
            return random.choice(list(q_values.keys()))
        else:
            # select greedy action
            return max(q_values, key=q_values.get)


class EpsilonGreedyGLIE(EpsilonGreedy):
    def __init__(self):
        super().__init__(1.0)
        self.episode = 0

    def update_epsilon(self):
        self.episode += 1
        self.epsilon = 1.0 / self.episode

    def select_action(self, q_values: Dict[Action, float], new_episode: bool) -> Action:
        if new_episode:
            self.update_epsilon()
        return super().select_action(q_values, new_episode)

from policy import EpsilonGreedyGLIE
