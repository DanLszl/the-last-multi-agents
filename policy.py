import random
from typing import Dict

from grid import Action


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
