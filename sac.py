from collections import deque

import numpy as np


class Actor:
    pass


class Critic:
    pass


class SAC:

    def __init__(self, buffer_max_size=1000):
        self.buffer = deque(maxlen=buffer_max_size)

    def action(self, state):
        return np.array([0])

    def train_step(self):
        pass
