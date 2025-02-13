import math
from collections import defaultdict


class Stat:

    def __init__(self):
        self.min = math.inf
        self.max = -math.inf
        self.m2 = 0
        self.mean = 0
        self.n = 0

    def update(self, value):
        self.n += 1
        self.min = min(value, self.min)
        self.max = max(value, self.max)
        # Welford's update for variance
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def var(self):
        return self.m2 / self.n

    def std(self):
        return np.sqrt(self.var()) if self.n > 1 else 0.0


class Stats:

    def __init__(self):
        self.stats = defaultdict(Stat)

    def update(self, name, value):
        self.stats[name].update(value)

    def get(self, reset=False):
        result = {}
        for name, stat in self.stats.items():
            result[f'{name}_min'] = stat.min
            result[f'{name}_max'] = stat.max
            result[f'{name}_mean'] = stat.mean
            result[f'{name}_std'] = stat.std()
        if reset:
            self.reset()
        return result

    def reset(self):
        self.stats = defaultdict(Stat)
