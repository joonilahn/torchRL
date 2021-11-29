import random

import numpy as np
from gym.spaces import Space

random.seed(99)


class DiscreteList(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    """

    def __init__(self, n, num_out):
        assert n > 0 and num_out > 0
        self.n = n
        self.num_out = num_out
        self.actions = tuple(range(self.n))
        super(DiscreteList, self).__init__((), np.int64)

    def sample(self):
        return sorted(random.sample(self.actions, self.num_out))

    def contains(self, x):
        for a in x:
            if a < 0 or a >= self.n:
                return False
        return True

    def __repr__(self):
        return f"DiscreteList, {self.n}, {self.num_out}"

    def __eq__(self, other):
        return (
                isinstance(other, DiscreteList)
                and self.n == other.n
                and self.num_out == other.num_out
        )
