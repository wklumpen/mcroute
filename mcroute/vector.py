import numpy as np


def uniform(state_space):
    p = 1.0/state_space.size
    r = [p for i in range(state_space.size)]
    r[-1] = 1.0 - sum(r[:-1])
    return np.array(r)


def unit(state_space, state_name):
    r = [0.0 for i in range(state_space.size)]
    r[state_space.index_at(state_name)] = 1.0
    return np.array(r)