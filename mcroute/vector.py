"""
Vector construction and manipulation functions for MCRoute networks.

This module contains methods that enable the constructing of vectors needed as
initial conditions for traversal and probability evolution through the network.
Typically, these methods require a defined :class:`mcroute.StateSpace` object
which is used to provide context for building the distribution.
For example:

.. code-block:: Python

    from mcroute import StateSpace
    import mcroute.vector as vector

    space = StateSpace.from_range(-2, 3)
    vector.unit(space, '0')

will produce a unit vetor with zero probability in all rows except for the named
state '0', which will have probability 1.0. All vector creation
methods in this module will return :class:`numpy.Array` objects.
"""
import numpy as np


def uniform(state_space):
    """Create a uniformly distributed probability vector under a state space.

    :param state_space: The state space under which to construct the distribution
    :type state_space: :class:`mcroute.StateSpace`
    :return: A vector of probabilities
    :rtype: :class:`numpy.array`
    """    
    p = 1.0/state_space.size
    r = [p for i in range(state_space.size)]
    r[-1] = 1.0 - sum(r[:-1])
    return np.array(r)


def unit(state_space, state_name):   
    """Create a unit probability vector at a specified state name.

    This method creates a vector with all zeroes except for at the specified
    state name, which will have probability 1.0.

    :param state_space: The state space under which to construct the vector
    :type state_space: :class:`mcroute.StateSpace`
    :param state_name: The name of the state with unit probability
    :type state_name: str
    :return: A vector of probabilities
    :rtype: :class:`numpy.array`
    """    
    r = [0.0 for i in range(state_space.size)]
    r[state_space.index_at(state_name)] = 1.0
    return np.array(r)