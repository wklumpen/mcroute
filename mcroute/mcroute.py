import csv

import networkx as nx
import numpy as np

class SizeError(Exception):
    pass

class DataError(Exception):
    pass

class MatrixIntegrityError(Exception):
    pass


class Network(nx.DiGraph):
    def __init__(self, state_space):
        super().__init__()

        if not isinstance(state_space, StateSpace):
            raise TypeError("State space must be of type StateSpace")
        # Check for positive states
        self._state_space = state_space

    def add_node(self, name, matrix):
        super().add_node(name, matrix=_check_matrix(matrix))
    
    def add_edge(self, u, v, matrix):
        super().add_edge(u, v, matrix=_check_matrix(matrix))
    
    def traverse(self, nodes, starting_vector):
        """
        Traverse a path along a prescribed set of nodes

        The function evolves the `starting_vector` across a provided series of 
        nodes and edges, keeping track of each probability distribution along 
        the way.

        Args:
            nodes (list): A list of node names to traverse along
            starting_vector (numpy.Array): A starting probability distribution
                from which to evolve.

        Returns:
            list of numpy.Array: A list of probability distribution vectors at 
            each stage of traversal.
        """
        vecs = []
        vec = starting_vector
        vecs.append(vec)

        for i in range(len(nodes)-1):
            # Evolve through current node
            vec = np.dot(vec, self.nodes[nodes[i]]['matrix'])
            vecs.append(vec)
            # Evolve through the subsequent link
            vec = np.dot(vec, self.edges[nodes[i], nodes[i+1]]['matrix'])
            vecs.append(vec)
        
        return(vecs)

    def trajectory(self, nodes, starting_vector):
        vecs = self.traverse(nodes, starting_vector)
        states = self._state_space.values
        run = []
        for v in vecs:
            val = np.random.choice(states, p=v)
            run.append(val)
        return run

    def trajectories(self, nodes, starting_vector, n=1):
        vecs = self.traverse(nodes, starting_vector)
        states = self._state_space.values
        runs = []
        for i in range(n):
            run = []
            for v in vecs:
                val = np.random.choice(states, p=v)
                run.append(val)
            runs.append(run)
        return runs


class StateSpace:
    def __init__(self, states):
        if len(states) <= 1:
            raise SizeError("State spaces must have more than one item")
        
        if len(set([s._name for s in states])) != len(states):
            raise DataError("States must have unique names")
        
        if len(set([s._value for s in states])) != len(states):
            raise DataError("States must have unique values")

        self._states = states
    
    @classmethod
    def from_list(cls, name_list, value_list):
        states = [State(i[0], i[1]) for i in zip(name_list, value_list)]
        return cls(states)
    
    @classmethod
    def from_csv(cls, filepath):
        with open(filepath, 'r') as infile:
            reader = csv.reader(infile)
            name_list = []
            value_list = []
            next(reader)
            for row in reader:
                name_list.append(row[0])
                value_list.append(float(row[1]))
        return cls.from_list(name_list, value_list)

    def value_at(self, name):
        return self.values[self.index_at(name)]
    
    def index_at(self, name):
        return self.names.index(name)
    
    @property
    def states(self):
        return self._states
    
    @property
    def values(self):
        return [s._value for s in self._states]
    
    @property
    def names(self):
        return [s._name for s in self._states]
    
    @property
    def size(self):
        return len(self.states)

    @property
    def mean(self):
        return float(sum([i._value for i in self._states]))/len(self._states)


class State:
    def __init__(self, name, value):
        self._name = name
        self._value = float(value)


def _check_matrix(matrix):
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise SizeError("Matrix is not square")
    for row in matrix:
        if sum(row) != 1.0:
            raise MatrixIntegrityError("Matrix rows do not all sum to one")
    return matrix