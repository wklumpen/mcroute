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
    """The base network class for MCRoute analysis.

    A Network stores nodes and edges, and holds a copy of the state space for
    internal analysis.

    :param state_space: The state space under which the network is built.
    :type state_space: :class:`mcroute.StateSpace`
    """    
    def __init__(self, state_space):
        super().__init__()

        if not isinstance(state_space, StateSpace):
            raise TypeError("State space must be of type StateSpace")
        # Check for positive states
        self._state_space = state_space

    def add_node(self, name, matrix, data=None):
        """Add a node to the network.

        Nodes can include non-matrix data, but must include a matrix.

        :param name: The name of the node. Can be any Python object except none

        :type name: node
        :param matrix: The matrix of transition probabilites for the node.
        :type matrix: :class:`numpy.array`
        :param data: Additional data to attach to the node, defaults to None
        :type data: dict, optional
        """        
        if data:
            super().add_node(name, matrix=_check_matrix(matrix), **data)
        else:
            super().add_node(name, matrix=_check_matrix(matrix)) 
    
    def add_edge(self, u, v, matrix, data=None):
        """Add an edge between u and v.

        Nodes u and v must exist in the graph already. Edge attributes can be
        specified by passing a dictionary of key-value pairs via the data
        argument.

        :param u: The node from which to start the edge
        :type u: node
        :param v: The node to which to end the edge
        :type v: node
        :param matrix: The transition probability matrix
        :type matrix: :class:`numpy.array`
        :param data: A set of data arguments to store on the edge, defaults to 
            None
        :type data: dict, optional
        :raises DataError: Raises an error if either node is not initialized.
        """
        # Need to ensure nodes exist in database:
        if (u not in self.nodes) or (v not in self.nodes):
            raise DataError("One or both of the nodes are not in the network.")        
        if data:
            super().add_edge(u, v, matrix=_check_matrix(matrix), **data)
        else:
            super().add_edge(u, v, matrix=_check_matrix(matrix))
    
    def multi_step(self, path_nodes):
        """Return a multi-step transition probability matrix for a set of
        nodes.

        :param path_nodes: A list of node names to compute multi-state transitions
        :type path_nodes: list
        :return: A transition probability matrix representing the steps.
        :rtype: :class:`numpy.array`
        """
        mtx = self.nodes[path_nodes[0]]['matrix']
        for i in range(1, len(path_nodes)-1):
            u = path_nodes[i]
            v = path_nodes[i+1]
            mtx = np.dot(mtx, self.nodes[u]['matrix'])
            mtx = np.dot(mtx, self.edges[(u, v)]['matrix'])

        return mtx
    
    def traverse(self, nodes, starting_vector):
        """Traverse a path along a prescribed set of nodes

        The function evolves the `starting_vector` across a provided series of 
        nodes and edges, keeping track of each probability distribution along 
        the way.

        :param nodes: A list of node names to traverse along
        :type nodes: list
        :param starting_vector: A starting probability distribution
        :type starting_vector: :class:`numpy.array`
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

    def trajectories(self, nodes, starting_vector, n=1, smoothing=None):
        """Get a set of trajectories for a given path and initial probabilities.

        The trajectories are sampled from the distributions resulting from
        the evoluation of the intial ``starting_vector`` probability
        distribution. Trajectories can be smoothed by limiting the total size
        of the jump possible from one state to another.

        :param nodes: A list of nodes to traverse over as a path.
        :type nodes: list of node names
        :param starting_vector: The initial probability distribution vector.
        :type starting_vector: :class:`numpy.array`
        :param n: The number of trajectories to generate, defaults to 1
        :type n: int, optional
        :param smoothing: The smoothing parameter, defaults to None
        :type smoothing: numeric (int, float), optional
        :return: A list of state values sampled.
        :rtype: list
        """        
        vecs = self.traverse(nodes, starting_vector)
        runs = []
        for i in range(n):
            runs.append(self._get_trajectory(vecs, smoothing))
        if n == 1:
            return runs[0]
        else:
            return runs
    
    def _get_trajectory(self, vecs, smoothing):
        run = []
        for idx, v in enumerate(vecs):
            val = np.random.choice(self._state_space.values, p=v)
            # If it's not the first value and smoothing is turned on
            if idx > 0 and smoothing >= 0:
                jump = val - run[-1]
                if abs(jump) > smoothing:
                    if jump < 0:
                        val = run[-1] - smoothing
                    else:
                        val = run[-1] + smoothing
            run.append(val)
        return run


class StateSpace:
    """The state space class representing a Markov chain state space.

    A state space consists of a list of State objects containing two fields, a
    value (for numerical analysis) and a label (for reference and readability).
    
    .. note::
        State space names and values must be unique.

    :param states: A list of :class:`mcroute.States`
    :type state_space: list
    """    
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
        """Create a state space from a list of names and a list of values.

        :param name_list: The list of state labels or names
        :type name_list: list
        :param value_list: The list of values for numerical analysis or use
        :type value_list: list
        :raises DataError: :exc:`mcroute.DataError`
        :return: A constructed state space.
        :rtype: :class:`mcroute.StateSpace`
        """
        if len(name_list) != len(value_list):
            raise DataError("State name and value lists must be the same length")      
        states = [State(i[0], i[1]) for i in zip(name_list, value_list)]
        return cls(states)
    
    @classmethod
    def from_range(cls, low_state, high_state, interval=1):
        """Create a state space from a range of numbers.

        This method will create a state space with matching values and labels
        (i.e. a value of -2 will have a label of '-2'). Much like Python's own
        `range()` function, the high state is **not incldued**.

        :param low_state: The lower bound of the state space
        :type low_state: int
        :param high_state: The upper bound of the state space (not included)
        :type high_state: int
        :param interval: The interval between state values, defaults to 1
        :type interval: int, optional
        :return: A constructed state space.
        :rtype: :class:`mcroute.StateSpace`
        """        
        states = []
        for i in range(low_state, high_state, interval):
            states.append(State(str(i), float(i)))
        return cls(states)

    @classmethod
    def from_csv(cls, filepath):
        """Create a state space from a CSV file. The file must have two columns,
        the first being the name of the state, the second being the value
        associated with that name.

        :param filepath: The path to the CSV file containing the states
        :type filepath: str
        :return: A constructed state space
        :rtype: :class:`mcroute.StateSpace`
        """        
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
        """Get the value of a state with a given name.

        :param name: The name of the state to look up.
        :type name: str
        :return: The value of the state looked up.
        :rtype: numeric (float, int)
        """        
        return self.values[self.index_at(name)]
    
    def index_at(self, name):
        """Get the index of a state with a given name.

        :param name: The name of the state to look up.
        :type name: str
        :return: The value of the state looked up.
        :rtype: numeric (float, int)
        """        
        return self.names.index(name)
    
    @property
    def states(self):
        """Get the state objects in the state space.

        :return: A list of states in the state space.
        :rtype: list of :class:`mcroute.State`
        """         
        return self._states
    
    @property
    def values(self):
        """Get the state values in the state space.

        :return: A list of values in the state space.
        :rtype: list of numeric (int, float) values
        """  
        return [s._value for s in self._states]
    
    @property
    def names(self):
        """Get the state names in the state space.

        :return: A list of names in the state space.
        :rtype: list of str
        """  
        return [s._name for s in self._states]
    
    @property
    def size(self):
        """The number of states in the state space.

        :return: The size of the state space.
        :rtype: int
        """        
        return len(self.states)

    @property
    def mean(self):
        """The mean state value in the state space.

        :return: The mean value of all state values.
        :rtype: float
        """        
        return float(sum([i._value for i in self._states]))/len(self._states)


class State:
    """The class containing information about a state.

    :param name: The state's name
    :type name: str
    :param value: The state's value
    :type value: A numeric (int, float) value.
    """    
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