"""
Matrix construction and manipulation functions for MCRoute networks.

This module contains methods that enable the constructing of matrices needed to
construct a MCRoute network. Typically, these methods require a defined
:class:`mcroute.StateSpace` object which is used to provide context for building
distributions, reading files, or validating data. Matrices produced by this
module are validated and can be used for any node or edge creation. For example:

.. code-block:: Python

    from mcroute import StateSpace
    import mcroute.matrix as matrix

    space = StateSpace.from_range(-2, 3)
    matrix.uniform(space)

will produce a uniformly distributed matrix of dimension 5. All matrix creation
methods in this module will return :class:`numpy.array` objects.
"""
import csv
import numpy as np

import scipy.stats as sp
import networkx as nx

from mcroute.mcroute import DataError

def _tbeta_cdf(x, a, b, alpha, beta) -> float:
    """Calcualte a truncated beta distribution cumulative probability.

    The beta distribution is supported only on the interval [0,1] and values
    passed must therefore be scaled appropriately.

    :param x: The CDF value to calculate. 
    :type x: float
    :param a: The lower bound of the distribution
    :type a: float
    :param b: The upper bound of the distribution
    :type b: float
    :param alpha: The alpha (or `a`) parameter. Must be strictly positive.
    :type alpha: float
    :param beta: The beta (or `b`) parameter. Must be strictly positive.
    :type beta: float
    :return: A cumulative probability
    :rtype: float
    """
    return (sp.beta.cdf(x, alpha, beta) - sp.beta.cdf(a, alpha, beta))/ \
        (sp.beta.cdf(b, alpha, beta) - sp.beta.cdf(a, alpha, beta))

def _tlognorm_cdf(x, b, mu, sigma) -> float:
    """Calculate a truncated log-normal distribution cumulative probability.

    :param x: The CDF value to calculate
    :type x: float
    :param b: The upper bound of the distribtuion 
    :type b: float
    :param mu: the mu (loc)
    :type mu: float
    :param sigma: the sigma (s) value of the distribtuion
    :type sigma: float
    """
    return sp.lognorm.cdf(x, sigma, loc=mu)/sp.lognorm.cdf(b, sigma, loc=mu)

def uniform(state_space):
    """Create a unfiromally distributed transition probability matrix.

    The resulting probability transition matrix will have equal probabilities
    accross all state transitions.

    :param state_space: The state space under which to build the matrix
    :type state_space: :class:`mcroute.StateSpace`
    :return: A matrix of transition probabilities
    :rtype: :class:`numpy.array`
    """
    p = 1.0/state_space.size
    mx = []
    for row in range(state_space.size):
        r = [p for i in range(state_space.size)]
        r[-1] = 1.0 - sum(r[:-1])
        mx.append(r)
    return np.array(mx)


def truncnorm(state_space, mean=0, std=1):
    """Create a truncated normally distributed transition matrix accross a given
    state space.

    The mean and standard deviations used in these calculations refer to `state
    transition values`. This means that a mean of zero centers the normal
    distribtuion around no state change (a delta of zero). Note that this
    probability distribution is truncated so that any probabilities that fall
    outside the possible transition space in each row are redistributed accross
    all probabilities within the allowed domain.

    :param state_space: The state space under which to build the matrix
    :type state_space: :class:`mcroute.StateSpace`
    :param mean: The mean value of the distribtuion, defaults to 0
    :type mean: int, optional
    :param std: The standard deviation of the distribution, defaults to 1
    :type std: int, optional
    :return: A transition probability matrix
    :rtype: :class:`numpy.array`
    """
    # Start by grabbing the state_values.
    state_values = state_space.values
    mtx = []
    for r in range(state_space.size):
        # The mean is that the delta between states is zero.
        # Get the first state jump
        rowData = []

        # Most negative jump possible
        a = state_values[0] - state_values[r]
        # Most positive jump possible
        b = state_values[-1] - state_values[r]
        for c in range(state_space.size):
            row = state_values[r]
            col = state_values[c]
            if c < state_space.size-1:
                col = state_values[c]
                if c == 0: # First State
                    p = sp.truncnorm.cdf(col - row, a, b,  loc=mean, scale=std)
                else:  # Do the 'normal' thing
                    p = sp.truncnorm.cdf(col - row, a, b, loc=mean, scale=std) - \
                        sp.truncnorm.cdf(state_values[c-1] - row, a, b, loc=mean, scale=std)
            elif c == state_space.size:  # Last state
                    p = 1-sp.truncnorm.cdf(col - row, a, b, loc=mean, scale=std)
            rowData.append(p)
        rowData[-1] = 1.0 - sum(rowData[:-1])
        mtx.append(rowData)
    return np.array(mtx)

def truncexpon(state_space, mu=1.0, sigma=1.0):
    """Create a truncated exponentially distributed probability transition 
    matrix accross a given state space.

    Note that the exponential distribution only supports state values in the
    positive domain. Other state values will be assigned a probabiliy of zero.

    The mean and shape values used in these calculations refer to `state
    transition values`. This means that the distributions created are centered
    around a given state transition, and are shifted by each row.
    
    Note that this probability distribution is truncated so that any 
    probabilities that fall outside the possible positive transition space in
    each row are redistributed accross all probabilities within the allowed 
    domain.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncexpon.html
    for more information on the truncated exponential distribution.

    :param state_space: The state space under which to build the matrix
    :type state_space: :class:`mcroute.StateSpace`
    :param mu: The lambda (loc) of the exponential distribution, defaults to 1
    :type mu: float, optional
    :param sigma: The scale value of the exponential distribution, defaults to 1
    :type sigma: float, optional
    :return: A truncated exponential matrix of transition probabilities.
    :rtype: :class:`numpy.array`
    """
    # Start by grabbing the state_values.
    state_values = state_space.values
    mtx = []
    for r in range(state_space.size):
        # The mean is that the delta between states is zero.
        # Get the first state jump
        rowData = []
        # Most positive jump possible
        b = state_values[-1] - state_values[r]
        for c in range(state_space.size):
            row = state_values[r]
            col = state_values[c]
            if c < state_space.size-1:
                col = state_values[c]
                if col - row < 0:
                    # Negative values are set to zero.
                    p = 0.0
                else:
                    if c == 0: # First State
                        p = sp.truncexpon.cdf(col - row, b,  loc=mu, scale=sigma)
                    else:  # Do the 'normal' thing
                        p = sp.truncexpon.cdf(col - row, b, loc=mu, scale=sigma) - \
                            sp.truncexpon.cdf(state_values[c-1] - row, b, loc=mu, scale=sigma)
            elif c == state_space.size:  # Last state
                p = 1-sp.truncexpon.cdf(col - row, b, loc=mu, scale=sigma)
            rowData.append(p)

        # Final step to ensure it sums to 1.
        rowData[-1] = 1.0 - sum(rowData[:-1])
        mtx.append(rowData)
    return np.array(mtx)

def trunclognorm(state_space, mu=0, sigma=1):
    """Create a truncated log-normally distributed probability transition 
    matrix accross a given state space.

    The parameters mu and sigma follow the definition used by Wikipedia: 
    https://en.wikipedia.org/wiki/Log-normal_distribution. For conversion to
    `scipy`'s format, `sigma` becomes `s`, `mu` becomes `loc`, and the Scipy
    `scale` parameter is held as it's default 1.
    
    Note: The log-normal distribution only supports state values in the
    positive domain. Other state values will be assigned a probabiliy of zero.
    This probability distribution is truncated so that any probabilities that 
    fall outside the possible positive transition space in each row are 
    redistributed accross all probabilities within the allowed domain.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
    for more information on the `scipy` lognormal distribution.

    :param state_space: The state space under which to build the matrix
    :type state_space: :class:`mcroute.StateSpace`
    :param mu: The mu parameter (loc) of the log-normal distribtuion
    :type mu: float, optional
    :param sigma: The sigma value (s) of the log-normal distribution
    :type sigma: float, optional
    :return: A truncated lognormal matrix of transition probabilities.
    :rtype: :class:`numpy.array`
    """
    # Start by grabbing the state_values.
    state_values = state_space.values
    mtx = []
    for r in range(state_space.size):
        # The mean is that the delta between states is zero.
        # Get the first state jump
        rowData = []
        # Most positive jump possible
        b = state_values[-1] - state_values[r]
        for c in range(state_space.size):
            row = state_values[r]
            col = state_values[c]
            if c < state_space.size-1:
                col = state_values[c]
                if col - row < 0:
                    # Negative values are set to zero.
                    p = 0.0
                else:
                    if c == 0: # First State
                        p = _tlognorm_cdf(col - row, b,  mu, sigma)
                    else:  # Do the 'normal' thing
                        p = _tlognorm_cdf(col - row, b, mu, sigma) - \
                            _tlognorm_cdf(state_values[c-1] - row, b, mu, sigma)
            elif c == state_space.size:  # Last state
                p = 1-_tlognorm_cdf(col - row, b, mu, sigma)
            rowData.append(p)

        # Final step to ensure it sums to 1.
        rowData[-1] = 1.0 - sum(rowData[:-1])
        mtx.append(rowData)
    return np.array(mtx)


def beta(state_space, alpha, beta):
    """Create a beta distributed transition probability matrix accross
    a given state space.

    A reader is referred to https://en.wikipedia.org/wiki/Beta_distribution
    for examples of the shapes that various values of alpha and beta will
    produce.

    The beta distribution's support lies on the interval [0,1]. To account for
    this, values are first normalized based on the highest and lowest jumps
    possible accross the entire matrix (the magnitude of which is the last value
    in the state space minus the first). Then, the distribution is truncated
    based on the available transition space in a given row of the matrix.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
    for more information on the beta distribution.

    :param state_space: The state space under which to build the matrix
    :type state_space: :class:`mcroute.StateSpace`
    :param alpha: The alpha (or `a`) parameter. Must be strictly positive.
    :type alpha: float
    :param beta: The beta (or `b`) parameter. Must be strictly positive.
    :type beta: float
    :return: A truncated beta matrix of transition probabilities.
    :rtype: :class:`numpy.array`
    """
        # Start by grabbing the state_values.
    state_values = state_space.values
    mtx = []
    tip = state_values[-1] - state_values[0]  # Largest possible jump in the mtx
    toe = state_values[0] - state_values[-1]  # Smallest possible jump in mtx
    span = tip - toe
    print(toe, tip, span)
    for r in range(state_space.size):
        # Get the first state jump
        rowData = []
        print("Row", r)
        # Most negative jump possible
        a = state_values[0] - state_values[r]
        a = (a - toe)/span
        # Most positive jump possible
        b = state_values[-1] - state_values[r]
        b = (b - toe)/span
        for c in range(state_space.size):
            row = state_values[r]
            col = state_values[c]
            x = ((col - row) - toe)/span
            print(x)
            if c < state_space.size-1:
                if c == 0: # First State
                    p = _tbeta_cdf(x, a, b, alpha, beta)
                else:  # Do the 'normal' thing
                    p = _tbeta_cdf(x, a, b, alpha, beta) - \
                        _tbeta_cdf(((state_values[c-1] - row) - toe)/span, a, b, alpha, beta)
            elif c == state_space.size:  # Last state
                p = _tbeta_cdf(x, a, b, alpha, beta)
            rowData.append(p)

        # Final step to ensure it sums to 1.
        rowData[-1] = 1.0 - sum(rowData[:-1])
        mtx.append(rowData)
    return np.array(mtx)

def identity(state_space):
    """Create an identity probability transition matrix.

    An identity matrix has probabilities of 1.0 along the diagonal, and zero
    everywhere else.

    :param state_space: The state space under which to build the matrix.
    :type state_space: :class:`mcroute.StateSpace`
    :return: An identity matrix of transition probabilities.
    :rtype: :class:`numpy.array`
    """
    return np.eye(state_space.size)

def truncate_below(state_space, matrix, treshold_state_name):
    """Truncate a probability transition matrix below a given state

    :param state_space: The state space under which to act on the matrix
    :type state_space: :py:class:`mcroute.StateSpace`
    :param matrix: The matrix to truncate
    :type matrix: :class:`numpy.array`
    :param treshold_state_name: The name of the state at which to truncate
    :type treshold_state_name: str
    :return: A truncated matrix of transition probabilities
    :rtype: :class:`numpy.array`
    """
    idx = state_space.names.index(treshold_state_name)
    mx = []
    for r in range(state_space.size):
        row = []
        for c in range(state_space.size):
            if c < idx:
                row.append(0.0)
            elif c == idx:
                row.append(matrix[r][c] + sum(matrix[r][:idx]))
            else:
                row.append(matrix[r][c])
        mx.append(row)
    return np.array(mx)

def shift(matrix, idx=0):
    """Update a matrix with probabilities shifted by an index amount.

    Each row in the matrix will have probabilities that are shifted by the
    indiex provided. Probabilities that are 'pushed off' the edge are reassigned
    to the ends of the distribution (the edge states), so truncation effects are
    possible.

    :param matrix: The matrix to update
    :type matrix: :class:`numpy.array`
    :param idx: The number of indices to shift the distribution by, 
        defaults to 0
    :type idx: int, optional
    :return: The shifted matrix
    :rtype: :class:`numpy.array`
    """
    mx = []
    for row in matrix:
        newRow = [0.0 for i in range(len(row))]
        for c, col in enumerate(row):
            ## Keep the index in the probability
            newIdx = min(max(c + idx , 0), len(row)-1)
            newRow[newIdx] += col
        mx.append(newRow)
    return np.array(mx)


def from_csv(state_space, filepath):
    """Create a transition probability matrix using values stored in a CSV file

    :param state_space: The state space under which to create the matrix
    :type state_space: :class:`mcroute.StateSpace`
    :param filepath: The filepath to the matrix values
    :type filepath: str
    :raises: :exc:`mcroute.DataError` - An error will be thrown if the matrix 
        size is invalid.
    :return: The created matrix.
    :rtype: :class:`np.Array`
    """
    with open(filepath, 'r') as infile:
        reader = csv.reader(infile)
        mx = []
        for row in reader:
            if len(row) != state_space.size:
                raise DataError(f"Invalid matrix size for file {filepath} with a state space size of {state_space.size}")
            mx.append([float(i) for i in row])
        if len(mx) != state_space.size:
                raise DataError(f"Invalid matrix size for file {filepath} with a state space size of {state_space.size}")
    return np.array(mx)

def from_state_delta_csv(state_space, filepath):
    """Create a matrix using a set of state transition data.

    Transition data must be in the form of integers represending how many
    incidces a state transition shifted during the transition. Positive values
    shift forward on the state space, negative values shift backwards.

    For example, a value of '-2' indicates a transition of two state indices
    backwards, regardless of state values or names.

    :param state_space: The state space under which to construct the matrix
    :type state_space: :class:`mcroute.StateSpace`
    :param filepath: The filepath containing the state transitions.
    :type filepath: str
    :raises: :exc:`ZeroDivisionError` - Will throw an error when a state transition row
        is missing data, resulting in a zero division during normalization.
    :return: The populated matrix.
    :rtype: :class:`np.Array`

    .. note::
        This method effectively assumes that state transition values are evenly
        spaced. Use with caution under other conditions.
    """
    # Start with an empty matrix
    mtx = [[0.0 for i in range(state_space.size)] for j in range(state_space.size)]
    delta_dict = dict()
    count = 0
    with open(filepath, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for row in reader:
            count += 1
            delta = int(row[0])
            if delta in delta_dict:
                delta_dict[delta] += 1
            else:
                delta_dict[delta] = 1

    # Normalize the dictionary:
    for key in delta_dict:
        delta_dict[key] = delta_dict[key]/float(count)
    
    # Normalize the rows
    for r in range(state_space.size):
        for c in range(state_space.size):
            d = c - r
            if d in delta_dict:
                mtx[r][c] = delta_dict[d]
            else:
                mtx[r][c] = 0.0

            if c == 0:
                # Get all the earlier deltas in there too:
                for key in delta_dict:
                    if key < d:
                        mtx[r][c] += delta_dict[key]
            elif c == state_space.size-1:
                for key in delta_dict:
                    if key > d:
                        mtx[r][c] += delta_dict[key]


    # Normalize the rows
    for r in range(state_space.size):
        row_sum = sum(mtx[r])
        for c in range(state_space.size):
            try:
                mtx[r][c] = mtx[r][c]/row_sum
            except ZeroDivisionError:
                print(r, c, row_sum)
                print(mtx[r])
                raise ZeroDivisionError
    
        for x in range(100):
            if sum(mtx[r]) == 1.0:
                break
            if sum(mtx[r]) > 1.0:
                # We need to subtract
                diff = sum(mtx[r]) - 1.0
                indices = [idx for idx, val in enumerate(mtx[r]) if val > 0.0 ]
                each = diff/len(indices)
                for ind in indices:
                    mtx[r][ind] -= each
            # Correct for over-subtraction
            mtx[r] = [0.0 if i < 0.0 else i for i in mtx[r]]
            if sum(mtx[r]) < 1.0:
                # We need to add
                diff = 1.0 - sum(mtx[r])
                each = diff/len(mtx[r])
                for c in range(len(mtx[r])):
                    mtx[r][c] += each

        # Final correction
        mtx[r][-1] = 1.0 - sum(mtx[r][:-1])
    return np.array(mtx)

def absorbing_classes(matrix):
    """Summarize the absorbing classes for a given matrix.

    Absorbing classes are sets of one or more states that once entered, cannot
    leave. In certain cases, these absorbing classes can be indicative of data
    issues or design issues in the matrices.
    
    :param matrix: The matrix to examine
    :type matrix: :class:`numpy.array`
    :return: A list of sets of absorbing classes in the matrix, by index.
    :rtype: list
    """    
    g = nx.from_numpy_matrix(np.array(matrix), create_using=nx.DiGraph())
    internal = set()

    # Tarjan is approximately linear O(|V| + |E|)
    cpts = list(nx.strongly_connected_components(g))
    for cpt in cpts:
        sg = g.subgraph(cpt)
        for e in sg.edges:
            internal.add(e)

    # find all the edges that aren't part of the strongly connected components
    # ~ O(E)
    transient_edges = set(g.edges()) - internal
    # find the start of the directed edge leading out from a component
    # ~ O(E)
    transient_srcs = set([e[0] for e in transient_edges])

    # Assemble everything that don't have a vertex in transient_srcs
    abs_classes = []
    for sg in cpts:
        if transient_srcs - sg:
            abs_classes.append(sg)

    return abs_classes

def steady_state(matrix):
    """Determine the steady-state transition probability vector for a matrix.

    The steady-state transition probability vector is a vector that, when
    multiplied by the provided matrix, returns the same vector. This represents
    behaviour as the number of transitions trends towards infinity.

    .. note::
        This method uses a least squares approach to solving the system of
        equations, which allows for the number of linearly independent rows
        of the matrix to be less, equal to, or greater than the number of
        linearly independent columns. Since some transition probability matrices
        can have multiple solutions (periodic ones, for example), this solution
        minimizes the Euclidean 2-norm. See numpy's documentation on
        `numpy.linalg.lstsq` for further details.

    :param matrix: The matrix to calcualte steady-state probabilities for
    :type matrix: :class:`numpy.array`
    :return: A steady-state transition probability vector
    :rtype: :class:`numpy.array` (1-dimensional)
    """    
    dim = matrix.shape[0]
    Q = (matrix-np.eye(dim))
    e = np.ones(dim)
    QT = np.c_[Q, e].T
    b = np.append(np.zeros(dim), 1.0)
    pi, residuals, rank, s = np.linalg.lstsq(QT,b.T, rcond=None)
    return pi