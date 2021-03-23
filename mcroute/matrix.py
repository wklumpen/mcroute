import csv
import numpy as np
import scipy.stats as sp
from mcroute.mcroute import DataError


def uniform(state_space):
    """
    Create a probability transition matrix with uniform probability
    distributions accross all rows.

    Args:
        state_space (StateSpace): The state space under which to build the matrix

    Returns:
        numpy.array: A matrix of transition probabilities
    """
    p = 1.0/state_space.size
    mx = []
    for row in range(state_space.size):
        r = [p for i in range(state_space.size)]
        r[-1] = 1.0 - sum(r[:-1])
        mx.append(r)
    return np.array(mx)


def truncnorm(state_space, mean=0, std=1):
    """
    Create a truncated normally distributed transition matrix across a given
    state space.

    The mean and standard deviations used in these calculations refer to state
    transition values. This means that a mean of zero centers the normal
    distribtuion around no state change (a delta of zero). Note that this
    probability distribution is truncated so that any probabilities that fall
    outside the possible transition space in each row are redistributed accross
    all probabilities within the allowed domain.

    Args:
        state_space (StateSpace): The state space under which to build the matrix
        mean (int, optional): The mean state transition value. Defaults to 0.
        std (int, optional): The standard deviation transition value. Defaults to 1.

    Returns:
        numpy.array: A matrix of transition probabilities
    """
    # Start by grabbing the state_values.
    state_values = state_space.values
    mtx = []
    for r in range(state_space.size):
        # The mean is that the delta between states is zero.
        # Get the first state jump
        row = []

        # Most negative jump possible
        a = state_values[0] - state_values[r]
        # Most positive jump possible
        b = state_values[-1] - state_values[r]
        for c in range(state_space.size):
            if c == 0: # First State
                p = sp.truncnorm.cdf(state_values[c] - state_values[r], a, b,  loc=mean, scale=std)
            elif c == state_space.size:  # Last state
                p = 1-sp.truncnorm.cdf(state_values[c] - state_values[r], a, b, loc=mean, scale=std)
            else:  # Do the 'normal' thing
                p = sp.truncnorm.cdf(state_values[c] - state_values[r], a, b, loc=mean, scale=std) - \
                    sp.truncnorm.cdf(state_values[c-1] - state_values[r], a, b, loc=mean, scale=std)
            row.append(p)
        row[-1] = 1.0 - sum(row[:-1])
        mtx.append(row)
        # print(f"Starting at {state_values[r]}: Domain {a} to {b}, sum {sum(row)}")
    return np.array(mtx)


def identity(state_space):
    """
    Create a probability transition matrix as an identity matrix.

    An identity matrix has a probability of 1.0 along the diagonal, and zero
    everywhere else.

    Args:
        state_space (StateSpace): The state space under which to build the matrix

    Returns:
        numpy.array: A matrix of transition probabilities
    """
    mtx = []
    for r in range(state_space.size):
        mtx.append([0.0 if c != r else 1.0 for c in range(state_space.size)])
    return np.array(mtx)


def from_csv(state_space, filepath):
    """
    Create a transition probability matrix using values stored in a CSV file

    The file must have equal rows and columns, and the number of rows and
    columns must match the size of the state space provided. Each row in the
    CSV file must also sum to 1.0.

    Args:
        filepath (str): The path to the file
        state_space (StateSpace): The state space under which to build the matrix

    Raises:
        DataError: An error will be thrown if the matrix size is invalid

    Returns:
        numpy.array: A matrix of transition probabilities
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
    """Creates a matrix based on a set of state transition delta indices.
    These transition deltas are integers, and indicate moving ahead or 
    behind a given reference state of a row. These values will be used
    for all rows in a transition matrix, shifted by the row index. For
    example, a value of '-2' will indicate a transition of two state indices
    backwards, regardless of the state types or names. This is useful for static
    matrices or where the data is assumed to apply accross all origin states
    regardless of the current state (transition magnitutdes are not wholly
    dependent on the origin state)."""
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
    return np.array(mtx)