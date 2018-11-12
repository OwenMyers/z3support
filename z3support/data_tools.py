import numpy as np

EPSILON = 1e-5


def variance(array_in):
    """
    Given a list of numbers find the variance:

        <x^2> - <x>^2
    """

    assert isinstance(array_in, np.ndarray) or isinstance(array_in, list)

    if not isinstance(array_in, np.ndarray):
        array_in = np.array(array_in)

    squared_array = array_in**2

    return squared_array.mean() - array_in.mean()**2


def find_number_of_real_vertices_from_data(data):
    """
    Find the number of vertices of a given sublattice.

    This is mostly just to check the lattice size directory
    is likely commensurate with the data stored in it.

    :param data: 2D numpy array
    :return: the number of vertices for a given sublattice
    """

    counting_lines = 0

    for i, line in enumerate(data):
        if i == 0:
            continue

        counting_lines += 1

        if (line[0] < EPSILON) and (line[1] < EPSILON):
            return counting_lines

    raise Exception("Looks like, at best, there is only one bin")


def check_data(data):
    """
    Any general check on the data that you can think of.

    The point of this is to do a few simple checks to make sure
    the data file is good and obeys the standard you have been working
    with.
    :param data: 2D numpy array
    :return: Returns true if good but will fail because of assertion if bad.
    """

    # Make sure the first two numbers are 0, 0 which is the vertex at the origin
    assert data[0][0] < EPSILON
    assert data[0][1] < EPSILON
    # Make sure each row contains the x, y of the vertex and whatever the numbers/values
    # are at each of the links touching that vertex. 4 links plus 2D = length of 6
    assert len(data[0]) == 6

