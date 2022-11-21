import numpy as np
from typing import List

def rotate_shape(rotate_data: List, fixed_data: List) -> List:
    '''
    Function to rotate one set of shape data to minimise the distance beween points of another
    set of shape data.
    :param rotate_data: List,  The data to be rotated in the form of a list of x coordinates
    and a list of y coordinates.
    :param fixed_data: List, The fixed data to rotate the shape to in order to minimise the
    sum of the squared distance between the shapes.
    return: List, The rotated rotate_data.
    '''
    num_of_coords = len(rotate_data[0])

    angle_of_rotation = np.arctan((sum([rotate_data[0][i]*fixed_data[1][i] - rotate_data[1][i]*fixed_data[0][i] for i in range(len(rotate_data[0]))]))/
    (sum([rotate_data[0][i]*fixed_data[0][i] + rotate_data[1][i]*fixed_data[1][i] for i in range(num_of_coords)])))

    cos_angle_of_rotation = np.cos(angle_of_rotation)
    sin_angle_of_rotation = np.sin(angle_of_rotation)

    return [[cos_angle_of_rotation*rotate_data[0][i] - sin_angle_of_rotation*rotate_data[1][i] for i in range(num_of_coords)],
    [sin_angle_of_rotation*rotate_data[0][i] + cos_angle_of_rotation*rotate_data[1][i] for i in range(num_of_coords)]]

