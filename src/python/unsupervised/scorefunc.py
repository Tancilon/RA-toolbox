import math
import numpy as np


def linearagg(input_list):
    """
    Calculate the linear aggregated scores.

    Args:
        input_list (numpy.ndarray): A 2D array representing votes from voters,
                                    where rows are voters and columns are candidates.

    Returns:
        numpy.ndarray: A 2D array containing the scores for each candidate,
                       with rows being voters and columns being candidates.
    """
    # Initialize variables
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_score = np.zeros((num_voters, num_items))

    # Compute linear aggregated scores
    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = num_items - input_list[k, i] + 1

    return item_score


def reciprocalagg(input_list):
    """
    Calculate the reciprocal aggregated scores.

    Args:
        input_list (numpy.ndarray): A 2D array representing votes from voters,
                                    where rows are voters and columns are candidates.

    Returns:
        numpy.ndarray: A 2D array containing the scores for each candidate,
                       with rows being voters and columns being candidates.
    """
    # Initialize variables
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_score = np.zeros((num_voters, num_items))

    # Compute reciprocal aggregated scores
    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = 1 / input_list[k, i]

    return item_score


def poweragg(input_list):
    """
    Calculate the power aggregated scores.

    Args:
        input_list (numpy.ndarray): A 2D array representing votes from voters,
                                    where rows are voters and columns are candidates.

    Returns:
        numpy.ndarray: A 2D array containing the scores for each candidate,
                       with rows being voters and columns being candidates.
    """
    # Initialize variables
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_score = np.zeros((num_voters, num_items))

    # Compute power aggregated scores
    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = math.pow(1.1, num_items - input_list[k, i])

    return item_score


def logagg(input_list):
    """
    Calculate the logarithmic aggregated scores.

    Args:
        input_list (numpy.ndarray): A 2D array representing votes from voters,
                                    where rows are voters and columns are candidates.

    Returns:
        numpy.ndarray: A 2D array containing the scores for each candidate,
                       with rows being voters and columns being candidates.
    """
    # Initialize variables
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_score = np.zeros((num_voters, num_items))

    # Compute logarithmic aggregated scores
    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = math.log(input_list[k, i], 0.1)

    return item_score
