"""
Mean Algorithm

This module implements the Mean method for rank aggregation.
This implementation is based on the following reference:

Reference:
-----------
- Kaur, M., Kaur, P., & Singh, M. (2015, September). Rank aggregation using multi objective genetic algorithm. In 2015 1st International Conference on Next Generation Computing Technologies (NGCT) (pp. 836-840). IEEE.

Authors:
    fsw, tancilon
Date:
    2023-10-18

Input Format:
-------------
The input to the algorithm should be in CSV file format with the following columns:
- Query: Does not require consecutive integers starting from 1.
- Voter Name: Allowed to be in string format.
- Item Code: Allowed to be in string format.
- Item Rank: Represents the rank given by each voter.

Output Format:
--------------
The final output of the algorithm will be in CSV file format with the following columns:
- Query: The same as the input.
- Item Code: The same as the input.
- Item Rank: The rank information (not the score information).
  - Note: The smaller the Item Rank, the higher the rank.
"""

import numpy as np

import scorefunc as sc
from src.rapython.datatools import *

__all__ = ['mean']


def mean_agg(input_list):
    """
    Calculate the mean ranking for items based on the input rankings.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter's rankings of items.

    Returns
    -------
    result : ndarray
        A 1D numpy array containing the mean rankings for each item.
    """
    num_voters = input_list.shape[0]  # Get the number of voters
    num_items = input_list.shape[1]  # Get the number of items
    item_mean_score = np.zeros(num_items)

    # Calculate scores for items using linear aggregation
    item_score = sc.linearagg(input_list)

    # Compute the mean score for each item based on voter scores
    for i in range(num_items):
        item_voters_score = np.zeros(num_voters)
        for k in range(num_voters):
            item_voters_score[k] = item_score[k, i]
        item_mean_score[i] = np.mean(item_voters_score)

    first_row = item_mean_score
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1
    result = np.zeros(num_items)  # Initialize the result array for ranks
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result  # Return the array containing rankings


def mean(input_file_path, output_file_path):
    """
    Process input data, compute mean rankings using the mean_agg function,
    and save the results to a CSV file.

    Parameters
    ----------
    input_file_path : str
        The path to the input CSV file containing the data.
    output_file_path : str
        The path to the output CSV file where results will be saved.
    """
    df, unique_queries = csv_load(input_file_path)  # Load data from the CSV file
    result = []

    for query in unique_queries:
        # Filter the DataFrame for the current query
        query_data = df[df['Query'] == query]

        # Map item codes and prepare input lists for aggregation
        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(query_data)

        # Call the mean_agg function to get ranking information
        rank = mean_agg(input_lists)

        # Append the results to the output list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]  # Get item code from mapping
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    save_as_csv(output_file_path, result)  # Save the results to a CSV file
