"""
Borda Count Algorithm

This module implements the Borda Count method for rank aggregation.

Authors:
    fsw, tancilon
Date:
    2024-7-25

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

from src.rapython.datatools import *

__all__ = ['bordacount']


def borda_agg(input_list):
    """
    Aggregate Borda scores for items based on rankings provided by voters.

    Parameters
    ----------
    input_list : numpy.ndarray
        A 2D array where each row represents a voter's ranking of items.

    Returns
    -------
    numpy.ndarray
        An array containing the final ranks of the items after aggregation.
    """
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_borda_score = np.zeros(num_items)
    item_score = np.zeros((num_voters, num_items))

    # Calculate Borda scores for each item based on rankings
    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = num_items - input_list[k, i] + 1
            item_borda_score[i] += item_score[k, i]

    first_row = item_borda_score
    # Sort indices based on Borda scores in descending order
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result


def bordacount(input_file_path, output_file_path):
    """
    Process a CSV file containing voting data and output the aggregated ranks using Borda count.

    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file.
    output_file_path : str
        Path to the output CSV file.
    """
    df, unique_queries = csv_load(input_file_path)
    # Create an empty list to store results
    result = []

    for query in unique_queries:
        # Filter data for the current Query
        query_data = df[df['Query'] == query]
        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(
            query_data)

        # Call the function to get rank information
        rank = borda_agg(input_lists)

        # Add results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    # Write the results to the output CSV file
    save_as_csv(output_file_path, result)
