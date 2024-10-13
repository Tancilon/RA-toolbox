"""
Medium Algorithm

This module implements the Medium method for rank aggregation.
This implementation is based on the following reference:

Reference:
-----------
- Fagin, R., Kumar, R., & Sivakumar, D. (2003, June). Efficient similarity search and classification via rank aggregation. In Proceedings of the 2003 ACM SIGMOD international conference on Management of data (pp. 301-312).

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

__all__ = ['medium']


def medium_agg(input_list):
    """
    Calculate the median ranking for items based on the input rankings.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter's rankings of items.

    Returns
    -------
    result : ndarray
        A 1D numpy array containing the median rankings for each item.
    """
    num_voters = input_list.shape[0]  # Get the number of voters
    num_items = input_list.shape[1]  # Get the number of items
    item_mean_score = np.zeros(num_items)  # Initialize an array for median scores

    item_score = sc.linearagg(input_list)  # Calculate scores for items using linear aggregation

    for i in range(num_items):
        item_voters_score = np.zeros(num_voters)
        for k in range(num_voters):
            item_voters_score[k] = item_score[k, i]
        item_mean_score[i] = np.median(item_voters_score)  # Calculate the median score for item i

    first_row = item_mean_score
    sorted_indices = np.argsort(first_row)[::-1]  # Sort the indices based on median scores

    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result


def medium(input_file_path, output_file_path):
    """
    Process input data, compute median rankings using the medium_agg function,
    and save the results to a CSV file.

    Parameters
    ----------
    input_file_path : str
        The path to the input CSV file containing the data.
    output_file_path : str
        The path to the output CSV file where results will be saved.
    """
    df, unique_queries = csv_load(input_file_path)  # Load data from the CSV file
    result = []  # Initialize an empty list to store results

    for query in unique_queries:
        query_data = df[df['Query'] == query]  # Filter the DataFrame for the current query

        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(query_data)
        rank = medium_agg(input_lists)  # Get ranking information

        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]  # Get item code from mapping
            new_row = [query, item_code, item_rank]  # Create a new row for the result
            result.append(new_row)

    save_as_csv(output_file_path, result)  # Save the results to a CSV file
