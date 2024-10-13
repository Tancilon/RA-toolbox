"""
Comb* Family Algorithm

This module implements the Comb* family of algorithms for rank aggregation.

Authors:
    fsw, tancilon
Date:
    2023-10-18
"""

import numpy as np

import scorefunc as sc
from src.rapython.datatools import *

__all__ = ['combmin']


def combmin_agg(input_list):
    """
    Aggregate minimum scores for items based on rankings provided by voters.
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
    item_comb_score = np.zeros(num_items)
    # Convert ranks to scores using different methods
    item_score = sc.linearagg(input_list)
    print(item_score)

    for i in range(num_items):
        item_min_score = np.zeros(num_voters)
        for k in range(num_voters):
            item_min_score[k] = item_score[k, i]
        item_comb_score[i] = min(item_min_score)

    first_row = item_comb_score
    # Sort and return indices of sorted elements
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result


def combmin(input_file_path, output_file_path):
    """
    Process the input CSV file to aggregate rankings and write the results to an output CSV file.
    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file.
    output_file_path : str
        Path to the output CSV file where the aggregated rankings will be written.
    """
    df, unique_queries = csv_load(input_file_path)
    # Create an empty DataFrame to store results
    result = []

    for query in unique_queries:
        # Filter data for the current Query
        query_data = df[df['Query'] == query]

        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(
            query_data)

        # Call function to get ranking information
        rank = combmin_agg(input_lists)

        # Add results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    save_as_csv(output_file_path, result)
