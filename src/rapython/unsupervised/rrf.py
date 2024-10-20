"""
Reciprocal Rank Fusion (RRF) Algorithm

This module implements the RRF method for rank aggregation.
This implementation is based on the following reference:

Reference:
-----------
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009, July). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758-759).

Authors:
    fsw, tancilon
Date:
    2023-10-19
"""

import numpy as np

from src.rapython.datatools import *

__all__ = ['rrf']


def rrf_agg(input_list):
    """
    Calculate the Reciprocal Rank Fusion (RRF) ranking for items based on the input rankings.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter's rankings of items.

    Returns
    -------
    result : ndarray
        A 1D numpy array containing the final rankings for each item based on RRF.
    """
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_borda_score = np.zeros(num_items)
    item_score = np.zeros((num_voters, num_items))

    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = 1 / (input_list[k, i] + 60)
            item_borda_score[i] += item_score[k, i]

    first_row = item_borda_score
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result


def rrf(input_file_path, output_file_path):
    """
    Process input data, compute rankings using the RRF method, and save the results to a CSV file.

    Parameters
    ----------
    input_file_path : str
        The path to the input CSV file containing the data.
    output_file_path : str
        The path to the output CSV file where results will be saved.
    """
    df, unique_queries = csv_load(input_file_path, InputType.RANK)
    result = []

    for query in unique_queries:
        query_data = df[df['Query'] == query]

        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(query_data)
        rank = rrf_agg(input_lists)

        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    save_as_csv(output_file_path, result)
