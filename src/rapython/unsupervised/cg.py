"""
Competitive Graph Algorithm

This module implements the competitive graph approach for rank aggregation.
This implementation is based on the following reference:

Reference:
-----------
- Xiao, Y., Deng, H. Z., Lu, X., & Wu, J. (2021). Graph-based rank aggregation method for high-dimensional and partial rankings. Journal of the Operational Research Society, 72(1), 227-236.

Authors:
    Shiwei Feng, Qi Deng
Date:
    2023-10-20

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

from common.constant import InputType
from src.rapython.datatools import *

__all__ = ['cg']


def cgagg(input_list):
    """
    Aggregate scores for items based on rankings provided by voters.
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
    indegree = np.zeros((num_voters, num_items))
    outdegree = np.zeros((num_voters, num_items))
    one = np.ones((num_voters, num_items))
    score = np.zeros((num_voters, num_items))
    score_sum = np.zeros(num_items)

    # Calculate indegree and outdegree for each voter-item pair
    for k in range(num_voters):
        for i in range(num_items):
            indegree[k, i] = indegree[k, i] + input_list[k, i] - 1
            outdegree[k, i] = num_items * num_voters - indegree[k, i]

    indegree += one
    outdegree += one

    # Compute scores based on outdegree and indegree
    for k in range(num_voters):
        for i in range(num_items):
            score[k, i] = outdegree[k, i] / indegree[k, i]

    # Sum scores across all voters for each item
    for i in range(num_items):
        item_sum_score = np.zeros(num_voters)
        for k in range(num_voters):
            item_sum_score[k] = score[k, i]
        score_sum[i] = sum(item_sum_score)

    first_row = score_sum
    # Sort indices based on aggregated scores
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    return result


def cg(input_file_path, output_file_path):
    """
    Process the input CSV file to aggregate rankings and output the results.
    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file containing voting data.
    output_file_path : str
        Path to the output CSV file where the aggregated rankings will be saved.

    Returns
    -------
    None
    """
    df, unique_queries = csv_load(input_file_path, InputType.RANK)
    # Create an empty DataFrame to store results
    result = []

    for query in unique_queries:
        # Filter data for the current Query
        query_data = df[df['Query'] == query]

        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(
            query_data)

        # Call the aggregation function to get ranks
        rank = cgagg(input_lists)

        # Add results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    # Write the results to the output CSV file
    save_as_csv(output_file_path, result)


if __name__ == "__main__":
    input_path = r'D:\LocalGit\RA-toolbox\datasets\ice-cream\ice-cream.csv'
    output_path = r'D:\LocalGit\RA-toolbox\src\rapython\unsupervised\test.csv'
    cg(input_path, output_path)
