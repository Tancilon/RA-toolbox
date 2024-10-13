"""
Dowdall Algorithm

This module implements the Dowdall method for rank aggregation.

Authors:
    fsw, tancilon
Date:
    2023-10-19

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

__all__ = ['dowdall']


def dowdall_agg(input_list):
    """
    Perform Dowdall aggregation on the given input list.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter and each column represents an item.
        The values in the array should be the scores given by each voter to each item.

    Returns
    -------
    result : ndarray
        A 1D numpy array where the index corresponds to the item, and the value at each index
        indicates the rank of that item based on the Dowdall aggregation method.
    """
    num_voters = input_list.shape[0]  # Number of voters
    num_items = input_list.shape[1]  # Number of items
    item_borda_score = np.zeros(num_items)  # Initialize item Borda scores
    item_score = np.zeros((num_voters, num_items))  # Initialize item scores for each voter

    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = 1 / input_list[k, i]  # Compute the score for each item
            item_borda_score[i] += item_score[k, i]  # Accumulate scores to Borda scores

    first_row = item_borda_score  # Store Borda scores for sorting
    # Sort the scores in descending order and return the sorted column indices
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1  # Initialize the current rank
    result = np.zeros(num_items)  # Initialize the result array for ranks
    for index in sorted_indices:
        result[index] = currrent_rank  # Assign rank to the corresponding item index
        currrent_rank += 1  # Increment the rank for the next item

    return result  # Return the final ranking of items


def dowdall(input_file_path, output_file_path):
    """
    Load data from a CSV file, perform Dowdall aggregation for each unique query,
    and save the results to a new CSV file.

    Parameters
    ----------
    input_file_path : str
        The file path of the input CSV file containing the voting data.

    output_file_path : str
        The file path where the output CSV file will be saved.

    Returns
    -------
    None
    """
    df, unique_queries = csv_load(input_file_path)  # Load the data and extract unique queries
    # Create an empty list to store the results
    result = []

    for query in unique_queries:
        # Filter data for the current query
        query_data = df[df['Query'] == query]

        # Map the query data to retrieve necessary structures for aggregation
        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(query_data)

        # Call the aggregation function to get ranking information
        rank = dowdall_agg(input_lists)

        # Append the results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]  # Retrieve item code
            new_row = [query, item_code, item_rank]  # Create a new row with query, item code, and rank
            result.append(new_row)  # Add the new row to the results

    save_as_csv(output_file_path, result)  # Save the results to a CSV file
