"""
Web Metasearch: Rank vs. Score Based Rank Aggregation Methods

This module implements the MC1-4 method for rank aggregation using the Markov Chain approach.

Authors:
    tancilon
Date:
    2023-09-26

Input Format:
-------------
The top-level input for the Markov Chain algorithm should be in CSV file format with the following columns:
- Query: Does not require consecutive integers starting from 1.
- Voter Name: Allowed to be in string format.
- Item Code: Allowed to be in string format.
- Item Rank: Represents the rank given by each voter.

Output Format:
--------------
The final output of the Markov Chain algorithm will be in CSV file format with the following columns:
- Query: The same as the input.
- Item Code: The same as the input.
- Item Rank: The rank information (not the score information).
  - Note: The smaller the Item Rank, the higher the rank.

Function Description:
---------------------
The `MC()` function takes `input_list` as the input ranking information, which is a 2D array of shape Voters x Items. The array stores the ranking values for the items, where smaller values indicate a higher rank.
- Note: If a Voter[i] does not rank a specific item[j], then `input_list[i][j]` should be set to NaN.
- Note: The input ranking accepts partial lists.
"""

from enum import Enum, auto

import numpy as np

from src.rapython.datatools import *

__all__ = ['markovchainmethod']


class MethodType(Enum):
    MC1 = auto()
    MC2 = auto()
    MC3 = auto()
    MC4 = auto()


def get_mc1_transfer_matrix(input_list):
    """
    Calculate the Markov Chain transfer matrix using the first method.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter and each column represents an item.
        The values in the array should be the rankings given by each voter to each item.

    Returns
    -------
    pmatrix : ndarray
        A 2D numpy array representing the transfer matrix, where the entry pmatrix[i][j]
        indicates the probability of transitioning from item i to item j.
    """
    num_items = input_list.shape[1]  # Number of items
    # Initialize the probability transition matrix pmatrix
    pmatrix = np.zeros((num_items, num_items))

    # Iterate over each item i
    for i in range(num_items):
        # Initialize a list to store all items ranked higher than or equal to i
        ranked_higher_or_equal = []

        # Iterate over each voter's ranking
        for voter_ranking in input_list:
            # Check if the current voter's ranking includes item i
            if not np.isnan(voter_ranking[i]):
                # Get the ranking position of item i
                i_rank = int(voter_ranking[i])

                # Add items ranked higher or equal to i into the list
                for j in range(num_items):
                    if not np.isnan(voter_ranking[j]) and int(voter_ranking[j]) <= i_rank:
                        ranked_higher_or_equal.append(j)

        # Calculate the probability of transitioning from state i to state j, uniformly distributed
        total_ranked = len(ranked_higher_or_equal)
        if total_ranked > 0:
            probability = 1.0 / total_ranked  # Calculate the uniform transition probability
            for j in ranked_higher_or_equal:
                pmatrix[i][j] += probability  # Update the transition matrix

    # Normalize the probability transition matrix so that the sum of probabilities in each row equals 1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix  # Return the transition matrix


def get_mc2_transfer_matrix(input_list):
    """
    Calculate the Markov Chain transfer matrix using the second method.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter and each column represents an item.
        The values in the array should be the rankings given by each voter to each item.

    Returns
    -------
    pmatrix : ndarray
        A 2D numpy array representing the transfer matrix, where the entry pmatrix[i][j]
        indicates the probability of transitioning from item i to item j.
    """
    num_items = input_list.shape[1]  # Number of items
    # Initialize the probability transition matrix pmatrix
    pmatrix = np.zeros((num_items, num_items))

    # Iterate over each item i
    for i in range(num_items):
        # Find indices of voters who ranked item i
        voters_with_i = [voter_index for voter_index, rankings in enumerate(input_list) if not np.isnan(rankings[i])]

        # If no voters ranked item i, skip to the next item
        if len(voters_with_i) == 0:
            continue

        voter_probability = 1.0 / len(voters_with_i)  # Calculate probability for each voter
        # Iterate over each voter who ranked item i
        for voter_i in voters_with_i:
            # Find indices of all items j ranked higher or equal to item i for this voter
            ranked_higher_or_equal = [j for j, ranking in enumerate(input_list[voter_i]) if
                                      not np.isnan(ranking) and ranking <= input_list[voter_i][i]]

            # If no such items, skip to the next voter
            if len(ranked_higher_or_equal) == 0:
                continue

            # Calculate the probability of transitioning from state i to state j, uniformly distributed
            probability = 1.0 / len(ranked_higher_or_equal)
            for j in ranked_higher_or_equal:
                pmatrix[i][j] += probability * voter_probability  # Update the transition matrix

    # Normalize the probability transition matrix so that the sum of probabilities in each row equals 1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix  # Return the transition matrix


def get_mc3_transfer_matrix(input_list):
    """
    Calculate the Markov Chain transfer matrix using the third method.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter and each column represents an item.
        The values in the array should be the rankings given by each voter to each item.

    Returns
    -------
    pmatrix : ndarray
        A 2D numpy array representing the transfer matrix, where the entry pmatrix[i][j]
        indicates the probability of transitioning from item i to item j.
    """
    num_items = input_list.shape[1]  # Get the number of items
    pmatrix = np.zeros((num_items, num_items))  # Initialize the probability transition matrix

    # Iterate over each item i
    for i in range(num_items):
        # Find indices of voters who ranked item i
        voters_with_i = [voter_index for voter_index, rankings in enumerate(input_list) if not np.isnan(rankings[i])]

        # Skip if no voters ranked item i
        if len(voters_with_i) == 0:
            continue

        voter_probability = 1.0 / len(voters_with_i)  # Calculate probability for each voter
        # Iterate over each voter who ranked item i
        for voter_i in voters_with_i:
            # Count the number of non-NaN rankings for this voter
            non_nan_count = sum(1 for element in input_list[voter_i] if not np.isnan(element))
            item_probability = 1.0 / non_nan_count  # Calculate probability for each item

            # Update the transition probabilities
            for j in range(num_items):
                if not np.isnan(input_list[voter_i][j]) and int(input_list[voter_i][j]) < input_list[voter_i][i]:
                    pmatrix[i][j] += voter_probability * item_probability

    # Assign values to the diagonal of the transition matrix
    for i in range(num_items):
        diagonal_sum = np.sum(pmatrix[i]) - pmatrix[i][i]  # Sum of off-diagonal entries in row i
        pmatrix[i][i] = 1 - diagonal_sum  # Set the diagonal entry to ensure row sums to 1

    # Normalize the transition matrix so that each row sums to 1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix  # Return the transition matrix


def get_mc4_transfer_matrix(input_list):
    """
    Calculate the Markov Chain transfer matrix using the fourth method.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter and each column represents an item.
        The values in the array should be the rankings given by each voter to each item.

    Returns
    -------
    pmatrix : ndarray
        A 2D numpy array representing the transfer matrix, where the entry pmatrix[i][j]
        indicates the probability of transitioning from item i to item j.
    """
    num_items = input_list.shape[1]  # Get the number of items
    num_voters = len(input_list)  # Get the number of voters
    pmatrix = np.zeros((num_items, num_items))  # Initialize the probability transition matrix

    # Iterate over each pair of items (i, j)
    for i in range(num_items):
        for j in range(num_items):
            if i == j:
                pmatrix[i][j] += 1  # Keep the diagonal value at 1
                continue  # Skip this iteration for the diagonal

            voter_count = 0  # Count voters who ranked both items
            majority_count = 0  # Count voters who prefer item j over item i

            # Iterate over each voter
            for voter_idx in range(num_voters):
                if not np.isnan(input_list[voter_idx][i]) and not np.isnan(input_list[voter_idx][j]):
                    voter_count += 1  # Count the voter
                    if input_list[voter_idx][j] < input_list[voter_idx][i]:
                        majority_count += 1  # Count the preference for item j

            # Update the transition matrix based on majority preference
            if voter_count > 0 and majority_count > voter_count / 2:
                pmatrix[i][j] += 1  # Item j is preferred over item i
            else:
                pmatrix[i][i] += 1  # Item i remains preferred

    # Normalize the transition matrix so that each row sums to 1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix  # Return the transition matrix


def mc(input_list, mc_type, max_iteration=50):
    """
    Perform Markov Chain ranking based on the specified method.

    Parameters
    ----------
    input_list : ndarray
        A 2D numpy array where each row represents a voter and each column represents an item.
    mc_type : MethodType
        The type of Markov Chain method to use (e.g., MC1, MC2, MC3, MC4).
    max_iteration : int, optional
        The maximum number of iterations for the power method (default is 50).

    Returns
    -------
    result : ndarray
        A 1D numpy array representing the rank of each item.
    """
    transfer_matrix = None
    # Obtain the transfer matrix based on the specified MC type
    if mc_type == MethodType.MC1:
        transfer_matrix = get_mc1_transfer_matrix(input_list)
    elif mc_type == MethodType.MC2:
        transfer_matrix = get_mc2_transfer_matrix(input_list)
    elif mc_type == MethodType.MC3:
        transfer_matrix = get_mc3_transfer_matrix(input_list)
    elif mc_type == MethodType.MC4:
        transfer_matrix = get_mc4_transfer_matrix(input_list)

    # Initialize the array for power iteration
    num_items = input_list.shape[1]
    init_array = np.full(num_items, 1.0 / num_items)  # Uniform distribution
    restmp = init_array

    # Power iteration to find the stationary distribution
    for i in range(max_iteration):
        res = np.dot(restmp, transfer_matrix)  # Multiply current distribution by the transfer matrix
        restmp = res  # Update the distribution

    # Extract the final distribution and sort to obtain rankings
    first_row = restmp
    sorted_indices = np.argsort(first_row)[::-1]  # Get indices of sorted items in descending order

    currrent_rank = 1
    result = np.zeros(input_list.shape[1])  # Initialize result array for rankings
    for index in sorted_indices:
        result[index] = currrent_rank  # Assign ranks based on sorted indices
        currrent_rank += 1

    return result  # Return the array containing ranking information


def markovchainmethod(input_file_path, output_file_path, mc_type=MethodType.MC1, max_iteration=50):
    """
    Load input data, process it using the Markov Chain method, and save the results.

    Parameters
    ----------
    input_file_path : str
        The path to the input CSV file containing the data.
    output_file_path : str
        The path to the output CSV file where results will be saved.
    mc_type : MethodType, optional
        The type of Markov Chain method to use (default is MC1).
    max_iteration : int, optional
        The maximum number of iterations for the power method (default is 50).
    """
    df, unique_queries = csv_load(input_file_path)  # Load data from the CSV file
    result = []  # Initialize an empty list to store results

    for query in unique_queries:
        # Filter the DataFrame for the current query
        query_data = df[df['Query'] == query]

        # Map the item codes and prepare input lists for the MC function
        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(query_data)

        # Call the MC function to get ranking information
        rank = mc(input_lists, mc_type, max_iteration)

        # Append the results to the output list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]  # Get item code from mapping
            new_row = [query, item_code, item_rank]  # Create a new row for the result
            result.append(new_row)

    save_as_csv(output_file_path, result)  # Save the results to a CSV file
