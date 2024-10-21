"""
UTF-8
Python Version: 3.11.4

Reference:
-----------
- Huang, J., Liang, C., Zhang, Y., Wang, Z., & Zhang, C. (2022). Ranking Aggregation with Interactive Feedback for Collaborative Person Re-identification. In BMVC (p. 386).

Author:
-------
- Qi Deng

Date:
-----
- 2024-10-18
"""
from enum import Enum, auto

import numpy as np

from src.rapython.datatools import *


class MethodType(Enum):
    """
    MethodType enum class representing different IRA  methods.

    Attributes:
    - RANK: Uses the rank-based IRA method.
    - SCORE: Uses the score-based IRA method.
    """
    RANK = auto()
    SCORE = auto()


def rank_based_ira(sim, rel_label, k_set, iteration, error_rate=0.02):
    """
    Perform rank-based Iterative Rank Aggregation (IRA) for a given similarity matrix
    and relevance labels.

    Parameters:
    -----------
    sim : np.ndarray
        A 3D array of shape (rankernum, querynum, gallerynum) representing the similarity scores
        between items and queries from different rankers.

    rel_label : np.ndarray
        A 2D array of shape (querynum, gallerynum) indicating the relevance labels of items
        for each query, where a higher value signifies a more relevant item.

    k_set : int
        The number of top items to consider for feedback and updating the ranks.

    iteration : int
        The number of iterations to run the re-ranking process.

    error_rate : float, optional
        The interaction error rate that simulates the uncertainty in feedback. Default is 0.02.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (querynum, gallerynum) representing the final ranked item indices
        for each query after the iterations.
    """
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    gallerynum = sim.shape[2]

    # Get the initial rank list based on the similarity scores (higher is better)
    ranklist = np.argsort(-sim, axis=2)

    # Initialize feedback matrices
    feedtrue_g = np.zeros((querynum, gallerynum))  # Feedback for true relevance
    feeded_g = np.zeros((querynum, gallerynum))  # Items that have been fed back

    # Initialize weights for each query and ranker
    weight = np.ones((querynum, rankernum))

    total_rank = None  # To store the final ranking result
    origin_sim = np.zeros((querynum, gallerynum))  # To hold the origin similarity scores

    # Calculate the initial combined similarity based on weights
    for i in range(querynum):
        for j in range(rankernum):
            origin_sim[i, :] += sim[j, i, :] * weight[i, j]

    # Get the initial rank list based on combined similarity
    origin_ranklist = np.argsort(-origin_sim, axis=1)
    total_ranklist = origin_ranklist

    for i in range(iteration):
        new_weight = np.zeros((querynum, rankernum))  # Reset weights for the new iteration

        # Iterate over each query
        for q in range(querynum):
            sed = 0  # Sed represents the number of items fed back
            now_num = 1  # Index for the rank list
            rt = []  # Retrieved top items for the current query

            # Collect top-k items that have not been fed back yet
            while sed < k_set:
                if feeded_g[q, total_ranklist[q, now_num - 1]] == 0:
                    sed += 1
                    rt.append(total_ranklist[q, now_num - 1])
                    feeded_g[q, total_ranklist[q, now_num - 1]] = 1
                now_num += 1

            # Get the relevance labels for the retrieved items
            rt_label = rel_label[q, rt]
            scored_g = np.where(rt_label >= 1)[0]  # Indices of scored items

            # Assign feedback based on relevance and error rate
            for j in range(k_set):
                if j in scored_g:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = 10  # Positive feedback
                    else:
                        feedtrue_g[q, rt[j]] = -10  # Negative feedback
                else:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = -10  # Negative feedback
                    else:
                        feedtrue_g[q, rt[j]] = 10  # Positive feedback

            # Update the new weights based on the scored items
            scored_g = np.where(feedtrue_g[q, :] == 10)[0]
            for j in range(rankernum):
                ranker_rt = ranklist[j, q, :]  # Rank list for the current ranker
                for k in scored_g:
                    x = np.where(ranker_rt == k)[0][0]  # Find position of scored item
                    score = np.ceil(x / k_set)
                    if score == 0:
                        continue
                    new_weight[q, j] += 1 / score  # Update weight based on rank

        # Update the weights using exponential moving average
        weight = weight * 0.1 + new_weight * 0.9

        # Normalize the weights
        for j in range(querynum):
            weight[j, :] /= np.max(weight[j, :])

        # Calculate the new similarity scores based on updated weights
        new_sim = np.zeros((querynum, gallerynum))
        for j in range(querynum):
            for k in range(rankernum):
                new_sim[j, :] += sim[k, j, :] * weight[j, k]

        new_sim += feedtrue_g  # Incorporate feedback into similarity scores

        # Update the total rank list based on new similarity scores
        total_ranklist = np.argsort(-new_sim, axis=1)
        total_rank = np.argsort(total_ranklist, axis=1)  # Final ranking

    return total_rank  # Return the final ranked indices


def score_based_ira(sim, rel_label, k_set, iteration, error_rate=0.02):
    """
    Perform score-based Iterative Rank Aggregation (IRA) for a given similarity matrix
    and relevance labels.

    Parameters:
    -----------
    sim : np.ndarray
        A 3D array of shape (rankernum, querynum, gallerynum) representing the similarity scores
        between items and queries from different rankers.

    rel_label : np.ndarray
        A 2D array of shape (querynum, gallerynum) indicating the relevance labels of items
        for each query, where a higher value signifies a more relevant item.

    k_set : int
        The number of top items to consider for feedback and updating the ranks.

    iteration : int
        The number of iterations to run the re-ranking process.

    error_rate : float, optional
        The interaction error rate that simulates uncertainty in feedback. Default is 0.02.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (querynum, gallerynum) representing the final ranked item indices
        for each query after the iterations.
    """
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    gallerynum = sim.shape[2]

    # Initialize feedback matrices
    feedtrue_g = np.zeros((querynum, gallerynum))  # Feedback for true relevance
    feeded_g = np.zeros((querynum, gallerynum))  # Items that have been fed back
    weight = np.ones((querynum, rankernum))  # Initialize weights for rankers
    total_rank = None  # To store the final ranking result

    # Calculate the initial combined similarity based on weights
    origin_sim = np.zeros((querynum, gallerynum))
    for i in range(querynum):
        for j in range(rankernum):
            origin_sim[i, :] += sim[j, i, :] * weight[i, j]

    # Get the initial rank list based on combined similarity
    origin_ranklist = np.argsort(-origin_sim, axis=1)
    total_ranklist = origin_ranklist

    for i in range(iteration):
        new_weight = np.zeros((querynum, rankernum))  # Reset weights for the new iteration

        # Iterate over each query
        for q in range(querynum):
            sed = 0  # Sed represents the number of items fed back
            now_num = 0  # Current index in the total rank list
            rt = []  # Retrieved top items for the current query

            # Collect top-k items that have not been fed back yet
            while sed < k_set:
                if feeded_g[q, total_ranklist[q, now_num]] == 0:
                    sed += 1
                    rt.append(total_ranklist[q, now_num])
                    feeded_g[q, total_ranklist[q, now_num]] = 1
                now_num += 1

            # Get the relevance labels for the retrieved items
            rt_label = rel_label[q, rt]
            scored_g = np.where(rt_label >= 1)[0]  # Indices of scored items

            # Assign feedback based on relevance and error rate
            for j in range(k_set):
                if j in scored_g:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = 10  # Positive feedback
                    else:
                        feedtrue_g[q, rt[j]] = -10  # Negative feedback
                else:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = -10  # Negative feedback
                    else:
                        feedtrue_g[q, rt[j]] = 10  # Positive feedback

            # Update the new weights based on the scored items
            scored_g = np.where(feedtrue_g[q, :] == 10)[0]
            if scored_g.size > 1:
                # Calculate the standard deviation of scores for the scored items
                anno_g = sim[:, q, scored_g].reshape(rankernum, scored_g.size)
                std_w = np.std(anno_g, axis=1)  # Standard deviation across rankers
                max_std = np.max(std_w)  # Find the maximum standard deviation
                std_w = std_w / max_std  # Normalize standard deviation
                new_weight[q, :] += 1.0 / std_w  # Update weights inversely with std
                total_weight = np.max(new_weight[q, :])  # Normalize new weights
                new_weight[q, :] /= total_weight

        # Update the weights using exponential moving average
        weight = weight * 0.1 + new_weight * 0.9
        weight /= np.max(weight, axis=1, keepdims=True)  # Normalize weights

        # Calculate the new similarity scores based on updated weights
        new_sim = np.zeros((querynum, gallerynum))
        for j in range(querynum):
            for k in range(rankernum):
                new_sim[j, :] += sim[k, j, :] * weight[j, k]

        new_sim += feedtrue_g  # Incorporate feedback into similarity scores

        # Update the total rank list based on new similarity scores
        total_ranklist = np.argsort(-new_sim, axis=1)
        total_rank = np.argsort(total_ranklist, axis=1)  # Final ranking

    return total_rank  # Return the final ranked indices


def ira(input_file_path, output_file_path, input_rel_path, mode=MethodType.RANK, input_type=InputType.SCORE):
    """
    Execute the Iterative Rank Aggregation (IRA) using specified input and output file paths.
    The function loads data from CSV files, processes the data into a numerical format, and
    applies the appropriate IRA method based on the specified mode.

    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file containing query, voter name, item code, and item rank/score.

    output_file_path : str
        Path to the output CSV file where the ranked results will be saved.

    input_rel_path : str
        Path to the input CSV file containing query, item code, and relevance.

    mode : MethodType, optional
        The mode of operation to determine which IRA method to use (RANK or SCORE). Default is MethodType.RANK.

    input_type : InputType, optional
        The type of input data to determine how the item rank/score is interpreted (RANK or SCORE). Default is InputType.SCORE.

    Returns:
    --------
    None
        The function saves the ranked results directly to the specified output file path.
    """
    # Load input data and relevance data from CSV files
    input_data, input_rel_data, unique_queries = csv_load(input_file_path, input_rel_path, input_type)

    # Convert the loaded data into NumPy arrays for processing
    numpy_data, numpy_rel_data, queries_mapping_dict = df_to_numpy(input_data, input_rel_data, input_type)

    total_rank = None  # Initialize the total rank variable
    if mode == MethodType.RANK:
        # Execute rank-based IRA
        total_rank = rank_based_ira(numpy_data, numpy_rel_data, 3, 1)
    elif mode == MethodType.SCORE:
        # Execute score-based IRA
        total_rank = score_based_ira(numpy_data, numpy_rel_data, 3, 1)

    # Save the ranked results to a CSV file
    save_as_csv(output_file_path, total_rank, queries_mapping_dict)
