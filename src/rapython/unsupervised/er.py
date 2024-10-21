"""
ER Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Mohammadi, M., & Rezaei, J. (2020). Ensemble ranking: Aggregation of rankings produced by different multi-criteria decision-making methods. Omega, 96, 102254.

Authors:
    Qi Deng
Date:
    2024-10-13
"""
import numpy as np
from scipy.stats import norm

from src.rapython.datatools import *

__all__ = ['er']


def ensemble_ranking(rank_data, tol=1e-10, max_iter=1000):
    alt_no, method_no = rank_data.shape
    i = 0
    r_star = np.zeros((alt_no, 1))

    alpha = np.zeros((1, method_no))  # half-quadratic auxiliary variable
    lambda_weights = alpha  # Lambda plays the role of weights in Ensemble Ranking

    sigma = 0.0
    while i < max_iter:
        error = rank_data - np.tile(r_star, (1, method_no))
        error_norm = np.sqrt(np.sum(error ** 2, axis=0))

        if np.sum(error_norm) == 0:
            break

        sigma = (np.linalg.norm(error_norm, 2) ** 2 / (2 * len(error_norm) ** 2))

        alpha_old = alpha.copy()
        alpha = delta(error, sigma)

        r_star_old = r_star.copy()
        lambda_weights = alpha / np.sum(alpha)
        r_star = rank_data @ lambda_weights.T
        r_star = r_star.reshape(-1, 1)
        # Convergence conditions
        if np.linalg.norm(alpha - alpha_old, 2) < tol and np.linalg.norm(r_star - r_star_old) < tol:
            break

        i += 1

    # Computing confidence index and trust level
    confmat = norm.pdf(rank_data - np.tile(r_star, (1, rank_data.shape[1])), 0, sigma) / norm.pdf(0, 0, sigma)
    consensusindex = np.sum(confmat) / (method_no * alt_no)
    trustmat = confmat @ np.diag(lambda_weights)
    trustlevel = np.sum(trustmat) / alt_no

    # Computing final rankings
    sorted_indices = np.argsort(r_star.flatten())
    finalranking = np.zeros(len(sorted_indices))
    for idx, rank in enumerate(sorted_indices):
        finalranking[rank] = idx + 1

    return r_star, finalranking, lambda_weights, consensusindex, trustlevel, sigma


def delta(v, sigma):
    """
    The minizer function delta() of the Welsch M-estimator
    """
    _, rnkrs = v.shape
    e = np.zeros(rnkrs)
    for i in range(rnkrs):
        e[i] = np.exp(-(np.linalg.norm(v[:, i])) ** 2 / (2 * sigma ** 2))
    return e


def er_agg(sim):
    """
    Aggregate rankings from multiple rankers using the rank aggregation method.

    Parameters:
    -----------
    sim : numpy.ndarray
        A 3D array of shape (rankernum, querynum, item_num) representing the similarity scores
        or rankings from multiple rankers. Each slice sim[:, i, :] contains the rankings for
        the i-th query, where rankernum is the number of rankers, querynum is the number of queries,
        and item_num is the number of items.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (querynum, item_num) containing the aggregated rankings for each query.
        Each row corresponds to a query, and each column corresponds to an item. The values represent
        the final ranks of the items after aggregation, where lower values indicate higher ranks.
    """
    rankernum = sim.shape[0]  # Number of rankers
    querynum = sim.shape[1]  # Number of queries
    item_num = sim.shape[2]  # Number of items

    # Get ranks by sorting similarity scores in descending order
    rank = np.argsort(-sim, axis=2)
    # Get final ranks by ranking the ranks from multiple rankers
    rank = np.argsort(rank, axis=2)

    res = np.zeros((querynum, item_num))  # Initialize the result array

    for i in range(querynum):
        # Ensemble the rankings for the i-th query and get the final ranking
        _, finalranking, *_ = ensemble_ranking(
            rank[:, i, :].reshape(rankernum, item_num).T)
        res[i, :] = finalranking  # Store the final ranking for the query

    return res  # Return the aggregated rankings


def er(input_file_path, output_file_path, input_type=InputType.SCORE):
    """
    Process the input CSV file to aggregate rankings and write the results to an output CSV file.
    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file.
        The input to the algorithm should be in CSV file format with the following columns:

        - Query: Does not require consecutive integers starting from 1.
        - Voter Name: Allowed to be in string format.
        - Item Code: Allowed to be in string format.
        - Item Score/Item Rank: Represents the score/rank given by each voter. It is recommended to choose the score format
    output_file_path : str
        Path to the output CSV file.
    input_type : InputType, optional
        The type of input data, defaults to InputType.RANK. It determines
        the naming of the fourth column, which will either be 'Item Rank'
        or 'Item Score' based on this value.
    """
    df, unique_queries = csv_load(input_file_path, input_type)
    numpy_data, queries_mapping_dict = df_to_numpy(df, input_type)
    save_as_csv(output_file_path, er_agg(numpy_data), queries_mapping_dict)
