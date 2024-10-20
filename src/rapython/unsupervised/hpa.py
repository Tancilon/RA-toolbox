"""
HPA Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised ensemble of ranking models for news comments using pseudo answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.

Authors:
    tancilon
Date:
    2024-10-13
"""
import numpy as np

from src.rapython.datatools import *


def hpa_func(sim, topk):
    # Get the number of gallery items and the number of rankers
    gallerynum = sim.shape[0]
    rankernum = sim.shape[1]

    # Calculate the average rank for each gallery item
    averagerank = np.sum(sim, axis=1)
    averagerank = averagerank / np.max(averagerank)  # Prevent division by zero

    # Obtain the ranking list by sorting in descending order
    ranklist = np.argsort(-sim, axis=0)

    # Initialize ndcg (Normalized Discounted Cumulative Gain)
    ndcg = np.zeros(rankernum)

    # Calculate ndcg values for each ranker
    for i in range(rankernum):
        for j in range(topk):
            # Update ndcg using the average rank and logarithmic scaling
            ndcg[i] += averagerank[ranklist[j, i]] * np.log(2) / np.log(i + 2)

    # Get the ranking indices of ndcg values in descending order
    ndcgrank = np.argsort(-ndcg)

    # Initialize finalrank to hold the cumulative scores
    finalrank = np.zeros(gallerynum)

    # Calculate the final ranking based on ndcg values and similarity scores
    for i in range(rankernum):
        finalrank += ndcg[ndcgrank[i]] * sim[:, ndcgrank[i]]

    # Sort the final ranking and transform the indices back to their original order
    finalrank = np.argsort(-finalrank)
    finalrank = np.argsort(finalrank)

    return finalrank


def hpa_agg(sim):
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    item_num = sim.shape[2]

    topk = item_num
    result = np.zeros((querynum, item_num))

    for i in range(querynum):
        finalrank = hpa_func(sim[:, i, :].reshape(rankernum, item_num).T, topk)
        result[i, :] = finalrank.flatten()
    return result


def hpa(input_file_path, output_file_path, input_type=InputType.SCORE):
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
    save_as_csv(output_file_path, hpa_agg(numpy_data), queries_mapping_dict)
