"""
UTF-8
Python Version: 3.11.4

Reference:
-----------
- Hu, C., Zhang, H., Liang, C., & Huang, H. (2024, March). QI-IRA: Quantum-Inspired Interactive Ranking Aggregation for Person Re-identification. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 3, pp. 2202-2210).

Author:
-------
- Qi Deng

Date:
-----
- 2024-10-18
"""
import numpy as np

from src.rapython.datatools import *


def qi_ira_agg(sim, rel_label, k_set, iteration, error_rate=0.02):
    """
    Perform Quantum-Inspired Interactive Ranking Aggregation (QI-IRA) to iteratively
    update the ranking of items based on feedback from the relevance labels. The algorithm
    combines feedback from relevant and non-relevant items to adjust the weights of
    different rankers, enhancing the ranking process.

    Parameters:
    -----------
    sim : np.ndarray
        A 3D array of shape (rankernum, querynum, gallerynum) representing the similarity
        scores between items and queries from different rankers.

    rel_label : np.ndarray
        A 2D array of shape (querynum, gallerynum) indicating the relevance labels of items
        for each query, where a higher value signifies a more relevant item.

    k_set : int
        The number of top items to consider for feedback and weight adjustment during each iteration.

    iteration : int
        The number of iterations to run the interactive ranking aggregation process.

    error_rate : float, optional
        The interaction error rate that simulates uncertainty in feedback. Default is 0.02.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (querynum, gallerynum) representing the final re-ranked item indices
        for each query after completing the iterations.
    """
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    gallerynum = sim.shape[2]

    # Initialize feedback and weights
    feedtrue = np.zeros((querynum, gallerynum))  # Feedback for true relevance
    feeded = np.zeros((querynum, gallerynum))  # Track fed items
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
                if feeded[q, total_ranklist[q, now_num]] == 0:
                    sed += 1
                    rt.append(total_ranklist[q, now_num])
                    feeded[q, total_ranklist[q, now_num]] = 1
                now_num += 1

            # Get the relevance labels for the retrieved items
            rt_label = rel_label[q, rt]
            feedback_p = np.where(rt_label >= 1)[0]  # Indices of relevant items

            # Assign feedback based on relevance and error rate
            for j in range(k_set):
                if j in feedback_p:
                    if np.random.rand() > error_rate:
                        feedtrue[q, rt[j]] = 10  # Positive feedback for relevant items
                else:
                    if np.random.rand() > error_rate:
                        feedtrue[q, rt[j]] = -10  # Negative feedback for non-relevant items

            feedback_p = np.where(feedtrue[q, :] == 10)[0]  # Positive feedback indices
            feedback_n = np.where(feedtrue[q, :] == -10)[0]  # Negative feedback indices

            # Update weights based on feedback
            if feedback_p.size > 0:
                score_p = sim[:, q, feedback_p]  # Scores of positively fed items
                score_n = sim[:, q, feedback_n]  # Scores of negatively fed items
                score_p = np.reshape(score_p, (rankernum, feedback_p.size))
                score_n = np.reshape(score_n, (rankernum, feedback_n.size))

                # Compute positive and negative average scores
                s_p = np.sum(score_p, axis=1) / feedback_p.size
                s_n = np.sum(score_n, axis=1) / feedback_n.size if feedback_n.size > 0 else np.zeros(rankernum)

                # Update weights based on the difference of positive and negative scores
                s = s_p - s_n if feedback_n.size > 0 else s_p
                new_weight[q, :] += s

        # Update the weights using exponential moving average
        weight = weight * 0.1 + new_weight * 0.9
        for j in range(querynum):
            weight[j, :] /= np.max(weight[j, :])  # Normalize weights

        # Calculate the new similarity scores based on updated weights
        new_sim = np.zeros((querynum, gallerynum))
        for j in range(querynum):
            for k in range(rankernum):
                new_sim[j, :] += sim[k, j, :] * weight[j, k]

        new_sim += feedtrue  # Incorporate feedback into similarity scores

        # Update the total rank list based on new similarity scores
        total_ranklist = np.argsort(-new_sim, axis=1)
        total_rank = np.argsort(total_ranklist, axis=1)  # Final ranking

    return total_rank  # Return the final ranked indices


def qi_ira(input_file_path, output_file_path, input_rel_path, input_type=InputType.SCORE):
    """
    Execute Quantum-Inspired Interactive Ranking Aggregation (QI-IRA) using specified
    input and output file paths. The function loads data from CSV files, processes
    the data into a numerical format, and applies the QI-IRA aggregation method to
    produce the final ranked results.

    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file containing query, voter name, item code, and item score.

    output_file_path : str
        Path to the output CSV file where the ranked results will be saved.

    input_rel_path : str
        Path to the input CSV file containing query, item code, and relevance.

    input_type : InputType, optional
        The type of input data to determine how the item score is interpreted (RANK or SCORE).
        Default is InputType.SCORE.

    Returns:
    --------
    None
        The function saves the ranked results directly to the specified output file path.
    """
    # Load input data and relevance data from CSV files
    input_data, input_rel_data, unique_queries = csv_load(input_file_path, input_rel_path, input_type)

    # Convert the loaded data into NumPy arrays for processing
    numpy_data, numpy_rel_data, queries_mapping_dict = df_to_numpy(input_data, input_rel_data, input_type)

    # Execute QI-IRA aggregation to get the total rank
    total_rank = qi_ira_agg(numpy_data, numpy_rel_data, 3, 1)

    # Save the ranked results to a CSV file
    save_as_csv(output_file_path, total_rank, queries_mapping_dict)
