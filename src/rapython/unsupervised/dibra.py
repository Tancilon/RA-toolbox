"""
DIBRA Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Akritidis, L., Fevgas, A., Bozanis, P., & Manolopoulos, Y. (2022). An unsupervised distance-based model for weighted rank aggregation with list pruning. Expert Systems with Applications, 202, 117435.

Authors:
    Qi Deng
Date:
    2024-10-13
"""

import numpy as np

from src.rapython.datatools import *

__all__ = ['dibra']


def dibra_agg(data, topk=10):
    # Get the size of the data matrix
    rankernum = data.shape[0]  # Equivalent to size(data, 1)
    querynum = data.shape[1]  # Equivalent to size(data, 2)
    gallerynum = data.shape[2]  # Equivalent to size(data, 3)

    converged = np.zeros((querynum, rankernum))
    new_w = np.zeros((querynum, rankernum))
    w0 = np.ones((querynum, rankernum)) / rankernum

    l_list = np.zeros((querynum, gallerynum))
    for i in range(querynum):
        for j in range(rankernum):
            l_list[i, :] += (data[j, i, :] * w0[i, j])

    origin_ranklist = np.argsort(-l_list, axis=1)
    origin_rank = np.argsort(origin_ranklist, axis=1)

    for q in range(querynum):
        now_l_rank = origin_rank[q, :]
        i = 0
        allconverged = 0
        pre_w = w0[q, :]
        now_w = np.zeros(rankernum)

        while allconverged == 0:
            i += 1
            allconverged = 1
            for r in range(rankernum):
                if converged[q, r] == 0:
                    distance = 0
                    v_ranklist = np.argsort(-data[r, q, :])  # Sort for v_ranklist
                    v_ranklist = v_ranklist.reshape(1, gallerynum)

                    for j in range(topk):
                        idx_v = v_ranklist[0, j]
                        distance += abs((j + 1) / topk - (now_l_rank[idx_v] / gallerynum))

                    distance /= (topk / 2)
                    now_w[r] = pre_w[r] + np.exp(-i * distance)
                    if (now_w[r] - pre_w[r]) > 0.001:
                        allconverged = 0
                    else:
                        converged[q, r] = 1
                else:
                    now_w[r] = pre_w[r]

            new_l = np.zeros(gallerynum)
            for r in range(rankernum):
                new_l += (data[r, q, :] * now_w[r])
                pre_w[r] = now_w[r]

            now_l = np.argsort(-new_l)
            now_l_rank = np.argsort(now_l)

        new_w[q, :] = now_w

    new_l = np.zeros((querynum, gallerynum))
    for j in range(querynum):
        for r in range(rankernum):
            new_l[j, :] += (data[r, j, :] * new_w[j, r])

    total_ranklist = np.argsort(-new_l, axis=1)
    result = np.argsort(total_ranklist, axis=1)

    return result


def dibra(input_file_path, output_file_path, input_type=InputType.SCORE):
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
    save_as_csv(output_file_path, dibra_agg(numpy_data), queries_mapping_dict)
