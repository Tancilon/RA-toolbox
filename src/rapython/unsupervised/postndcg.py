"""
PostNDCG Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised ensemble of ranking models for news comments using pseudo answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.

Authors:
    fsw, tancilon
Date:
    2024-9-18
"""
import numpy as np

from src.rapython.datatools import *


def postndcg_agg(sim):
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    item_num = sim.shape[2]

    ranklist = np.argsort(-sim, axis=2)
    rank = np.argsort(ranklist, axis=2)

    result = np.zeros((querynum, item_num))
    ndcglist = np.zeros((rankernum, rankernum))

    for i in range(querynum):
        for j in range(rankernum - 1):
            for k in range(j + 1, rankernum):
                ndcg = 0
                ranklist1 = ranklist[j, i, :]
                ranklist2 = ranklist[k, i, :]

                for m in range(item_num):
                    if ranklist1[m] == ranklist2[m]:
                        ndcg += np.log(2) / np.log(m + 2)

                ndcglist[j, k] = ndcg
                ndcglist[k, j] = ndcg

        ndcgrank = np.sum(ndcglist, axis=1)
        ndcgrank = np.argsort(-ndcgrank)
        result[i, :] = rank[ndcgrank[0], i, :]

    return result


def postndcg(input_file_path, output_file_path, input_type=InputType.SCORE):
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
    df, unique_queries = csv_load(input_file_path)
    numpy_data, queries_mapping_dict = df_to_numpy(df, input_type)
    save_as_csv(output_file_path, postndcg_agg(numpy_data), queries_mapping_dict)
