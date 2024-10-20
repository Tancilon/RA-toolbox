"""
iRANk Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Wei, F., Li, W., & Liu, S. (2010). iRANK: A rank‐learn‐combine framework for unsupervised ensemble ranking. Journal of the American Society for Information Science and Technology, 61(6), 1232-1243.

Authors:
    tancilon
Date:
    2024-10-11
"""
import numpy as np

from src.rapython.datatools import *


def irank_agg(sim):
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    itemnum = sim.shape[2]

    ranklist = np.argsort(-sim, axis=2)
    rank = np.argsort(ranklist, axis=2)

    topk = itemnum
    superviserank = np.zeros((rankernum - 1, querynum, itemnum))

    for iteration in range(3):
        newsim = sim * 0.9
        for i in range(rankernum):
            if i == 0:
                superviserank = rank[1:rankernum + 1, :, :]
            elif i == rankernum - 1:
                superviserank = rank[0:rankernum, :, :]
            else:
                superviserank[:i, :, :] = rank[:i, :, :]
                superviserank[i:rankernum, :, :] = rank[i + 1:rankernum + 1, :, :]

            dscore = 1.0 / superviserank
            # print(dscore.shape)
            dscoretotal = np.sum(dscore, axis=0)
            # print(dscoretotal.shape)
            sresultlist = np.argsort(-dscoretotal, axis=1)
            sresultlist = sresultlist.reshape(querynum, itemnum)

            for k in range(querynum):
                for ll in range(topk):
                    newsim[i, k, sresultlist[k, ll]] += 0.1

        sim = newsim
        ranklist = np.argsort(-sim, axis=2)
        rank = np.argsort(ranklist, axis=2)

    finalsim = np.sum(sim, axis=0)
    # print(finalsim.shape)
    finalsim = finalsim.reshape(querynum, itemnum)
    result = np.argsort(-finalsim, axis=1)
    result = np.argsort(result, axis=1)
    return result


def irank(input_file_path, output_file_path, input_type=InputType.SCORE):
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
    save_as_csv(output_file_path, irank_agg(numpy_data), queries_mapping_dict)
