"""
Mork-H Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Azzini, I., & Munda, G. (2020). A new approach for identifying the Kemeny median ranking. European Journal of Operational Research, 281(2), 388-401.

Authors:
    fsw, tancilon
Date:
    2024-9-18
"""
import numpy as np

from src.rapython.datatools import *

__all__ = ['mork_heuristic']


def outranking_matrix(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]

    # When the number of items is greater than or equal to 3, calculate the preferences between items and construct an outranking matrix
    outrankingmatrix = np.zeros((num_items, num_items))
    if num_items >= 3:
        for v in range(num_voters):
            for i in range(num_items):
                for j in range(num_items):
                    if i == j:
                        outrankingmatrix[i, j] = 0
                    else:
                        if input_list[v, i] < input_list[v, j]:
                            outrankingmatrix[i, j] += 1
                        elif input_list[v, i] == input_list[v, j]:
                            outrankingmatrix[i, j] += 0.5
                        else:
                            outrankingmatrix[i, j] += 0

        return outrankingmatrix


def calculate_max_row_score_index(matrix):
    row_sums = np.sum(matrix, axis=1)

    max_score_index = np.argmax(row_sums)

    return max_score_index


def calculate_max_row_score(matrix):
    row_sums = np.sum(matrix, axis=1)
    max_score_index = np.argmax(row_sums)
    max_score = row_sums[max_score_index]

    return max_score


def calculate_rank(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    ranked_list = []
    outrankingmatrix = outranking_matrix(input_list)
    outrankingmatrix_ = outrankingmatrix

    while outrankingmatrix.shape[0] > 0 and outrankingmatrix.shape[1] > 0:
        max_score_index_ = calculate_max_row_score_index(outrankingmatrix_)
        num_equal_score = max_score_index_.size
        if num_equal_score > 1:
            for v in range(num_voters):
                for i in range(num_equal_score):
                    new_input_list = []
                    new_input_list.append(input_list[v, max_score_index_[i]])

            new_matrix = outranking_matrix(new_input_list)
            max_score_index = calculate_max_row_score_index(new_matrix)

            num_equal_score = max_score_index.size

            if num_equal_score > 1:
                ranked_list.append(max_score_index[0])

                outrankingmatrix_[max_score_index[0], :] = 0
                outrankingmatrix_[:, max_score_index[0]] = 0

                max_score_index = calculate_max_row_score_index(outrankingmatrix)
                outrankingmatrix = np.delete(
                    outrankingmatrix, max_score_index, axis=0)
                outrankingmatrix = np.delete(
                    outrankingmatrix, max_score_index, axis=1)
            else:
                ranked_list.append(max_score_index)
                outrankingmatrix_[max_score_index, :] = 0
                outrankingmatrix_[:, max_score_index] = 0

                max_score_index = calculate_max_row_score_index(outrankingmatrix)
                outrankingmatrix = np.delete(
                    outrankingmatrix, max_score_index, axis=0)
                outrankingmatrix = np.delete(
                    outrankingmatrix, max_score_index, axis=1)

        else:
            if outrankingmatrix.shape[0] == 2 and outrankingmatrix.shape[1] == 2:
                row_sums = np.sum(outrankingmatrix_, axis=1)
                indices = np.where(row_sums > 0)[0]
                if indices.size >= 2:
                    if row_sums[indices[0]] > row_sums[indices[1]]:
                        ranked_list.append(indices[0])
                        ranked_list.append(indices[1])
                    elif row_sums[indices[0]] < row_sums[indices[1]]:
                        ranked_list.append(indices[1])
                        ranked_list.append(indices[0])
                    else:
                        ranked_list.append(indices[0])
                        ranked_list.append(indices[1])
                    print("1:", indices)
                elif indices.size == 1:
                    ranked_list.append(indices[0])
                    for i in range(num_items):
                        if i in ranked_list:
                            pass
                        else:
                            ranked_list.append(i)
                    print("2:", indices)
                    print("2:", indices[0])
                else:
                    for i in range(num_items):
                        if i in ranked_list:
                            pass
                        else:
                            ranked_list.append(i)
                    print("3:", indices)
                break

            else:
                print(max_score_index_)
                ranked_list.append(max_score_index_)

                max_score_index = calculate_max_row_score_index(outrankingmatrix)

                outrankingmatrix_[max_score_index_, :] = 0
                outrankingmatrix_[:, max_score_index_] = 0
                outrankingmatrix = np.delete(
                    outrankingmatrix, max_score_index, axis=0)
                outrankingmatrix = np.delete(
                    outrankingmatrix, max_score_index, axis=1)

    return ranked_list


def mork_heuristicagg(input_list):
    ranked_list = calculate_rank(input_list)
    return ranked_list


def mork_heuristic(input_file_path, output_file_path):
    df, unique_queries = csv_load(input_file_path, InputType.RANK)
    result = []

    for query in unique_queries:
        query_data = df[df['Query'] == query]
        int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists = wtf_map(query_data)

        full_input_lists, list_numofitems = partial_to_full(input_lists)

        item_ranked = mork_heuristicagg(full_input_lists)

        for i in range(len(item_ranked)):
            item_code = int_to_item_map[item_ranked[i]]
            item_rank = i + 1
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    save_as_csv(output_file_path, result)
