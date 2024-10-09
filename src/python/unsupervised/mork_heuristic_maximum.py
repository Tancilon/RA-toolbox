import csv
import time

import numpy as np
import pandas as pd


def partialtofull(input_list):
    # 扩充为full list的方式是将未排序的项目全部并列放在最后一名
    num_voters = input_list.shape[0]
    list_numofitems = np.zeros(num_voters)

    for k in range(num_voters):
        max_rank = np.nanmax(input_list[k])
        list_numofitems[k] = max_rank
        input_list[k] = np.nan_to_num(input_list[k], nan=max_rank + 1)

    return input_list, list_numofitems


def wdf_map(query_data):
    # Get the unique Item Code and Voter Name values and create a map indexed to integers
    unique_item_codes = query_data['Item Code'].unique()
    unique_voter_names = query_data['Voter Name'].unique()

    # Establish a reverse mapping from integers to strings
    int_to_item_map = {i: code for i, code in enumerate(unique_item_codes)}
    int_to_voter_map = {i: name for i, name in enumerate(unique_voter_names)}

    # Produces a string-to-integer mapping
    item_to_int_map = {v: k for k, v in int_to_item_map.items()}
    voter_to_int_map = {v: k for k, v in int_to_voter_map.items()}

    # Create a two-dimensional Numpy array of Voter Name*Item Code, starting with a value of 0
    num_voters = len(unique_voter_names)
    num_items = len(unique_item_codes)
    input_lists = np.full((num_voters, num_items), np.nan)

    # Filling an array
    for index, row in query_data.iterrows():
        voter_name = row['Voter Name']
        item_code = row['Item Code']
        item_rank = row['Item Rank']

        voter_index = voter_to_int_map[voter_name]
        item_index = item_to_int_map[item_code]

        input_lists[voter_index, item_index] = item_rank

    return int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists


def outranking_matrix(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]

    # 当items数量大于等于3时，计算出items之间的偏好，构建outranking矩阵
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
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    unique_queries = df['Query'].unique()
    start_time = time.perf_counter()
    result = []

    for query in unique_queries:
        query_data = df[df['Query'] == query]
        int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists = wdf_map(query_data)

        full_input_lists, list_numofitems = partialtofull(input_lists)

        item_ranked = mork_heuristicagg(full_input_lists)

        for i in range(len(item_ranked)):
            item_code = int_to_item_map[item_ranked[i]]
            item_rank = i + 1
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time}秒")

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)
