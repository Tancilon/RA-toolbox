# 参考文献Web Metasearch: Rank vs. Score Based Rank Aggregation Methods实现MC1-4方法
# Tancilon：20230926
# 定义MarKovChain的顶层输入为csv文件格式，4列 Query | Voter name | Item Code | Item Rank
#      - Query 不要求是从1开始的连续整数
#      - Voter name 和 Item Code允许是字符串格式
# 定义MarKovChain的最终输出为csv文件格式：3列 Query | Item Code | Item Rank
#      - 注意输出的为排名信息，不是分数信息
# def MC()函数中input_list 为输入排名信息，含义为二维数组Voters*Items, 数组内存放的是对Item的项目排名数值，数值越小表示排名越靠前。
# 注意：如果一个Voter[i]对某一个项目item[j]没有排序，则input_list[i][j] = NaN
# 注意：输入排序接受 partial list

import numpy as np
import pandas as pd
import csv
from enum import Enum, auto


class MethodType(Enum):
    MC1 = auto()
    MC2 = auto()
    MC3 = auto()
    MC4 = auto()


def get_mc1_transfer_matrix(input_list):
    num_items = input_list.shape[1]
    # 初始化概率转移矩阵 pmatrix
    pmatrix = np.zeros((num_items, num_items))
    # 遍历每个项目 i
    for i in range(num_items):
        # 初始化一个集合用于存储所有排名高于等于 i 的项目 j
        ranked_higher_or_equal = []

        # 遍历每个投票者的排名
        for voter_ranking in input_list:
            # 检查当前投票者的排名是否包含项目 i
            if not np.isnan(voter_ranking[i]):
                # 找到项目 i 的位置
                i_rank = int(voter_ranking[i])

                # 将排名高于等于 i 的项目添加到集合中
                for j in range(num_items):
                    if not np.isnan(voter_ranking[j]) and int(voter_ranking[j]) <= i_rank:
                        ranked_higher_or_equal.append(j)

        # 计算从状态 i 转移到状态 j 的概率，均匀分布
        total_ranked = len(ranked_higher_or_equal)
        if total_ranked > 0:
            probability = 1.0 / total_ranked
            for j in ranked_higher_or_equal:
                pmatrix[i][j] += probability

    # 归一化概率转移矩阵,使得每一行的概率之和为1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix


def get_mc2_transfer_matrix(input_list):
    num_items = input_list.shape[1]
    # 初始化概率转移矩阵 pmatrix
    pmatrix = np.zeros((num_items, num_items))

    # 遍历每个项目 i
    for i in range(num_items):
        # 找到所有包含项目 i 的投票者的索引
        voters_with_i = [voter_index for voter_index, rankings in enumerate(input_list) if not np.isnan(rankings[i])]

        # 如果没有包含项目 i 的投票者，则跳过
        if len(voters_with_i) == 0:
            continue

        voter_probability = 1.0 / len(voters_with_i)
        # 遍历每个包含项目 i 的投票者
        for voter_i in voters_with_i:
            # 找到所有排名在项目 i 或在项目 i 之前的项目 j 的索引
            ranked_higher_or_equal = [j for j, ranking in enumerate(input_list[voter_i]) if
                                      not np.isnan(ranking) and ranking <= input_list[voter_i][i]]

            # 如果没有这样的项目，则跳过
            if len(ranked_higher_or_equal) == 0:
                continue

            # 计算从状态 i 转移到状态 j 的概率，均匀分布
            probability = 1.0 / len(ranked_higher_or_equal)
            for j in ranked_higher_or_equal:
                pmatrix[i][j] += probability * voter_probability

    # 归一化概率转移矩阵,使得每一行的概率之和为1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix


def get_mc3_transfer_matrix(input_list):
    num_items = input_list.shape[1]
    # 初始化概率转移矩阵 pmatrix
    pmatrix = np.zeros((num_items, num_items))

    # 遍历每个项目 i
    for i in range(num_items):
        # 找到所有包含项目 i 的投票者的索引
        voters_with_i = [voter_index for voter_index, rankings in enumerate(input_list) if not np.isnan(rankings[i])]

        # 如果没有包含项目 i 的投票者，则跳过
        if len(voters_with_i) == 0:
            continue

        voter_probability = 1.0 / len(voters_with_i)
        # 遍历每个包含项目 i 的投票者
        for voter_i in voters_with_i:
            # 计算该投票者排序的项目数
            non_nan_count = sum(1 for element in input_list[voter_i] if not np.isnan(element))
            item_probability = 1.0 / non_nan_count
            for j in range(num_items):
                if not np.isnan(input_list[voter_i][j]) and int(input_list[voter_i][j]) < input_list[voter_i][i]:
                    pmatrix[i][j] += voter_probability * item_probability

    # 给M的对角线M[i][i]赋值
    for i in range(num_items):
        diagonal_sum = np.sum(pmatrix[i]) - pmatrix[i][i]  # 第i行其他列的数值之和
        pmatrix[i][i] = 1 - diagonal_sum

    # 归一化概率转移矩阵,使得每一行的概率之和为1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix


def get_mc4_transfer_matrix(input_list):
    num_items = input_list.shape[1]
    num_voters = len(input_list)
    pmatrix = np.zeros((num_items, num_items))

    for i in range(num_items):
        for j in range(num_items):
            if i == j:
                pmatrix[i][j] += 1
                continue  # M[i][i] 保持在i处，跳过这种情况

            voter_count = 0
            majority_count = 0

            for voter_idx in range(num_voters):
                if not np.isnan(input_list[voter_idx][i]) and not np.isnan(input_list[voter_idx][j]):
                    voter_count += 1
                    if input_list[voter_idx][j] < input_list[voter_idx][i]:
                        majority_count += 1

            if voter_count > 0 and majority_count > voter_count / 2:
                pmatrix[i][j] += 1
            else:
                pmatrix[i][i] += 1

    # 归一化概率转移矩阵,使得每一行的概率之和为1
    pmatrix /= pmatrix.sum(axis=1)[:, np.newaxis]
    return pmatrix


def mc(input_list, mc_type, max_iteration=50):
    transfer_matrix = None
    # 根据不同的MC类型获得转移矩阵
    if mc_type == MethodType.MC1:
        transfer_matrix = get_mc1_transfer_matrix(input_list)
    elif mc_type == MethodType.MC2:
        transfer_matrix = get_mc2_transfer_matrix(input_list)
    elif mc_type == MethodType.MC3:
        transfer_matrix = get_mc3_transfer_matrix(input_list)
    elif mc_type == MethodType.MC4:
        transfer_matrix = get_mc4_transfer_matrix(input_list)

    # 用幂迭代方法让转移矩阵收敛，用于之后得到Markov chain ordering
    num_items = input_list.shape[1]
    init_array = np.full(num_items, 1.0 / num_items)
    restmp = init_array

    for i in range(max_iteration):
        res = np.dot(restmp, transfer_matrix)
        restmp = res

    # 取出第一行
    first_row = restmp
    # 对第一行进行排序并返回排序后的列索引
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1
    result = np.zeros(input_list.shape[1])
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1

    # result 为1*items的二维数组，数组内存放的是排名信息
    return result


# input_file_path：输入文件路径，顶层输入csv文件格式，4列 Query | Voter name | Item Code | Item Rank. 其中Item Rank数值越小，排名越靠前
# output_file_path: 输出文件路径，内容：3列 Query | Item Code | Item Rank
def markovchainmethod(input_file_path, output_file_path, mc_type=MethodType.MC1, max_iteration=50):
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    # 获取唯一的Query值
    unique_queries = df['Query'].unique()
    # 创建一个空的DataFrame来存储结果
    result = []

    for query in unique_queries:
        # 筛选出当前Query的数据
        query_data = df[df['Query'] == query]

        # 获取唯一的Item Code和Voter Name值，并创建索引到整数的映射
        unique_item_codes = query_data['Item Code'].unique()
        unique_voter_names = query_data['Voter Name'].unique()

        # 建立整数到字符串的逆向映射
        item_code_reverse_mapping = {i: code for i, code in enumerate(unique_item_codes)}
        voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}

        # 生成字符串到整数的映射
        item_code_mapping = {v: k for k, v in item_code_reverse_mapping.items()}
        voter_name_mapping = {v: k for k, v in voter_name_reverse_mapping.items()}

        # 创建Voter Name*Item Code的二维Numpy数组，初始值为0
        num_voters = len(unique_voter_names)
        num_items = len(unique_item_codes)
        input_list = np.full((num_voters, num_items), np.nan)

        # 填充数组
        for index, row in query_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = voter_name_mapping[voter_name]
            item_index = item_code_mapping[item_code]

            input_list[voter_index, item_index] = item_rank

        # 调用MC函数，获取排名信息
        rank = mc(input_list, mc_type, max_iteration)

        # 将结果添加到result_df中
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)
