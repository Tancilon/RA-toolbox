"""
UTF-8 
rapython: 3.11.4

参考文献: Supervised Rank Aggregation for Predicting Influencers in Twitter(2011)
Tancilon: 20231226


训练集数据输入格式：
文件1: train_rel_data: 
                        1)csv文件格式 
                        2)4列 Query | 0 | Item | Relevance
文件2: train_base_data: 
                        1) csv文件格式 
                        2)4列 Query | Voter name | Item Code | Item Rank

- Query 不要求是从1开始的连续整数
- Voter name 和 Item Code允许是字符串格式


定义算法的最终输出为csv文件格式: 3列 Query | Item Code | Item Rank
    - 注意输出的为排名信息，不是分数信息

测试集数据输入格式：

文件1: test_data: 
                1) csv文件格式 
                2)4列 Query | Voter name | Item Code | Item Rank
                - Query 不要求是从1开始的连续整数
                - Voter name 和 Item Code允许是字符串格式

其他细节：
        1) 数据输入接受full lists,对Partial list的处理采用排在最后一名的方式
        2) Item Rank数值越小, 排名越靠前
        3) 训练集和测试集的Voter相同
"""

import csv

import numpy as np
from tqdm import tqdm

from evaluation import Evaluation


class WeightedBorda:
    """
    hyper-parameters:
        topk: 精度计算时topk的设置, 若不设置, 则默认为一个query下相关性文档的数量
    """

    def __init__(self, topk=None, is_partial_list=True):
        self.weights = None
        self.average_weight = None
        self.voter_name_mapping = None
        self.voter_name_reverse_mapping = None
        self.voter_num = None
        self.query_mapping = None
        self.topk = topk
        self.is_partial_list = is_partial_list

    @staticmethod
    def partialtofull(rank_base_data_matrix):
        # 扩充为full list的方式是将未排序的项目全部并列放在最后一名
        num_voters = rank_base_data_matrix.shape[0]

        for k in range(num_voters):
            if np.isnan(rank_base_data_matrix[k]).all():
                # 处理全为 NaN 的切片
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], nan=rank_base_data_matrix.shape[1])
            else:
                max_rank = np.nanmax(rank_base_data_matrix[k])
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], nan=max_rank + 1)

        return rank_base_data_matrix

    """
    return:
        score_base_data_matrix: voter * item 存储Borda分数
        rel_data_matrix: 1 * item 存储item的相关性
    """

    def convert_to_matrix(self, base_data, rel_data=None):
        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        rank_base_data_matrix = np.full((self.voter_num, item_num), np.nan)
        score_base_data_matrix = np.empty((self.voter_num, item_num))
        rel_data_matrix = np.empty(item_num)

        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]
            rank_base_data_matrix[voter_index, item_index] = item_rank

        if self.is_partial_list:
            rank_base_data_matrix = self.partialtofull(rank_base_data_matrix)

        for k in range(self.voter_num):
            for i in range(item_num):
                score_base_data_matrix[k, i] = item_num - rank_base_data_matrix[k, i]

        if rel_data is None:
            return score_base_data_matrix, item_mapping
        else:
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']

                item_index = item_mapping[item_code]
                rel_data_matrix[item_index] = item_relevance

            return score_base_data_matrix, rel_data_matrix, item_mapping

    def train(self, train_base_data, train_rel_data):
        """
        Data process
        """
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        # 建立映射
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}

        self.weights = np.zeros((len(unique_queries), self.voter_num))

        """
        Consider each query
        """
        for query in tqdm(unique_queries):
            """
            筛出当前query的数据
            """
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]

            """
            转为二维Numpy矩阵
            """
            base_data_matrix, rel_data_matrix, _ = self.convert_to_matrix(base_data, rel_data)
            """
            计算每一个voter的性能
            """
            evaluation = Evaluation()
            for voter_idx in range(self.voter_num):
                if self.topk is None:
                    topk = np.sum(rel_data_matrix > 0)
                else:
                    topk = self.topk
                voter_w = evaluation.compute_p_s(base_data_matrix[voter_idx, :], rel_data_matrix, topk) * self.voter_num
                # voter_w = evaluation.compute_AP_s(base_data_matrix[voter_idx, :], rel_data_matrix, topk)

                query_idx = self.query_mapping[query]
                self.weights[query_idx, voter_idx] = voter_w

        """
        下面算出针对所有Query的平均权重
        """
        self.average_weight = np.mean(self.weights, axis=0)

    """
    using_average:
        1.选择使用的权重参数是否是平均权重
        2.当using_average = false 时, 用于测试集的query和训练集的query相同的情况
    """

    def test(self, test_data, test_output_loc, using_average_w=True):
        test_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        unique_test_queries = test_data['Query'].unique()
        # 创建一个空的DataFrame来存储结果

        with open(test_output_loc, mode='w', newline='') as file:
            writer = csv.writer(file)

            for query in tqdm(unique_test_queries):
                query_data = test_data[test_data['Query'] == query]
                query_data_matrix, item_code_mapping = self.convert_to_matrix(query_data)
                item_code_reverse_mapping = {v: k for k, v in item_code_mapping.items()}

                if using_average_w:
                    score_list = np.dot(self.average_weight, query_data_matrix)
                else:
                    if query not in self.query_mapping:
                        score_list = np.dot(self.average_weight, query_data_matrix)
                    else:
                        query_id = self.query_mapping[query]
                        score_list = np.dot(self.weights[query_id, :], query_data_matrix)

                rank_list = np.argsort(score_list)[::-1]
                for rank_index, item_id in enumerate(rank_list):
                    item_code = item_code_reverse_mapping[item_id]
                    new_row = [query, item_code, (rank_index + 1)]
                    writer.writerow(new_row)
