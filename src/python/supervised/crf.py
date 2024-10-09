"""
UTF-8 
python: 3.11.4
Tensorflow: 2.15.0

参考文献: CRF framework for supervised preference aggregation(2013)
Tancilon: 20231224


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
        1) 数据输入接受partial list
        2) Item Rank数值越小, 排名越靠前
"""

import csv
from itertools import permutations

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from evaluation import Evaluation


class CRF:

    def __init__(self):
        self.weights = None
        self.average_weight = None
        self.voter_name_mapping = None
        self.voter_name_reverse_mapping = None
        self.voter_num = None

    """
    Return:
        r: 相关性标签, 1 * item一维Numpy数组, 数组内存放相关性
        R: item * voter二维Numpy数组, 数组内存放rank(排名), 如果voter_k 未给item_k排名, 则R[k,k] = 0
    """

    def convert_to_matrix(self, base_data, rel_data=None):
        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        r_matrix = np.zeros((item_num, self.voter_num))
        r = np.zeros(item_num)

        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]
            r_matrix[item_index, voter_index] = item_rank

        if rel_data is None:
            return r_matrix, item_mapping
        else:
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']

                item_index = item_mapping[item_code]
                r[item_index] = item_relevance
            return r, r_matrix, item_mapping

    @staticmethod
    def subsample_items(base_data, rel_data, epsilon):
        """
        对rel_data进行Item抽样
        """
        # 确保每个不同的 Relevance 值至少有一个样本
        unique_relevance_samples = rel_data.groupby('Relevance').apply(lambda x: x.sample(1))

        # debug
        # if len(unique_relevance_samples) == 1:
        #     print("Warning...")
        #     print(rel_data)

        # 如果总样本数超过 epsilon，则进行抽样
        if len(unique_relevance_samples) > epsilon:
            sampled_data = unique_relevance_samples.sample(epsilon)
        else:
            sampled_data = unique_relevance_samples

        # 确保 Item Code 互不相同并且样本数为N
        while len(sampled_data['Item Code'].unique()) < epsilon:
            additional_samples = rel_data.sample(1)
            sampled_data = pd.concat([sampled_data, additional_samples]).drop_duplicates(
                subset='Item Code').reset_index(drop=True)

        """
        根据抽样结果处理base_data
        """

        # 获取 sampled_data 中的 'Item Code' 列的唯一值
        sampled_item_codes = sampled_data['Item Code'].unique()
        # 过滤掉 base_data 中 'Item Code' 列中不在 sampled_item_codes 中的行
        filtered_base_data = base_data[base_data['Item Code'].isin(sampled_item_codes)]

        return filtered_base_data, sampled_data

    """
    y: 1 * item一维Numpy数组, 数组内存放项目排名
    r: 相关性标签, 1 * item一维Numpy数组, 数组内存放相关性
    """

    @staticmethod
    def compute_loss(y, r, loss_cut_off):
        # 如果没有手动设置loss_cut_off, 修改为所有相关性文档的数量
        if loss_cut_off is None:
            loss_cut_off = np.sum(r > 0)
        evaluation = Evaluation()
        ndcg = evaluation.compute_ndcg_r(y, r, loss_cut_off)
        return 1 - ndcg

    # 注意theta是tf中的对象
    @staticmethod
    def commpute_negative_engergy(y, r_matrix, theta):
        item_num = len(y)
        voter_num = r_matrix.shape[1]
        # i: 考虑所有的排名
        negative_energy = 0.0
        for i in range(1, item_num + 1):
            item_info = 0.0
            for k in range(voter_num):
                # item_id = np.where(y == i)[0]
                item_id = np.argmax(y == i)
                if r_matrix[item_id, k] == 0:
                    item_info += theta[3 * k]
                else:
                    for j in range(item_num):
                        if j == item_id or r_matrix[item_id, k] == 0 or r_matrix[j, k] == 0:
                            continue
                        if r_matrix[item_id, k] < r_matrix[j, k]:
                            item_info += theta[3 * k + 1] * (r_matrix[j, k] - r_matrix[item_id, k]) / np.max(
                                r_matrix[:, k])
                        if r_matrix[j, k] < r_matrix[item_id, k]:
                            item_info -= theta[3 * k + 2] * (r_matrix[item_id, k] - r_matrix[j, k]) / np.max(
                                r_matrix[:, k])
            item_info = item_info / tf.math.log(tf.cast(i + 1, tf.float64))
            negative_energy += item_info
        negative_energy /= item_num * item_num
        return negative_energy

    """
    Param:
        alpha: learning rate
        epsilon: cut-off ,该参数的取值至少要大于数据集中不同相关性标签的种类数(eg:如果相关性标签只有0和1两种取值,则epsilon至少大于2)
        loss_type_cut: ndcg@k 中k的取值, 若为None, 则赋值为所有相关性文档的数量
    """

    def train(self, train_base_data, train_rel_data, alpha=0.01, epsilon=5, epoch=300, loss_cut_off=None):
        """
        Data process
        """
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}

        """
        Initialize weights
        """
        self.weights = np.zeros(3 * self.voter_num)

        """
        Repeat CRF optimization
        """

        for epo in range(epoch):
            # debug
            # print("epo={0}".format(epo))
            for query in tqdm(unique_queries):
                """
                筛出当前query的数据
                """

                base_data = train_base_data[train_base_data['Query'] == query]
                rel_data = train_rel_data[train_rel_data['Query'] == query]
                unique_items = base_data['Item Code'].unique()
                if len(unique_items) > epsilon:
                    if len(rel_data) < epsilon:
                        continue
                    subs_base_data, subs_rel_data = self.subsample_items(base_data, rel_data, epsilon)
                    r, r_matrix, _ = self.convert_to_matrix(subs_base_data, subs_rel_data)
                else:
                    r, r_matrix, _ = self.convert_to_matrix(base_data, rel_data)

                """
                计算梯度
                """
                theta = tf.Variable(self.weights)

                # 使用 tf.GradientTape() 来记录计算过程

                with tf.GradientTape() as tape:

                    # 初始化 y
                    objective = 0.0
                    # 枚举每一种可能的排序
                    initial_perm = np.empty(len(r))
                    for i in range(len(r)):
                        initial_perm[i] = i + 1
                    all_permutations = permutations(initial_perm)
                    # y: 1 * item一维Numpy数组，数组内存放项目排名
                    sum_exp_negative_energy = 0.0
                    for perm in all_permutations:
                        y = np.array(perm)
                        loss = self.compute_loss(y, r, loss_cut_off)
                        negative_energy = self.commpute_negative_engergy(y, r_matrix, theta)
                        objective += loss * tf.exp(negative_energy)
                        sum_exp_negative_energy += tf.exp(negative_energy)
                    objective /= sum_exp_negative_energy

                # 计算函数 y 相对于向量 x 的梯度
                grad = tape.gradient(objective, theta)

                self.weights = self.weights - alpha * grad.numpy()

    def test(self, test_data, output):
        test_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        unique_test_queries = test_data['Query'].unique()

        with open(output, mode='w', newline='') as file:
            writer = csv.writer(file)
            for query in tqdm(unique_test_queries):
                query_data = test_data[test_data['Query'] == query]
                r_matrix, item_code_mapping = self.convert_to_matrix(query_data)
                item_code_reverse_mapping = {v: k for k, v in item_code_mapping.items()}
                item_num = r_matrix.shape[0]
                score_list = np.empty(item_num)

                for i in range(item_num):
                    score_i = 0.0
                    for k in range(self.voter_num):
                        if r_matrix[i, k] == 0:
                            score_i -= self.weights[3 * k]
                        else:
                            max_rank = np.max(r_matrix[:, k])
                            score_i -= self.weights[3 * k + 1] * (
                                    (max_rank - r_matrix[i, k]) * (max_rank + 1 - r_matrix[i, k]) / (max_rank * 2))
                            score_i += self.weights[3 * k + 2] * (
                                        (r_matrix[i, k] - 1) * r_matrix[i, k] / (2 * max_rank))
                    score_list[i] = score_i

                rank_list = np.argsort(score_list)
                for rank_index, item_id in enumerate(rank_list):
                    item_code = item_code_reverse_mapping[item_id]
                    new_row = [query, item_code, (rank_index + 1)]
                    writer.writerow(new_row)
