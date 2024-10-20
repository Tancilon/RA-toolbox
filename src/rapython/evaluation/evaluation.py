"""
Evaluation Metrics Implementation

This module contains the implementation of various evaluation metrics used for performance assessment.

Authors:
    Tancilon
Date:
    2023-12-19
"""

import math
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from multipledispatch import dispatch
from src.rapython.datatools import *


class Evaluation:
    """
    Evaluation Class

    This class implements several evaluation metrics for ranking algorithms, including
    precision, rank, average precision (AP), and normalized discounted cumulative gain (nDCG).
    It also provides methods to evaluate algorithm outputs against ground truth relevance data.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_precision(score_list, rel_list, topk, input_type):
        """
        Compute Precision @ k

        This method calculates the precision at a given cut-off rank k. Precision is
        defined as the proportion of relevant documents among the retrieved documents.

        Parameters:
        -----------
        score_list : np.ndarray
            1D array of scores or ranks for the items.

        rel_list : np.ndarray
            1D array of relevance scores for the items.

        topk : int
            The cut-off rank for computing precision.

        input_type : InputType
            The type of input data, either 'SCORE' or 'RANK'.

        Returns:
        --------
        float
            The precision value at top-k.
        """
        if topk <= 0 or topk > len(rel_list):
            topk = len(rel_list)
        rank_list = None
        if input_type == InputType.SCORE:
            # 将分数转化为排名，排名从0开始
            rank_list = np.argsort(score_list)[::-1]
        elif input_type == InputType.RANK:
            rank_list = np.argsort(score_list)

        r = 0

        for k in range(topk):
            item_idx = rank_list[k]
            if rel_list[item_idx] > 0:
                r += 1

        return r / topk

    @staticmethod
    def compute_rank(list_data, rel_list, topk, input_type):
        """
        Compute Rank @ k

        This method determines whether any relevant document is retrieved within the top-k
        results.

        Parameters:
        -----------
        list_data : np.ndarray
            1D array of scores or ranks for the items.

        rel_list : np.ndarray
            1D array of relevance scores for the items.

        topk : int
            The cut-off rank for checking relevant items.

        input_type : InputType
            The type of input data, either 'SCORE' or 'RANK'.

        Returns:
        --------
        int
            Returns 1 if a relevant document is retrieved within the top-k results, else 0.
        """
        if topk <= 0 or topk > len(rel_list):
            topk = len(rel_list)
        rank_list = None
        if input_type == InputType.RANK:
            rank_list = np.argsort(list_data)
        elif input_type == InputType.SCORE:
            # 将分数转化为排名，排名从0开始
            rank_list = np.argsort(list_data)[::-1]

        for k in range(topk):
            item_idx = rank_list[k]
            if rel_list[item_idx] > 0:
                return 1
        return 0

    def compute_average_precision(self, score_list, rel_list, topk, input_type):
        """
        Compute Average Precision (AP) @ k

        This method computes the average precision at a given cut-off rank k.
        Average precision is the average of precision values at ranks where relevant
        documents are retrieved.

        Parameters:
        -----------
        score_list : np.ndarray
            1D array of scores or ranks for the items.

        rel_list : np.ndarray
            1D array of relevance scores for the items.

        topk : int
            The cut-off rank for computing AP.

        input_type : InputType
            The type of input data, either 'SCORE' or 'RANK'.

        Returns:
        --------
        float
            The average precision value at top-k.
        """
        list_data = None
        if input_type == InputType.SCORE:
            sorted_indices = np.argsort(score_list)[::-1]
            list_data = np.argsort(sorted_indices) + 1
        elif input_type == InputType.RANK:
            list_data = score_list

        total_r = np.sum(rel_list > 0)
        ap = 0.0
        for k in range(1, topk + 1):
            item_id = np.argmax(list_data == k)
            if rel_list[item_id] > 0:
                p_k = self.compute_precision(list_data, rel_list, k, InputType.RANK)
                ap += p_k / total_r
        return ap

    @dispatch(pd.DataFrame, pd.DataFrame, int)
    def eval_mean_average_precision(self, test_data, rel_data, topk):
        """
        Evaluate Mean Average Precision (MAP) @ k

        This method computes the mean average precision across multiple queries using
        the provided test and relevance data.

        Parameters:
        -----------
        test_data : pd.DataFrame
            DataFrame containing the ranking results for each query.
            Expected columns: ['Query', 'Item Code', 'Item Rank'].

        rel_data : pd.DataFrame
            DataFrame containing the relevance scores for each query.
            Expected columns: ['Query', '0', 'Item Code', 'Relevance'].

        topk : int
            The cut-off rank for computing MAP.

        Returns:
        --------
        float
            The mean average precision across all queries.
        """
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = test_data['Query'].unique()
        sum_ap = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = covert_pd_to_csv(query_test_data, query_rel_data)
            if topk is None:
                topk = len(ra_list)
            sum_ap += self.compute_average_precision(ra_list, rel_list, topk, InputType.RANK)

        return sum_ap / len(unique_queries)

    @dispatch(str, str, str, str, InputType, int)
    def eval_mean_average_precision(self, test_path, rel_path, test_data_name, test_rel_name, data_type, topk):
        """
        evaluate mean average precision @ k
        """
        test_mat = loadmat(test_path)
        rel_mat = loadmat(rel_path)

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]

        sum_ap = 0.0
        for query in range(test_data.shape[0]):
            sum_ap += self.compute_average_precision(test_data[query, :], rel_data[query, :], topk, data_type)

        return sum_ap / test_data.shape[0]

    @dispatch(str, str, str, str, InputType, int)
    def eval_rank(self, test_path, rel_path, test_data_name, test_rel_name, data_type, topk):
        """
        evaluate rank @ k
        """
        test_mat = loadmat(test_path)
        rel_mat = loadmat(rel_path)

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]

        sum_r = 0.0
        for query in range(test_data.shape[0]):
            sum_r += self.compute_rank(test_data[query, :], rel_data[query, :], topk, data_type)

        return sum_r / test_data.shape[0]

    @dispatch(pd.DataFrame, pd.DataFrame, int)
    def eval_rank(self, test_data, rel_data, topk):
        """
        Evaluate Rank @ k

        This method evaluates whether any relevant item is retrieved within the top-k
        results for multiple queries.

        Parameters:
        -----------
        test_data : pd.DataFrame
            DataFrame containing the ranking results for each query.
            Expected columns: ['Query', 'Item Code', 'Item Rank'].

        rel_data : pd.DataFrame
            DataFrame containing the relevance scores for each query.
            Expected columns: ['Query', '0', 'Item Code', 'Relevance'].

        topk : int
            The cut-off rank for evaluating rank performance.

        Returns:
        --------
        float
            The average rank performance across all queries.
        """
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_r = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = covert_pd_to_csv(query_test_data, query_rel_data)
            sum_r += self.compute_rank(ra_list, rel_list, topk, InputType.RANK)

        return sum_r / len(unique_queries)

    @dispatch(pd.DataFrame, pd.DataFrame, int)
    def eval_precision(self, test_data, rel_data, topk):
        """
         evaluate precision @ k
         """
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_p = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = covert_pd_to_csv(query_test_data, query_rel_data)
            if topk is None:
                topk = np.sum(rel_list > 0)
            sum_p += self.compute_precision(ra_list, rel_list, topk, InputType.RANK)

        return sum_p / len(unique_queries)

    @staticmethod
    def compute_dcg(rank_list, rel_list, topk):
        """
        Compute Discounted Cumulative Gain (DCG) @ k

        This method calculates the discounted cumulative gain (DCG) at a given cut-off rank k.
        DCG measures the relevance of an item based on its position in the rank list.

        Parameters:
        -----------
        rank_list : np.ndarray
            1D array of item rankings.

        rel_list : np.ndarray
            1D array of relevance scores for the items.

        topk : int
            The cut-off rank for computing DCG.

        Returns:
        --------
        float
            The DCG value at top-k.
        """
        dcg = 0.0

        if topk > len(rel_list):
            topk = len(rel_list)

        for i in range(topk):
            item_id = rank_list[i]
            rel = rel_list[item_id]
            dcg += (2 ** rel - 1) / math.log(i + 2, 2)

        return dcg

    def compute_ndcg(self, list_data, rel_list, topk, input_type=InputType.RANK):
        """
        Compute Normalized Discounted Cumulative Gain (nDCG) @ k

        This method computes the normalized DCG at a given cut-off rank k. nDCG compares
        the ranking with an ideal ranking to provide a normalized score between 0 and 1.

        Parameters:
        -----------
        list_data : np.ndarray
            1D array of item rankings or scores.

        rel_list : np.ndarray
            1D array of relevance scores for the items.

        topk : int
            The cut-off rank for computing nDCG.

        input_type : InputType
            The type of input data, either 'SCORE' or 'RANK'.

        Returns:
        --------
        float
            The nDCG value at top-k.
        """

        # Check if inputype is equal to InputType RANK
        if input_type != InputType.RANK:
            # 如果不是 RANK 类型，发出警告
            warnings.warn(f"Warning: The input_type is {input_type}, expected {InputType.RANK}.", UserWarning)
        # array memory holds item code
        rank_list = np.argsort(list_data)
        rank_ideal_list = np.argsort(rel_list)[::-1]
        dcg = self.compute_dcg(rank_list, rel_list, topk)
        idcg = self.compute_dcg(rank_ideal_list, rel_list, topk)

        if idcg == 0:
            return 0
        else:
            return dcg / idcg

    def eval_ndcg(self, test_data, rel_data, topk=None):
        """
        Evaluate Normalized Discounted Cumulative Gain (nDCG) @ k

        This method evaluates the nDCG for multiple queries using the provided test and
        relevance data.

        Parameters:
        -----------
        test_data : pd.DataFrame
            DataFrame containing the ranking results for each query.
            Expected columns: ['Query', 'Item Code', 'Item Rank'].

        rel_data : pd.DataFrame
            DataFrame containing the relevance scores for each query.
            Expected columns: ['Query', '0', 'Item Code', 'Relevance'].

        topk : int, optional
            The cut-off rank for computing nDCG. If not provided, it defaults to the number
            of relevant items.

        Returns:
        --------
        float
            The mean nDCG across all queries.
        """
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = test_data['Query'].unique()
        sum_ndcg = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]

            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = covert_pd_to_csv(query_test_data, query_rel_data)
            if topk is None:
                topk = np.sum(rel_list > 0)
            sum_ndcg += self.compute_ndcg(ra_list, rel_list, topk, InputType.RANK)

        return sum_ndcg / len(unique_queries)
