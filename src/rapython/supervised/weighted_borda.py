"""
Encoding: UTF-8
Python Version: 3.11.4

Reference:
-----------
- Subbian, K., & Melville, P. (2011, October). Supervised rank aggregation for predicting influencers in twitter. In 2011 IEEE Third International Conference on Privacy, Security, Risk and Trust and 2011 IEEE Third International Conference on Social Computing (pp. 661-665). IEEE.

Author:
-------
- Tancilon

Date:
-----
- 2023-12-26

Input Format for Training Data:
-------------------------------
1. File 1: train_rel_data
   - Format: CSV
   - Columns: Query | 0 | Item | Relevance

2. File 2: train_base_data
   - Format: CSV
   - Columns: Query | Voter Name | Item Code | Item Rank

Notes:
------
- Query does not need to be consecutive integers starting from 1.
- Voter Name and Item Code are allowed to be in string format.

Output Format:
--------------
- The final output of the algorithm will be in CSV format with the following columns:
  - Query | Item Code | Item Rank
  - Note: The output contains ranking information, not score information.

Input Format for Testing Data:
-------------------------------
1. File 1: test_data
   - Format: CSV
   - Columns: Query | Voter Name | Item Code | Item Rank

Notes:
------
- Query does not need to be consecutive integers starting from 1.
- Voter Name and Item Code are allowed to be in string format.

Additional Details:
-------------------
1. The data input accepts full lists; partial lists will be treated as having the lowest rank.
2. Smaller Item Rank values indicate higher rankings.
3. The voters in the training and testing datasets are the same.
"""

import csv

import numpy as np
from tqdm import tqdm

from src.rapython.datatools import InputType, csv_load
from src.rapython.evaluation.evaluation import Evaluation


class WeightedBorda:
    """
    A class to implement the Weighted Borda Count algorithm for ranking items
    based on preferences from multiple voters.
    """

    def __init__(self, topk=None, is_partial_list=True):
        """
        Initializes the WeightedBorda instance with the specified parameters.

        Parameters:
        -----------
        topk : int, optional
            The number of top items to consider for accuracy calculations.
            If not provided, it defaults to the count of relevant documents
            for each query.

        is_partial_list : bool, optional
            Indicates whether the input data contains partial rankings.
            Defaults to True.
        """
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
        """
        Converts a rank base data matrix with potential missing values
        into a full list format by placing all unrated items at the
        bottom of the ranking.

        Parameters:
        -----------
        rank_base_data_matrix : numpy.ndarray
            A 2D numpy array of shape (voter_num, item_num) containing
            rankings, where missing values are represented by NaN.

        Returns:
        --------
        numpy.ndarray
            A modified rank base data matrix in full list format, where
            all unrated items are assigned the lowest rank.
        """
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

    def convert_to_matrix(self, base_data, rel_data=None):
        """
        Converts the provided base data and relevance data into matrices
        suitable for further processing. The method generates a rank base
        data matrix, a score base data matrix, and a relevance data matrix.

        Parameters:
        -----------
        base_data : pandas.DataFrame
            A DataFrame containing the base data with columns 'Query',
            'Voter Name', 'Item Code', and 'Item Rank'.

        rel_data : pandas.DataFrame, optional
            A DataFrame containing relevance data with columns 'Query',
            'Item Code', and 'Relevance'. Defaults to None.

        Returns:
        --------
        tuple
            - score_base_data_matrix : numpy.ndarray
              A 2D numpy array of shape (voter_num, item_num) storing
              Borda scores.
            - rel_data_matrix : numpy.ndarray
              A 1D numpy array of shape (item_num,) storing relevance
              scores if rel_data is provided; otherwise, only the
              score_base_data_matrix and item_mapping are returned.
            - item_mapping : dict
              A mapping of item codes to their corresponding indices.
        """
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

    def train(self, train_file_path, train_rel_path):
        """
        Trains the model using the provided training data, calculating
        weights for each voter based on their performance on different
        queries.

        Parameters:
        -----------
        train_file_path : str
            - The file path to the training base data (e.g., ranking data). It contains the training base data with columns
            'Query', 'Voter Name', 'Item Code', and 'Item Rank'.

        train_rel_path : str
            - The file path to the relevance data (e.g., ground truth relevance scores). It contains the training relevance data with columns
            'Query', '0', 'Item Code', and 'Relevance'.
        Returns:
        --------
        None
            The method updates the internal state of the class with
            calculated weights and average weights for the voters.
        """
        # Data process
        train_base_data, train_rel_data, unique_queries = csv_load(train_file_path, train_rel_path, InputType.RANK)
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        # Establish mapping
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}

        self.weights = np.zeros((len(unique_queries), self.voter_num))

        # Consider each query
        for query in tqdm(unique_queries):
            # Filter out the data of the current query
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]

            # Convert to 2D Numpy matrix
            base_data_matrix, rel_data_matrix, _ = self.convert_to_matrix(base_data, rel_data)
            # Calculate the performance of each voter
            evaluation = Evaluation()
            for voter_idx in range(self.voter_num):
                if self.topk is None:
                    topk = np.sum(rel_data_matrix > 0)
                else:
                    topk = self.topk
                voter_w = evaluation.compute_precision(base_data_matrix[voter_idx, :], rel_data_matrix, topk,
                                                       InputType.SCORE) * self.voter_num

                query_idx = self.query_mapping[query]
                self.weights[query_idx, voter_idx] = voter_w

        # Calculate the average weight for all queries below
        self.average_weight = np.mean(self.weights, axis=0)

    def test(self, test_file_path, test_output_path, using_average_w=True):
        """
        Tests the model on the provided test data and writes the results
        to the specified output location.

        Parameters:
        -----------
        test_file_path : str
            - The file path to the test data (e.g., ranking data). The test data containing columns for queries, voter names, item codes, and item ranks.

        test_output_path : str
            The file path where the output CSV file will be saved. The output file will
            contain the ranked results for each query in the format: [Query, Item Code, Item Rank].

        using_average_w : bool, optional
            A flag to indicate whether to use average weights for scoring.
            Defaults to True.

        Returns:
        --------
        None
            The method writes the ranking results to the specified output
            location.
        """
        test_data, unique_test_queries = csv_load(test_file_path, InputType.RANK)
        test_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        unique_test_queries = test_data['Query'].unique()

        # Create an empty DataFrame to store the results
        with open(test_output_path, mode='w', newline='') as file:
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
