"""
UTF-8
Python Version: 3.11.4
TensorFlow Version: 2.15.0

Reference:
-----------
- Volkovs, M. N., & Zemel, R. S. (2013, October). CRF framework for supervised preference aggregation. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management (pp. 89-98).

Author:
-------
- Tancilon

Date:
-----
- 2023-12-24

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
1. The data input accepts partial lists.
2. Smaller Item Rank values indicate higher rankings.
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

    def convert_to_matrix(self, base_data, rel_data=None):
        """
        Convert the input data into matrices for ranking and relevance.

        Parameters:
        -----------
        base_data : DataFrame
            A DataFrame containing the base data with columns:
            - 'Voter Name': The name of the voter providing the ranking.
            - 'Item Code': The code identifying the item being ranked.
            - 'Item Rank': The rank given by the voter to the item.

        rel_data : DataFrame, optional
            A DataFrame containing relevance data with columns:
            - 'Item Code': The code identifying the item.
            - 'Relevance': The relevance label assigned to the item.

        Returns:
        --------
        r : numpy.ndarray
            A 1D numpy array of shape (item_num,) containing the relevance labels for each item.
            The relevance value indicates how relevant each item is.

        r_matrix : numpy.ndarray
            A 2D numpy array of shape (item_num, voter_num) where each entry R[k, j] represents
            the rank assigned to item k by voter j. If voter j has not ranked item k, then
            R[k, j] is set to 0.

        item_mapping : dict
            A dictionary mapping each unique item code to its corresponding index in the matrix.
        """

        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        r_matrix = np.zeros((item_num, self.voter_num))  # Initialize the ranking matrix.
        r = np.zeros(item_num)  # Initialize the relevance labels array.

        # Populate the ranking matrix R based on base_data.
        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]
            r_matrix[item_index, voter_index] = item_rank  # Assign rank to the appropriate location.

        # If relevance data is not provided, return the rank matrix and item mapping.
        if rel_data is None:
            return r_matrix, item_mapping
        else:
            # Populate the relevance array r based on rel_data.
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']

                item_index = item_mapping[item_code]
                r[item_index] = item_relevance  # Assign relevance value.

            return r, r_matrix, item_mapping  # Return relevance, rank matrix, and item mapping.

    @staticmethod
    def subsample_items(base_data, rel_data, epsilon):
        """
        Subsample items from the relevance data to ensure diversity in sampling.

        This function samples items from the `rel_data` based on their relevance scores, ensuring that:
        - Each unique relevance value is represented.
        - The total number of samples does not exceed the specified epsilon.
        - Each sampled item code is unique.

        Parameters:
        -----------
        base_data : DataFrame
            A DataFrame containing the base data with an 'Item Code' column.
            This data will be filtered based on the sampled items from `rel_data`.

        rel_data : DataFrame
            A DataFrame containing the relevance data with an 'Item Code' and 'Relevance' columns.
            This data will be used to perform the sampling.

        epsilon : int
            The maximum number of unique samples to return. If the total number of unique samples exceeds
            this value, a random selection will be made.

        Returns:
        --------
        filtered_base_data : DataFrame
            A filtered version of `base_data` that includes only the rows with 'Item Code' present in the
            sampled items.

        sampled_data : DataFrame
            A DataFrame containing the sampled items from `rel_data`, ensuring that each relevance value
            is represented and that the number of unique sampled items is at least epsilon.
        """

        # Ensure at least one sample from each unique Relevance value
        unique_relevance_samples = rel_data.groupby('Relevance').apply(lambda x: x.sample(1))

        # Sample up to epsilon items if there are more than epsilon unique samples
        if len(unique_relevance_samples) > epsilon:
            sampled_data = unique_relevance_samples.sample(epsilon)
        else:
            sampled_data = unique_relevance_samples

        # Ensure that the number of unique Item Codes is at least epsilon
        while len(sampled_data['Item Code'].unique()) < epsilon:
            additional_samples = rel_data.sample(1)
            sampled_data = pd.concat([sampled_data, additional_samples]).drop_duplicates(
                subset='Item Code').reset_index(drop=True)

        # Process base_data based on the sampled results

        # Get unique item codes from sampled_data
        sampled_item_codes = sampled_data['Item Code'].unique()
        # Filter base_data to include only rows with Item Codes in the sampled item codes
        filtered_base_data = base_data[base_data['Item Code'].isin(sampled_item_codes)]

        return filtered_base_data, sampled_data

    @staticmethod
    def compute_loss(y, r, loss_cut_off):
        """
        Compute the loss based on the Normalized Discounted Cumulative Gain (NDCG) metric.

        This function calculates the loss as one minus the NDCG score, which measures the quality
        of ranking. A higher NDCG indicates a better ranking of relevant items. The loss cut-off
        can be specified to limit the evaluation to a certain number of relevant items.

        Parameters:
        -----------
        y : numpy.ndarray
            A 1D array containing the ranks of the items, where the values represent the ranking
            assigned to each item.

        r : numpy.ndarray
            A 1D array containing the relevance labels for the items, where the values indicate
            the relevance of each item (typically binary or graded relevance).

        loss_cut_off : int, optional
            The number of relevant items to consider for calculating the NDCG. If set to None, it
            defaults to the total count of relevant items (i.e., r > 0).

        Returns:
        --------
        float
            The computed loss, calculated as 1 minus the NDCG score. A lower loss indicates a
            better ranking performance.

        Notes:
        ------
        - The function assumes that the NDCG score is computed using the relevant items, and the
          loss is minimized as the ranking improves.
        """

        # If loss_cut_off is not manually set, modify it to the number of relevant documents
        if loss_cut_off is None:
            loss_cut_off = np.sum(r > 0)

        evaluation = Evaluation()
        ndcg = evaluation.compute_ndcg_r(y, r, loss_cut_off)
        return 1 - ndcg

    @staticmethod
    def compute_negative_energy(y, r_matrix, theta):
        """
        Compute the negative energy based on item rankings and voter relevance.

        This function calculates the negative energy, which is a measure used in ranking systems
        to evaluate how well the ranked items align with the relevance scores provided by voters.
        The negative energy is influenced by the ranks assigned to each item and the relevance
        ratings in the r_matrix.

        Parameters:
        -----------
        y : numpy.ndarray
            A 1D array containing the ranks of the items, where each value corresponds to the rank
            assigned to each item (1 to item_num).

        r_matrix : numpy.ndarray
            A 2D array of shape (item_num, voter_num), containing the relevance scores for each item
            as ranked by the voters. A value of 0 indicates that a voter did not rank the item.

        theta : tf.Variable
            A TensorFlow variable containing weight parameters that influence the calculation of
            negative energy. The specific elements of theta correspond to the contributions from
            various ranks.

        Returns:
        --------
        float
            The computed negative energy, averaged over the number of items. A lower negative energy
            indicates better alignment between item rankings and voter relevance scores.

        Notes:
        ------
        - The computation considers the contributions of all voters to the negative energy for each
          item based on their ranking and relevance scores.
        - The logarithmic normalization of the item index is used to adjust the impact of each rank
          on the negative energy computation.
        """
        item_num = len(y)
        voter_num = r_matrix.shape[1]
        # i: Consider all rankings
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

    def train(self, train_base_data, train_rel_data, alpha=0.01, epsilon=5, epoch=300, loss_cut_off=None):
        """
        Train the model using the provided training data.

        This function processes the training data, initializes weights, and performs
        optimization through multiple epochs to update the model's parameters. It
        calculates gradients using TensorFlow's automatic differentiation and
        adjusts the weights based on the computed gradients.

        Parameters:
        -----------
        train_base_data : pandas.DataFrame
            The base training data containing columns for queries, voter names, item codes, and item ranks.

        train_rel_data : pandas.DataFrame
            The relevance training data containing columns for queries, an unused column, item codes, and relevance scores.

        alpha : float, optional
            The learning rate for weight updates. Default is 0.01.

        epsilon : int, optional
            The cut-off threshold for sampling items. Default is 5.

        epoch : int, optional
            The number of iterations for training. Default is 300.

        loss_cut_off : int or None, optional
            The cut-off for the loss computation (k in ndcg@k). If None, will use the number of relevant documents.

        Returns:
        --------
        None
            This function updates the internal state of the model (i.e., the weights) but does not return a value.

        Notes:
        ------
        - The function initializes weights to zero and iterates through a specified number of epochs.
        - During each epoch, it processes the data for each unique query, performing subsampling if necessary.
        - The loss is computed based on the rankings of items, and gradients are calculated for updating weights.
        - The function uses permutations of item rankings to explore all possible arrangements, impacting the objective function.
        """
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}

        # Initialize weights
        self.weights = np.zeros(3 * self.voter_num)

        # Repeat CRF optimization
        for epo in range(epoch):
            for query in tqdm(unique_queries):
                # Filter data for the current query
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

                # Compute gradients
                theta = tf.Variable(self.weights)

                # Use tf.GradientTape() to record the computation process
                with tf.GradientTape() as tape:

                    # Initialize y
                    objective = 0.0
                    # Enumerate all possible rankings
                    initial_perm = np.empty(len(r))
                    for i in range(len(r)):
                        initial_perm[i] = i + 1
                    all_permutations = permutations(initial_perm)
                    # y: 1 * item 1D numpy array storing item rankings
                    sum_exp_negative_energy = 0.0
                    for perm in all_permutations:
                        y = np.array(perm)
                        loss = self.compute_loss(y, r, loss_cut_off)
                        negative_energy = self.compute_negative_energy(y, r_matrix, theta)
                        objective += loss * tf.exp(negative_energy)
                        sum_exp_negative_energy += tf.exp(negative_energy)
                    objective /= sum_exp_negative_energy

                # Compute the gradient of the objective function with respect to theta
                grad = tape.gradient(objective, theta)

                # Update weights using the calculated gradient
                self.weights = self.weights - alpha * grad.numpy()

    def test(self, test_data, output):
        """
        Test the model using the provided test data and output the results to a CSV file.

        This function processes the test data to generate scores for each item based
        on the model's learned weights. It ranks the items for each unique query
        and writes the rankings to an output CSV file.

        Parameters:
        -----------
        test_data : pandas.DataFrame
            The test data containing columns for queries, voter names, item codes, and item ranks.

        output : str
            The file path where the output CSV file will be saved. The output file will
            contain the ranked results for each query in the format: [Query, Item Code, Item Rank].

        Returns:
        --------
        None
            This function does not return a value but writes the output to the specified CSV file.

        Notes:
        ------
        - The function first processes the test data to create a relevance matrix (r_matrix)
          for each query.
        - It calculates a score for each item based on the voter's weights and ranks the items
          accordingly.
        - The results are written to the output CSV file, with each row representing a
          query, an item code, and its corresponding rank.

        Example Output Format:
        -----------------------
        Query | Item Code | Item Rank
        """
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
