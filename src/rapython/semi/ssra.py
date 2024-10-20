"""
Semi-supervised Ranking Aggregation Algorithm

This module is based on the following reference:

Reference:
-----------
- Chen, S., Wang, F., Song, Y., & Zhang, C. (2008, October). Semi-supervised ranking aggregation. In Proceedings of the 17th ACM conference on Information and knowledge management (pp. 1427-1428).

Authors:
    tancilon
Date:
    2024-01-18

Training Data Input Format:
---------------------------
1. train_rel_data (Relevance Data):
   - CSV file format
   - 4 columns: Query | 0 | Item | Relevance
2. train_base_data (Base Ranking Data):
   - CSV file format
   - 4 columns: Query | Voter name | Item Code | Item Rank

Notes:
- The 'Query' column does not need to be a consecutive sequence starting from 1.
- 'Voter name' and 'Item Code' can be in string format.

Output Format:
--------------
The algorithm's output is a CSV file with 3 columns: Query | Item Code | Item Rank
- The output provides rank information, not scores.

Test Data Input Format:
-----------------------
1. test_data:
   - CSV file format
   - 4 columns: Query | Voter name | Item Code | Item Rank

Notes:
- The 'Query' column does not need to be a consecutive sequence starting from 1.
- 'Voter name' and 'Item Code' can be in string format.

Additional Details:
-------------------
1. Input data can be a full list; for partial lists, items not ranked are treated as having the lowest rank.
2. A smaller 'Item Rank' value indicates a higher rank (i.e., the item is ranked closer to the top).
"""

import csv

import cvxpy as cp
import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm


class SSRA:
    def __init__(self):
        self.weights = None
        self.average_weight = None
        self.voter_name_mapping = None
        self.voter_name_reverse_mapping = None
        self.voter_num = None
        self.is_partial_list = None
        self.query_mapping = None
        self.rank_base_data_matrix = None  # np.ndarray (voter, item)

    @staticmethod
    def partial_to_full(rank_base_data_matrix):
        num_voters = rank_base_data_matrix.shape[0]

        for k in range(num_voters):
            if np.isnan(rank_base_data_matrix[k]).all():
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], nan=rank_base_data_matrix.shape[1])
            else:
                max_rank = np.nanmax(rank_base_data_matrix[k])
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], nan=max_rank + 1)

        return rank_base_data_matrix

    def convert_to_matrix(self, base_data, rel_data=None):
        """
        Converts base ranking data and optional relevance data into matrices.

        This method takes in a base ranking dataset (which contains the rank of items as per different voters)
        and optionally relevance data, and converts them into two matrices:
        1. A score matrix for Borda scores based on rankings.
        2. A relevance matrix if relevance data is provided.

        Parameters:
        -----------
        base_data : pd.DataFrame
        rel_data : pd.DataFrame, optional

        Returns:
        --------
        score_base_data_matrix : np.ndarray
            - A matrix (item * voter) where each element stores the Borda score for a given item by a voter.
            - If `rel_data` is not provided, only the score matrix is returned.
        rel_data_matrix : np.ndarray, optional
            - A 1D array (1 * item) storing the relevance scores for each item.
            - Only returned if `rel_data` is provided.
        item_mapping : dict
            - A mapping of item codes to their corresponding index in the matrices.
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
            rank_base_data_matrix = self.partial_to_full(rank_base_data_matrix)

        self.rank_base_data_matrix = rank_base_data_matrix

        for k in range(self.voter_num):
            max_rank = np.max(rank_base_data_matrix[k, :])
            for i in range(item_num):
                score_base_data_matrix[k, i] = max_rank - rank_base_data_matrix[k, i] + 1

        if rel_data is None:
            score_base_data_matrix = score_base_data_matrix.T
            return score_base_data_matrix, item_mapping
        else:
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']

                item_index = item_mapping[item_code]
                rel_data_matrix[item_index] = item_relevance

            score_base_data_matrix = score_base_data_matrix.T
            return score_base_data_matrix, rel_data_matrix, item_mapping

    def get_norm_similarity(self):
        p = np.zeros(self.voter_num)
        for i in range(self.voter_num):
            for j in range(i + 1, self.voter_num):
                tau, _ = kendalltau(self.rank_base_data_matrix[i, :], self.rank_base_data_matrix[j, :])
                if np.isnan(tau):
                    normalized_tau = 0
                else:
                    normalized_tau = (1 - tau) / 2
                p[i] += (1 / self.voter_num) * normalized_tau
                p[j] += (1 / self.voter_num) * normalized_tau

        return p

    @staticmethod
    def borda_count(rankings):
        num_voters, num_items = rankings.shape
        scores = np.zeros(num_items)
        for i in range(num_items):
            scores[i] = np.sum(num_items - rankings[:, i])
        return scores

    @staticmethod
    def borda_weighted_graph(scores):
        num_items = len(scores)
        adjacency_matrix = np.zeros((num_items, num_items))

        for i in range(num_items):
            for j in range(i + 1, num_items):
                if i != j:
                    # Use the reciprocal of the absolute value of the score difference as the weight
                    adjacency_matrix[i, j] = 1.0 / (np.abs(scores[i] - scores[j]) + 1e-5)

                    adjacency_matrix[j, i] = adjacency_matrix[i, j]

        return adjacency_matrix

    def get_laplacian(self):
        # Calculate Borda count score
        borda_scores = self.borda_count(self.rank_base_data_matrix)

        # Create adjacency matrix based on Borda count
        w = self.borda_weighted_graph(borda_scores)
        row_sums = np.sum(w, axis=1)
        d = np.diag(row_sums)
        l_val = d - w
        return l_val

    @staticmethod
    def regularize_to_positive_semidefinite_matrix(arr, regularization_param):
        n = arr.shape[0]  # Obtain the dimensions of the matrix
        i_matrix = np.identity(n)  # Create an identity matrix

        # Add regularization parameters to diagonal elements
        arr_reg = arr + regularization_param * i_matrix

        # Calculate the corrected Cholesky decomposition
        l_matrix = np.linalg.cholesky(arr_reg)

        # Constructing a semi positive definite matrix
        positive_semidefinite_matrix = np.dot(l_matrix, l_matrix.T)

        return positive_semidefinite_matrix

    def train(self, train_base_data, train_rel_data, alpha=0.03, beta=0.1, constraints_rate=0.3, is_partial_list=True):
        """
        Train the model using the provided training data.

        Parameters:
        -----------
        train_base_data : pd.DataFrame
            - Contains ranking information with the following columns:
              'Query', 'Voter Name', 'Item Code', and 'Item Rank'.
        train_rel_data : pd.DataFrame
            - Contains relevance information with the following columns:
              'Query', '0', 'Item Code', and 'Relevance'.
        alpha : float, optional
            - A hyperparameter typically ranging from 0.01 to 0.1 that controls the influence of the quadratic form in the objective function.
        beta : float, optional
            - A hyperparameter typically ranging from 0.01 to 0.1 that regulates the L2 norm of the weight vector in the objective function.
        constraints_rate : float, optional
            - The proportion of supervisory information used, ranging from 0 to 1. If it is 1, the method becomes fully supervised.
        is_partial_list : bool, optional
            - Indicates whether the training data contains a partial list of items.

        Returns:
        --------
        None
            - The method modifies the model's internal state by calculating and storing the weights based on the training data.
        """
        # Set column names for base and relevance data
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        # Get unique queries and voter names from the data
        unique_queries = train_rel_data['Query'].unique()
        unique_voter_names = train_base_data['Voter Name'].unique()

        self.is_partial_list = is_partial_list
        self.voter_num = len(unique_voter_names)

        # Create mappings for voter names and queries
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}

        # Initialize weight matrix
        self.weights = np.zeros((len(unique_queries), self.voter_num))

        # Iterate over each unique query to compute weights
        for query in tqdm(unique_queries):
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]

            # Convert the base and relevance data into matrix form
            r_matrix, rel_data_matrix, _ = self.convert_to_matrix(base_data, rel_data)

            # Identify relevant and irrelevant items based on the relevance data
            rel_items = np.where(rel_data_matrix >= 1)[0]
            irrel_items = np.where(rel_data_matrix <= 0)[0]

            # Skip if there are no relevant items
            if len(rel_items) == 0:
                continue

            # Determine the number of relevant items to use based on constraints_rate
            rel_nums = int(len(rel_items) * constraints_rate)
            if rel_nums < 1:
                rel_nums = 1

            # Calculate the normalized similarity and Laplacian
            p = self.get_norm_similarity()
            l_prime = self.get_laplacian()

            # Define optimization variable
            w = cp.Variable(p.shape[0])

            # Compute the matrix for the quadratic form in the objective
            matrix = r_matrix.T @ l_prime @ r_matrix

            # Ensure the matrix is positive semi-definite
            if not np.all(np.linalg.eigvals(matrix) >= 0):
                regularization_param = 0.01
                matrix = self.regularize_to_positive_semidefinite_matrix(matrix, regularization_param)

            matrix = cp.psd_wrap(matrix)

            # Define the objective function
            objective = cp.Minimize(
                cp.norm(w - p) ** 2 + alpha * cp.quad_form(w, matrix) + (beta / 2) * cp.norm(w) ** 2)

            # Define the constraints for the optimization problem
            constraints = []
            for indexi in range(rel_nums):
                for indexj in range(len(irrel_items)):
                    i = rel_items[indexi]
                    j = irrel_items[indexj]
                    constraints.append(r_matrix[i, :] @ w.T - r_matrix[j, :] @ w.T >= 1)

            constraints.append(w >= 0)
            constraints.append(cp.sum(w) == 1)

            # Set up the optimization problem
            problem = cp.Problem(objective, constraints)

            # Solve the optimization problem
            problem.solve(solver=cp.SCS)

            query_idx = self.query_mapping[query]

            # Store the computed weights for the current query
            self.weights[query_idx, :] = w.value

        # Create a boolean index array to identify non-zero rows
        non_zero_rows = np.any(self.weights != 0, axis=1)

        # Check for NaN values in each row
        not_nan_rows = ~np.any(np.isnan(self.weights), axis=1)

        # Select rows that are both non-zero and not NaN for averaging
        filtered_weights = self.weights[non_zero_rows & not_nan_rows]

        # Compute the average of the selected weights
        self.average_weight = np.mean(filtered_weights, axis=0)

    def test(self, test_data, test_output_loc, using_average_w=True):
        """
        Test the model using the provided test data and save the results to a specified location.

        Parameters:
        -----------
        test_data : pd.DataFrame
            - Contains testing data with the following columns:
              'Query', 'Voter Name', 'Item Code', and 'Item Rank'.
        test_output_loc : str
            - The file path where the results will be saved in CSV format.
        using_average_w : bool, optional
            - Indicates whether to use the average weights for scoring. If False, uses specific query weights.

        Returns:
        --------
        None
            - The method writes the ranked results to a CSV file at the specified location.
        """
        # Rename the columns of the test data for consistency
        test_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

        # Get the unique queries present in the test data
        unique_test_queries = test_data['Query'].unique()

        # Create a CSV file to store the test results
        with open(test_output_loc, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Iterate through each unique query in the test data
            for query in tqdm(unique_test_queries):
                # Filter the test data for the current query
                query_data = test_data[test_data['Query'] == query]

                query_data_matrix, item_code_mapping = self.convert_to_matrix(query_data)
                query_data_matrix = query_data_matrix.T  # Transpose the matrix for scoring

                # Create a reverse mapping for item codes to retrieve them later
                item_code_reverse_mapping = {v: k for k, v in item_code_mapping.items()}

                # Compute scores based on whether to use average weights or specific weights
                if using_average_w:
                    score_list = np.dot(self.average_weight, query_data_matrix)
                else:
                    # If not using average weights, check for specific query weights
                    if query not in self.query_mapping:
                        # If the query is not found, fall back to average weights
                        score_list = np.dot(self.average_weight, query_data_matrix)
                    else:
                        # Retrieve the query ID and use its specific weights
                        query_id = self.query_mapping[query]
                        score_list = np.dot(self.weights[query_id, :], query_data_matrix)

                # Generate a rank list by sorting the scores in descending order
                rank_list = np.argsort(score_list)[::-1]

                # Write the ranked items to the CSV file
                for rank_index, item_id in enumerate(rank_list):
                    item_code = item_code_reverse_mapping[item_id]
                    new_row = [query, item_code, (rank_index + 1)]
                    writer.writerow(new_row)
