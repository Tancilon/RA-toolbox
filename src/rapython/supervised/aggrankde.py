"""
AggRankDE Algorithm

This module implements the AggRankDE algorithm for rank aggregation.
This implementation is based on the following reference:

Reference:
-----------
- Bałchanowski, M., & Boryczka, U. (2022). Aggregation of rankings using metaheuristics in recommendation systems. Electronics, 11(3), 369.

Authors:
    Qi Deng
Date:
    2023-12-19

Training Data Format:
---------------------
The training data consists of two CSV files:

1. train_rel_data:
   - Format: CSV
   - Columns: `Query`, `0`, `Item`, `Relevance`

2. train_base_data:
   - Format: CSV
   - Columns: `Query`, `Voter Name`, `Item Code`, `(Item Rank / Item Score)`
   - Notes:
     - `Query`: Does not need to be consecutive integers starting from 1.
     - `Voter Name` and `Item Code`: Can be strings.
     - By default, `train_base_data` is in `Item Rank` format. To use `Item Score`, set the model parameter `type = score`.
       - If `type = score` is set, both training and testing datasets must use the `score` format.

Output Format:
--------------
The algorithm produces the following output as a CSV file:

- Columns: `Query`, `Item Code`, `Item Rank`
  - Note: The output contains rank information, not score information.

Test Data Format:
-----------------
The test data consists of one CSV file:

1. test_data:
   - Format: CSV
   - Columns: `Query`, `Voter Name`, `Item Code`, `(Item Rank / Item Score)`
   - Notes:
     - `Query`: Does not need to be consecutive integers starting from 1.
     - `Voter Name` and `Item Code`: Can be strings.

Additional Notes:
-----------------
1. Full lists of data are recommended for best performance. For partial lists, the algorithm assigns a score of 0 to unrated items.
2. The smaller the `Item Rank`, the higher the item is ranked.
3. The same voters should be used in both the training and testing datasets.
"""

import csv
import random

import numpy as np
from tqdm import tqdm

from src.rapython.evaluation import Evaluation
from src.rapython.datatools import InputType, csv_load


def matrix_fitness_function(a_value, p1_value, p1_fitness, p2_value, rel_list, n_value):
    """
    Evaluate the fitness of individuals after crossover and mutation in a ranking-based genetic algorithm.

    Parameters
    ----------
    a_value : numpy.ndarray
        The initial ranking matrix, where rows represent items and columns represent voters.
        Each element stores the score for a given item from a voter.

    p1_value : numpy.ndarray
        The current population's individual values (solutions) after selection, mutation, and crossover.

    p1_fitness : numpy.ndarray
        The fitness scores of the current population.

    p2_value : numpy.ndarray
        The new candidate solutions after crossover and mutation.

    rel_list : numpy.ndarray
        A list containing relevance values for each item, used to compute Average Precision (AP).

    n_value : int or None
        The value of N used for computing AP@N (Average Precision at N). If None, N will be set to
        the number of relevant items (i.e., items where rel_list > 0).

    Returns
    -------
    p1_fitness : numpy.ndarray
        The updated fitness scores for the population after considering the new candidate solutions.

    p1_value : numpy.ndarray
        The updated individual values (solutions) after replacing the worse-performing individuals
        with better-performing ones based on the fitness evaluation.

    Notes
    -----
    - For each individual in the candidate population (p2_value), the function evaluates its performance
      based on the Average Precision (AP) score using the relevance list. If the new individual has a
      better AP score than the current one, the new individual replaces the old one.
    - The fitness evaluation uses a sorted list of items based on scores for each voter.
    """
    evaluation = Evaluation()
    c2_value = np.dot(a_value, p2_value)
    np_value = p2_value.shape[1]

    # If N is not provided, set N to the number of relevant items.
    if n_value is None:
        n_value = np.sum(rel_list > 0)

    for i in range(np_value):
        item_column = c2_value[:, i]
        # Sort item scores in descending order
        rank_list = np.argsort(item_column)[::-1] + 1

        # Compute the average precision (AP) for the current rank list
        ap = evaluation.compute_average_precision(rank_list, rel_list, n_value, InputType.RANK)

        # If the new individual's AP is better than the current one, replace it
        if ap > p1_fitness[i]:
            p1_fitness[i] = ap
            p1_value[:, i] = p2_value[:, i]

    return p1_fitness, p1_value


def de(a_value, rel_data, max_iteration, cr_value, f_value, np_value, d_value, n_value, item_mapping):
    """
    Differential Evolution (DE) algorithm for optimizing ranking-based problems using average precision (AP) as the fitness metric.

    Parameters
    ----------
    a_value : numpy.ndarray
        The initial ranking matrix where rows represent items and columns represent voters. Each element stores the score for a given item from a voter.

    rel_data : pandas.DataFrame
        Relevance data used to evaluate the fitness of solutions. It contains columns 'Item Code' and 'Relevance', which map items to their relevance scores.

    max_iteration : int
        The maximum number of iterations for the DE algorithm.

    cr_value : float
        The crossover probability, a value between 0 and 1 that controls the probability of crossover during the evolution process.

    f_value : float
        The mutation factor, a constant that scales the difference between two randomly selected individuals for mutation.

    np_value : int
        The number of individuals in the population (i.e., population size).

    d_value : int
        The dimensionality of the solution space, which corresponds to the number of voters.

    n_value : int or None
        The value of N used for computing AP@N (Average Precision at N). If None, N is set to the number of relevant items (where `rel_list > 0`).

    item_mapping : dict
        A dictionary mapping 'Item Code' to its corresponding item index.

    Returns
    -------
    numpy.ndarray
        The best individual (solution) found after running the DE algorithm. This is a column vector representing the best scoring solution based on AP.

    Notes
    -----
    - The algorithm initializes a population of random individuals (solutions) and evolves them using mutation, crossover, and selection.
    - The fitness of each individual is evaluated based on the Average Precision (AP) computed from the ranking generated by the individual's scores.
    - The algorithm iterates until the maximum number of iterations is reached or the population converges to a satisfactory solution.
    """

    # Initialize relevance list for items
    rel_list = np.zeros(a_value.shape[0])
    for _, row in rel_data.iterrows():
        item_code = row['Item Code']
        item_rel = row['Relevance']
        item_id = item_mapping[item_code]
        rel_list[item_id] = item_rel

    # Generate initial population matrix: voter * NP (number of individuals)
    voter_num = a_value.shape[1]
    p1_value = np.empty((voter_num, np_value))

    # Set upper and lower bounds for random initialization
    upperbound = 1
    lowerbound = 0

    # Randomly initialize the population
    for i in range(voter_num):
        for j in range(np_value):
            p1_value[i, j] = random.uniform(lowerbound, upperbound)

    # Compute initial fitness of the population
    c0_val = np.dot(a_value, p1_value)
    fitness = np.empty(np_value)
    evaluation = Evaluation()

    if n_value is None:
        n_value = np.sum(rel_list > 0)

    for i in range(np_value):
        item_column = c0_val[:, i]
        rank_list = np.argsort(item_column)[::-1] + 1
        fitness[i] = evaluation.compute_average_precision(rank_list, rel_list, n_value, InputType.RANK)

    # Evolution process begins
    p2_val = np.empty((voter_num, np_value))
    iteration = 0

    while iteration < max_iteration:
        for i in range(np_value):
            # Randomly select 3 distinct individuals for mutation
            a = random.randint(0, np_value - 1)
            while a == i:
                a = random.randint(0, np_value - 1)

            b = random.randint(0, np_value - 1)
            while b == i or b == a:
                b = random.randint(0, np_value - 1)

            c = random.randint(0, np_value - 1)
            while c == i or c == a or c == b:
                c = random.randint(0, np_value - 1)

            # Generate a random index to ensure at least one gene is mutated
            rnbr_i = random.randint(0, d_value - 1)

            for j in range(d_value):
                if random.random() <= cr_value or j == rnbr_i:
                    # Mutation step
                    p2_val[j, i] = p1_value[j, a] + f_value * (p1_value[j, b] - p1_value[j, c])
                else:
                    # Crossover step
                    p2_val[j, i] = p1_value[j, i]

        # Evaluate new population and select the best individuals
        fitness, p1_value = matrix_fitness_function(a_value, p1_value, fitness, p2_val, rel_list, n_value)
        iteration += 1

    # After the evolution, find the best individual in the population
    max_index = np.argmax(fitness)
    return p1_value[:, max_index]


class AggRankDE:
    def __init__(self, np_val=50, max_iteration=100, cr=0.9, f=0.5, input_type=InputType.RANK, n=None):
        """
        Initializes the parameters for the AggRankDE algorithm.

        Parameters
        ----------
        np_val : int, optional
            The size of the population (NP), representing the number of candidate solutions.
            Default is 50.

        max_iteration : int, optional
            The maximum number of iterations for the optimization process.
            Default is 100.

        cr : float, optional
            Crossover probability (CR) for the differential evolution algorithm,
            which should be in the range [0, 1]. A higher value increases the chance
            of crossover in the population. Default is 0.9.

        f : float, optional
            The amplification factor (F) for the differential evolution algorithm,
            which controls the amount of perturbation applied to the population.
            It should be in the range [0, 2]. Default is 0.5.

        input_type : InputType, optional
            Specifies whether the input data is treated as ranks or scores.
            The default is InputType.RANK.
            Both training and test datasets should be consistently formatted as ranks or scores.

        n : int, optional
            Represents the parameter for the fitness function, specifically for calculating
            average precision at N (AP@N). If set to None, it assumes N is the number of relevant items.
            Default is None.
        """
        self.NP = np_val
        self.max_iteration = max_iteration
        self.CR = cr
        self.F = f
        self.type = input_type
        self.N = n
        self.weights = None
        self.average_weight = None
        self.voter_name_reverse_mapping = None
        self.voter_name_mapping = None
        self.voter_num = None
        self.query_mapping = None

    def convert_to_matrix(self, base_data):
        """
        Converts the base_data DataFrame into a 2D matrix where rows represent items and columns represent voters.
        The matrix stores either Item Scores or Item Ranks based on the input type.

        Parameters
        ----------
        base_data : pandas.DataFrame
            The input data containing the following columns:
            - 'Item Code': The unique code identifying each item.
            - 'Voter Name': The name of the voter (can be a string).
            - 'Item Attribute': The attribute (either rank or score) provided by each voter for an item.

        Returns
        -------
        numpy.ndarray
            A 2D matrix (item_num * voter_num) where each element represents the normalized score or rank for an item provided by a voter.

        item_mapping : dict
            A dictionary mapping each 'Item Code' to its corresponding row index in the matrix.

        Notes
        -----
        - If the input type (`self.type`) is `RANK`, the item attribute will be converted to a score by taking the reciprocal of the rank (1 / rank).
        - If the input type is `SCORE`, the item attribute will be stored directly as the score.
        - After constructing the matrix, it is normalized by the L2 norm along each column (i.e., each voter's scores are normalized).
        - A small epsilon (1e-10) is added to the L2 norm to avoid division by zero during normalization.
        """

        # Extract unique items from the base data and create a mapping
        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}

        # Initialize a zero matrix to store item scores or ranks (item_num * voter_num)
        a = np.zeros((item_num, self.voter_num))

        # Iterate through the base_data to fill the matrix
        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_attribute = row['Item Attribute']

            # Get the corresponding voter and item indices
            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]

            # Convert based on input type (rank or score)
            if self.type == InputType.RANK:
                a[item_index, voter_index] = 1 / item_attribute  # Convert rank to score
            elif self.type == InputType.SCORE:
                a[item_index, voter_index] = item_attribute  # Use score directly

        # Normalize the matrix along each column (L2 norm)
        column_norms = np.linalg.norm(a, axis=0)

        # Add a small epsilon to avoid division by zero
        column_norms_safe = column_norms + 1e-10
        normalized_a = a / column_norms_safe

        return normalized_a, item_mapping

    def train(self, train_file_path, train_rel_path, input_type):
        """
        Trains the model using the provided training data and relevance information.

        Parameters:
        -----------
        train_file_path : str
            - The file path to the training base data (e.g., ranking data).
        train_rel_path : str
            - The file path to the relevance data (e.g., ground truth relevance scores).
        input_type : InputType
            - Specifies the format or type of the input data. InputType.RANK is recommended.

        Returns:
        --------
        None
            - The function does not return a value, but it trains the model and stores the
              resulting weights in `self.weights` and `self.average_weight`.
        """

        train_base_data, train_rel_data, unique_queries = csv_load(train_file_path, train_rel_path, input_type)
        # Rename columns in the training base data for consistency
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Attribute']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        # Get unique voter names from the training relevance data
        unique_voter_names = train_base_data['Voter Name'].unique()

        # Store the number of unique voters
        self.voter_num = len(unique_voter_names)

        # Create mappings for voter names
        # Integer to string reverse mapping for voters
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        # String to integer mapping for voters
        self.voter_name_mapping = {v: k for k, v in self.voter_name_reverse_mapping.items()}
        # Mapping for queries
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}

        # Initialize weights array for the queries
        self.weights = np.empty((len(unique_queries), self.voter_num))

        # Process each query to calculate weights
        for query in tqdm(unique_queries):
            # Filter the base and relevance data for the current query
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]

            # Convert the base data into a matrix form
            a, item_mapping = self.convert_to_matrix(base_data)

            # Apply differential evolution to calculate weights for the current query
            w = de(a, rel_data, self.max_iteration, self.CR, self.F, self.NP, self.voter_num, self.N, item_mapping)

            # Store the weight vector obtained for the current query
            query_index = self.query_mapping[query]
            self.weights[query_index, :] = w

        # Calculate the average weight across all queries

        self.average_weight = np.mean(self.weights, axis=0)

    def test(self, test_file_path, test_output_path, using_average_w=True):
        """
        Tests the model using the provided test data and writes the ranked results to a CSV file.

        Parameters
        ----------
        test_file_path : str
            - The file path to the test data (e.g., ranking data).
        test_output_path : str
            - The file path where the ranked results will be written in CSV format.
        using_average_w : bool, optional
            - A flag indicating whether to use the average weights across queries for scoring.
              If True, the model uses the average weights; otherwise, it uses query-specific weights if available.
              Default is True.

        Returns
        -------
        None
            - The function does not return a value but writes the ranked items to the CSV file specified by `test_output_path`.
        """
        test_data, unique_test_queries = csv_load(test_file_path, InputType.RANK)
        # Rename columns in the test data for consistency
        test_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Attribute']

        # Open the output CSV file for writing results
        with open(test_output_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Process each unique query in the test data
            for query in tqdm(unique_test_queries):
                # Filter the test data for the current query
                query_data = test_data[test_data['Query'] == query]

                # Convert the filtered query data into a matrix format
                A, item_code_mapping = self.convert_to_matrix(query_data)
                item_code_reverse_mapping = {v: k for k, v in item_code_mapping.items()}

                # Calculate scores based on the selected weight parameters
                if using_average_w:
                    # Use average weights to compute scores for the current query
                    score_list = np.dot(self.average_weight, A.T)
                else:
                    # Check if the current query exists in the mapping
                    if query not in self.query_mapping:
                        # Fallback to average weights if the query is not found
                        score_list = np.dot(self.average_weight, A.T)
                    else:
                        # Use the specific weights for the current query
                        query_id = self.query_mapping[query]
                        score_list = np.dot(self.weights[query_id, :], A.T)

                # Rank items based on the calculated scores
                rank_list = np.argsort(score_list)[::-1]

                # Write the ranked results to the output CSV file
                for rank_index, item_id in enumerate(rank_list):
                    # Retrieve the item code using the reverse mapping
                    item_code = item_code_reverse_mapping[item_id]
                    # Prepare a new row with query, item code, and rank index
                    new_row = [query, item_code, (rank_index + 1)]
                    writer.writerow(new_row)
