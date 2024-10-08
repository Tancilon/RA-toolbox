# Reference: Comb* family
# fsw, Tancilon：20231018
# Define the input to the algorithm as a csv file format: Query | Voter name | Item Code | Item Rank
#      - Query does not require consecutive integers starting from 1.
#      - Voter name and Item Code are allowed to be in String format.
# Define the final output of the algorithm as a csv file format： Query | Item Code | Item Rank
#      - Output is the rank information, not the score information
# The smaller the Item Rank, the higher the rank.

import numpy as np
import pandas as pd
import csv
import scorefunc as sc


def combmnz_agg(input_list):
    """
    Aggregate scores for items based on rankings provided by voters.
    Parameters
    ----------
    input_list : numpy.ndarray
        A 2D array where each row represents a voter's ranking of items.

    Returns
    -------
    numpy.ndarray
        An array containing the final ranks of the items after aggregation.
    """
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_comb_score = np.zeros(num_items)
    # Convert rankings to scores using different methods
    item_score = sc.linearagg(input_list)
    print(item_score)

    for i in range(num_items):
        item_min_score = np.zeros(num_voters)
        for k in range(num_voters):
            item_min_score[k] = item_score[k, i]
        # Calculate aggregated score for each item
        item_comb_score[i] = num_voters * sum(item_min_score)
    first_row = item_comb_score
    # Sort scores and get indices
    sorted_indices = np.argsort(first_row)[::-1]

    currrent_rank = 1
    result = np.zeros(num_items)
    for index in sorted_indices:
        result[index] = currrent_rank
        currrent_rank += 1
    return result


def combmnz(input_file_path, output_file_path):
    """
    Process the input CSV file to aggregate rankings and write results to an output CSV file.
    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file.
    output_file_path : str
        Path to the output CSV file where aggregated rankings will be written.
    """
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    # Get unique Query values
    unique_queries = df['Query'].unique()
    # Create an empty DataFrame to store results
    result = []

    for query in unique_queries:
        # Filter data for the current Query
        query_data = df[df['Query'] == query]

        # Get unique Item Code and Voter Name values, create mappings
        unique_item_codes = query_data['Item Code'].unique()
        unique_voter_names = query_data['Voter Name'].unique()

        # Create reverse mappings
        item_code_reverse_mapping = {i: code for i, code in enumerate(unique_item_codes)}
        voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}

        # Create forward mappings
        item_code_mapping = {v: k for k, v in item_code_reverse_mapping.items()}
        voter_name_mapping = {v: k for k, v in voter_name_reverse_mapping.items()}

        # Initialize a 2D Numpy array for Voter Name * Item Code
        num_voters = len(unique_voter_names)
        num_items = len(unique_item_codes)
        input_list = np.full((num_voters, num_items), np.nan)

        # Fill the array with ranking data
        for index, row in query_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = voter_name_mapping[voter_name]
            item_index = item_code_mapping[item_code]

            input_list[voter_index, item_index] = item_rank
        # Call function to get aggregated ranks
        rank = combmnz_agg(input_list)

        # Add results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    # Write results to the output CSV file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)
