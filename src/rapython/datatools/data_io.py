from multipledispatch import dispatch
import numpy as np
import pandas as pd
import os
import csv
import warnings

from src.rapython.datatools import InputType, wtf_map, SingleQueryMappingResults

__all__ = ['csv_load', 'save_as_csv', 'df_to_numpy']


def validate_csv_data(df):
    """
    Validate the CSV data for required columns and missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to validate.

    Raises
    ------
    ValueError
        If the DataFrame does not have the required data type.
    """

    # Optionally check for data types (e.g., 'Item Rank' should be numeric)
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 3]):
        raise ValueError("'Item Rank/Score' column must be numeric.")


@dispatch(str, InputType)
def csv_load(input_file_path, input_type):
    """
    Load a CSV file and process it with data validation.

    Parameters
    ----------
    input_file_path : str
        The path to the input CSV file.
    input_type : InputType
        The type of input data. It determines
        the naming of the fourth column, which will either be 'Item Rank'
        or 'Item Score' based on this value.

    Returns
    -------
    tuple
        A tuple containing:
            - df : pandas.DataFrame
                The loaded DataFrame with columns: 'Query', 'Voter Name',
                'Item Code', and either 'Item Rank' or 'Item Score', depending
                on the 'type' parameter.
            - unique_queries : numpy.ndarray
                An array of unique Query values from the DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the DataFrame cannot be loaded or does not have the required structure.
    """
    # Check if the file exists
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"The file {input_file_path} does not exist.")

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file_path, header=None)
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")

    if input_type == InputType.RANK:
        item_attribute = 'Item Rank'
    else:
        item_attribute = 'Item Score'
    # Set the column names
    df.columns = ['Query', 'Voter Name', 'Item Code', item_attribute]

    # Validate the CSV data
    validate_csv_data(df)

    # Get unique Query values
    unique_queries = df['Query'].unique()

    return df, unique_queries  # Return DataFrame and unique queries


@dispatch(str, str, InputType)
def csv_load(input_file_path, input_rel_path, input_type):
    """
    Load and process CSV files containing ranking or scoring data and relevance labels.

    Parameters:
    -----------
    input_file_path : str
        The file path to the CSV file containing ranking or scoring data for items.

    input_rel_path : str
        The file path to the CSV file containing relevance labels for the items.

    input_type : InputType
        An enumeration or type that specifies whether the input data contains 'Rank' or 'Score' information.
        - If input_type is `InputType.RANK`, the input data contains item rankings.
        - If input_type is not `InputType.RANK`, the input data contains item scores.

    Returns:
    --------
    tuple
        A tuple containing:
        - input_data (pd.DataFrame): DataFrame with columns ['Query', 'Voter Name', 'Item Code', 'Item Rank/Score'].
          Depending on `input_type`, the fourth column will be labeled 'Item Rank' or 'Item Score'.
        - input_rel_data (pd.DataFrame): DataFrame with columns ['Query', '0', 'Item Code', 'Relevance'] containing
          the relevance labels for the items.
        - unique_queries (numpy.ndarray): Array containing the unique query identifiers from the input data.
    """
    # Load ranking/scoring data and relevance data from CSV files
    input_data = pd.read_csv(input_file_path, header=None)
    input_rel_data = pd.read_csv(input_rel_path, header=None)

    # Determine whether input contains rankings or scores based on input_type
    if input_type == InputType.RANK:
        item_attribute = 'Item Rank'
    else:
        item_attribute = 'Item Score'

    # Assign column names to the input data based on the item attribute (rank or score)
    input_data.columns = ['Query', 'Voter Name', 'Item Code', item_attribute]

    # Assign column names to the relevance data
    input_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

    # Extract the unique queries from the input data
    unique_queries = input_data['Query'].unique()

    # Return the input data, relevance data, and unique query list
    return input_data, input_rel_data, unique_queries


@dispatch(str, list)
def save_as_csv(output_file_path, result):
    """
    save_as_csv_str_list:
    Save the given result data to a CSV file.

    Parameters
    ----------
    output_file_path : str
        The path where the output CSV file will be saved.
    result : list of lists
        The data to be written to the CSV file, where each inner list represents a row.

    Raises
    ------
    IOError
        If there is an issue writing to the file.
    """
    try:
        with open(output_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in result:
                writer.writerow(row)
    except IOError as e:
        raise IOError(f"Error writing to the file {output_file_path}: {e}")


@dispatch(str, np.ndarray, dict)
def save_as_csv(output_file_path, result, queries_mapping_dict):
    """
    save_as_csv_str_ndarray:
    Saves a 2D numpy array of results to a CSV file.

    This function writes query and item data, along with the results, to a CSV file.
    It uses a mapping dictionary to convert integer indices to their corresponding
    query and item values.

    Parameters
    ----------
    output_file_path : str
        The path to the output CSV file where the data will be saved.

    result : numpy.ndarray
        A 2D numpy array containing the ranking results. The rows correspond
        to different queries, and the columns correspond to different items.

    queries_mapping_dict : dict
        A dictionary mapping query indices to `SingleQueryMappingResults` objects.

    Raises
    ------
    IOError
        If there is an issue writing to the specified file, an IOError is raised
        with a descriptive message.

    """
    try:
        with open(output_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for query_index in range(result.shape[0]):
                for item_index in range(result.shape[1]):
                    new_row = [
                        queries_mapping_dict[query_index].int_to_query_map[query_index],
                        queries_mapping_dict[query_index].int_to_item_map[item_index],
                        result[query_index, item_index]
                    ]
                    writer.writerow(new_row)
    except IOError as e:
        raise IOError(f"Error writing to the file {output_file_path}: {e}")


def ranktoscore(df):
    """
    Converts item rankings into scores for each query and voter.

    This function transforms the ranks of items into scores based on the maximum
    rank within each query and voter group. The score is calculated as the difference
    between the maximum rank and the item's rank, plus one.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'Query': Identifier for the query.
        - 'Voter Name': Identifier for the voter.
        - 'Item Code': Identifier for the item.
        - 'Item Rank': Rank of the item (lower rank means higher priority).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'Query': The query identifier.
        - 'Voter Name': The voter identifier.
        - 'Item Code': The item identifier.
        - 'Item Score': The score calculated from the rank.
    """
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    # Group by 'Query' and 'Voter Name' and get the max rank for each group
    grouped = df.groupby(['Query', 'Voter Name'])['Item Rank'].transform('max')

    # Calculate the score for each item based on its rank
    df['Item Score'] = grouped - df['Item Rank'] + 1
    result_df = df[['Query', 'Voter Name', 'Item Code', 'Item Score']]

    return result_df


def scoretorank(df):
    """
    Converts item scores back into rankings for each query and voter.

    This function takes the scores of items and converts them into ranks. The items
    with higher scores will receive a lower (better) rank.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'Query': Identifier for the query.
        - 'Voter Name': Identifier for the voter.
        - 'Item Code': Identifier for the item.
        - 'Item Score': Score of the item (higher score means higher priority).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'Query': The query identifier.
        - 'Voter Name': The voter identifier.
        - 'Item Code': The item identifier.
        - 'Item Rank': The rank calculated from the score (lower rank means higher priority).
    """
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Score']

    # Convert scores to ranks within each 'Query' and 'Voter Name' group
    df['Item Rank'] = df.groupby(['Query', 'Voter Name'])['Item Score'].rank(method='dense', ascending=False)

    result_df = df[['Query', 'Voter Name', 'Item Code', 'Item Rank']]
    return result_df


def review_to_numpy(df):
    """
    Reviews the DataFrame to check for consistency in the item sets across queries and voters.

    Logic
    -----
    1. Groups the DataFrame by 'Query' and checks if all 'Item Code' sets within each group are identical.
    2. For each 'Query' group, further groups by 'Voter Name' and checks if the 'Item Code' sets are identical
       within each 'Voter Name' group.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to review, with columns 'Query', 'Voter Name', and 'Item Code'.

    Returns
    -------
    bool
        Returns True if the item sets are consistent across queries and voters.
        Returns False if there are any discrepancies in the 'Item Code' sets.
    """

    # Group by 'Query' and check for consistency in 'Item Code' sets across each query
    query_groups = df.groupby('Query')['Item Code'].apply(set)
    if not query_groups.apply(lambda x: x == query_groups.iloc[0]).all():
        return False

    # For each query group, group by 'Voter Name' and check consistency in 'Item Code' sets within each voter
    for query, query_data in df.groupby('Query'):
        voter_groups = query_data.groupby('Voter Name')['Item Code'].apply(set)
        if not voter_groups.apply(lambda x: x == voter_groups.iloc[0]).all():
            return False

    return True


@dispatch(pd.DataFrame, InputType)
def df_to_numpy(df, input_type):
    """
    Converts a DataFrame with columns Query, Voter Name, Item Code, and Item Rank/Score
    into a 3D NumPy array format, where the array represents the data as
    [Voter Name, Query, Item Code] = Item Score.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'Query': The query identifier.
        - 'Voter Name': The voter identifier.
        - 'Item Code': The item identifier.
        - 'Item Rank/Score': The rank or score of the item.

    input_type : InputType
        An enum or variable that specifies whether the 'Item Rank' should be
        converted to 'Item Score' (typically InputType.RANK) or left as is.

    Returns
    -------
    numpy.ndarray
        A 3D NumPy array with dimensions [Voter Name, Query, Item Code], where each
        value represents the item's score for the corresponding voter and query.

    dict
        A dictionary mapping query indices to `SingleQueryMappingResults`, which
        contains mapping details for each query, such as mappings of voters and items
        to their respective numerical representations.

    Notes
    -----
    - The input DataFrame is expected to be in full ranking or scoring format. If the
      dataset does not meet this expectation, a warning is issued.
    - If `input_type` is InputType.RANK, the ranks are converted to scores before
      processing.

    Warnings
    --------
    A warning is raised if the dataset might have issues during conversion, such as
    differing item sets across queries or incomplete rankings.
    """

    # Check if the dataset can be converted properly
    if not review_to_numpy(df):
        warnings.warn(
            "There might be issues with the dataset conversion. The item sets differ across queries or the dataset is not in full ranking format."
        )

    # Convert rank to score if the input type is RANK
    if input_type == InputType.RANK:
        df = ranktoscore(df)

    # Extract unique identifiers for queries, items, and voters
    unique_queries = df['Query'].unique()
    unique_items = df['Item Code'].unique()
    unique_voters = df['Voter Name'].unique()

    # Initialize a dictionary to store query mapping data
    queries_mapping_dict = {}

    # Create a 3D NumPy array to store scores, initialized with NaN values
    numpy_data = np.full((len(unique_voters), len(unique_queries), len(unique_items)), np.nan)

    # Loop through each unique query and process the data
    for index, query in enumerate(unique_queries):
        query_data = df[df['Query'] == query]
        mapping_data = SingleQueryMappingResults(*wtf_map(query_data), {index: query})

        # Store the mapping data in the dictionary
        queries_mapping_dict[index] = mapping_data

        # Assign the input lists (scores) to the NumPy array
        numpy_data[:, index, :] = mapping_data.input_lists

    return numpy_data, queries_mapping_dict


@dispatch(pd.DataFrame, pd.DataFrame, InputType)
def df_to_numpy(input_data, input_rel_data, input_type):
    if not review_to_numpy(input_data):
        warnings.warn(
            "There might be issues with the dataset conversion. The item sets differ across queries or the dataset is not in full ranking format."
        )
    # Convert rank to score if the input type is RANK
    if input_type == InputType.RANK:
        input_data = ranktoscore(input_data)
    # Extract unique identifiers for queries, items, and voters

    unique_queries = input_data['Query'].unique()
    unique_items = input_data['Item Code'].unique()
    unique_voters = input_data['Voter Name'].unique()

    # Initialize a dictionary to store query mapping data
    queries_mapping_dict = {}

    # Create a 3D NumPy array to store scores, initialized with NaN values
    numpy_data = np.full((len(unique_voters), len(unique_queries), len(unique_items)), np.nan)
    numpy_rel_data = np.full((len(unique_queries), len(unique_items)), np.nan)
    # Loop through each unique query and process the data
    for index, query in enumerate(unique_queries):
        query_data = input_data[input_data['Query'] == query]
        rel_data = input_rel_data[input_rel_data['Query'] == query]
        mapping_data = SingleQueryMappingResults(*wtf_map(query_data), {index: query})

        # Store the mapping data in the dictionary
        queries_mapping_dict[index] = mapping_data

        # Assign the input lists (scores) to the NumPy array
        numpy_data[:, index, :] = mapping_data.input_lists
        for _, row in rel_data.iterrows():
            item_code = row['Item Code']
            item_relevance = row['Relevance']

            item_index = mapping_data.item_to_int_map[item_code]
            numpy_rel_data[index, item_index] = item_relevance  # Assign relevance value.

    return numpy_data, numpy_rel_data, queries_mapping_dict
