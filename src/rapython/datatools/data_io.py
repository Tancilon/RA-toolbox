import pandas as pd
import os
import csv

__all__ = ['csv_load', 'save_as_csv']


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
    if not pd.api.types.is_numeric_dtype(df['Item Rank']):
        raise ValueError("'Item Rank' column must be numeric.")


def csv_load(input_file_path):
    """
    Load a CSV file and process it with data validation.

    Parameters
    ----------
    input_file_path : str
        The path to the input CSV file.

    Returns
    -------
    tuple
        A tuple containing:
            - df : pandas.DataFrame
                The loaded DataFrame with specified columns.
            - unique_queries : numpy.ndarray
                An array of unique Query values from the DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the DataFrame does not have the required data type.
    """
    # Check if the file exists
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"The file {input_file_path} does not exist.")

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file_path, header=None)
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")

    # Set the column names
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    # Validate the CSV data
    validate_csv_data(df)

    # Get unique Query values
    unique_queries = df['Query'].unique()

    return df, unique_queries  # Return DataFrame and unique queries


def save_as_csv(output_file_path, result):
    """
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


def ranktoscore(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter', 'Item', 'Rank', 'Algorithm']

    grouped = df.groupby(['Query', 'Voter'])['Rank'].transform('max')

    df['Score'] = grouped - df['Rank'] + 1
    result_df = df[['Query', 'Voter', 'Item', 'Score', 'Algorithm']]

    result_df.to_csv(output_file_path, index=False, header=False)


def scoretorank(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter', 'Item', 'Score', 'Algorithm']

    df['Rank'] = df.groupby(['Query', 'Voter'])['Score'].rank(method='dense', ascending=False)

    df.to_csv(output_file_path, index=False, columns=['Query', 'Voter', 'Item', 'Rank'], header=False)
