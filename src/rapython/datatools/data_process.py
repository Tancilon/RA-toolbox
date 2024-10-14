import numpy as np
import pandas as pd

__all__ = ['partial_to_full', 'wtf_map']


def partial_to_full(input_list):
    """
    Convert partial ranking lists of each voter into a complete ranking matrix.

    Parameters
    ----------
    input_list : numpy.ndarray
        A 2D array where each row contains a voter's partial ranking of items,
        with np.nan indicating unranked items.

    Returns
    -------
    tuple
        A tuple containing:

        - input_list : numpy.ndarray
            The updated input_list with np.nan values replaced by the next highest rank.
        - list_maxrankofvoters : numpy.ndarray
            An array indicating the number of ranked items for each voter.
    """
    if not isinstance(input_list, np.ndarray):
        raise TypeError("input_list must be a numpy.ndarray")

    num_voters = input_list.shape[0]
    list_maxrankofvoters = np.zeros(num_voters)

    for k in range(num_voters):
        max_rank = np.nanmax(input_list[k])
        list_maxrankofvoters[k] = max_rank
        input_list[k] = np.nan_to_num(input_list[k], nan=max_rank + 1)

    return input_list, list_maxrankofvoters


def wtf_map(query_data):
    """
    Process query data to create mappings between unique identifiers and integer indices.

    Parameters
    ----------
    query_data : pandas.DataFrame
        DataFrame containing 'Item Code', 'Voter Name', and 'Item Rank' columns.

    Returns
    -------
    tuple
        A tuple containing:
        - int_to_item_map : dict
            Mapping from integer indices to item codes.
        - int_to_voter_map : dict
            Mapping from integer indices to voter names.
        - item_to_int_map : dict
            Mapping from item codes to integer indices.
        - voter_to_int_map : dict
            Mapping from voter names to integer indices.
        - input_lists : numpy.ndarray
            A 2D array representing rankings, with rows corresponding to voters and columns to items.
    """
    if not isinstance(query_data, pd.DataFrame):
        raise TypeError(f"Expected query_data to be a pandas DataFrame, but got {type(query_data).__name__}")

    unique_item_codes = query_data['Item Code'].unique()
    unique_voter_names = query_data['Voter Name'].unique()

    int_to_item_map = {i: code for i, code in enumerate(unique_item_codes)}
    int_to_voter_map = {i: name for i, name in enumerate(unique_voter_names)}

    item_to_int_map = {v: k for k, v in int_to_item_map.items()}
    voter_to_int_map = {v: k for k, v in int_to_voter_map.items()}

    num_voters = len(unique_voter_names)
    num_items = len(unique_item_codes)
    input_lists = np.full((num_voters, num_items), np.nan)

    # Fill the array with ranking data
    for index, row in query_data.iterrows():
        voter_name = row['Voter Name']
        item_code = row['Item Code']
        item_rank = row['Item Rank']

        voter_index = voter_to_int_map[voter_name]
        item_index = item_to_int_map[item_code]

        input_lists[voter_index, item_index] = item_rank

    return int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists
