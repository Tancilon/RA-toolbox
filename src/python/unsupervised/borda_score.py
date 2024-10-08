"""
Reference: Rank Aggregation using Score Rule
fsw, Tancilon：20240725
Define the input to the algorithm as a csv file format: Query | Voter name | Item Code | Item Rank
     - Query does not require consecutive integers starting from 1.
     - Voter name and Item Code are allowed to be in String format.
Define the final output of the algorithm as a csv file format： Query | Item Code | Item Rank
     - Output is the rank information, not the score information
The smaller the Item Rank, the higher the rank.
"""
import csv
from functools import cmp_to_key

import numpy as np
import pandas as pd


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
        - list_numofitems : numpy.ndarray
            An array indicating the number of ranked items for each voter.
    """
    num_voters = input_list.shape[0]
    list_numofitems = np.zeros(num_voters)

    for k in range(num_voters):
        max_rank = np.nanmax(input_list[k])
        list_numofitems[k] = max_rank
        input_list[k] = np.nan_to_num(input_list[k], nan=max_rank + 1)

    return input_list, list_numofitems


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
    unique_item_codes = query_data['Item Code'].unique()
    unique_voter_names = query_data['Voter Name'].unique()

    int_to_item_map = {i: code for i, code in enumerate(unique_item_codes)}
    int_to_voter_map = {i: name for i, name in enumerate(unique_voter_names)}

    item_to_int_map = {v: k for k, v in int_to_item_map.items()}
    voter_to_int_map = {v: k for k, v in int_to_voter_map.items()}

    num_voters = len(unique_voter_names)
    num_items = len(unique_item_codes)
    input_lists = np.full((num_voters, num_items), np.nan)

    for index, row in query_data.iterrows():
        voter_name = row['Voter Name']
        item_code = row['Item Code']
        item_rank = row['Item Rank']

        voter_index = voter_to_int_map[voter_name]
        item_index = item_to_int_map[item_code]

        input_lists[voter_index, item_index] = item_rank

    return int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists


def borda(input_list):
    """
    Aggregate Borda scores for items based on rankings provided by voters.

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
    item_borda_score = np.zeros(num_items)
    item_score = np.zeros((num_voters, num_items))

    for k in range(num_voters):
        for i in range(num_items):
            item_score[k, i] = num_items - input_list[k, i] + 1
            item_borda_score[i] += item_score[k, i]

    return item_borda_score


def eliminate_top(vot, m, rule, tiebreaking):
    """
    Eliminate top-ranked candidates iteratively until all are placed according to their rank.

    Parameters
    ----------
    vot : list of lists
        Each inner list represents a voter's preference order.
    m : int
        The number of candidates.
    rule : function
        A scoring rule function to calculate candidate scores.
    tiebreaking : list
        A list representing the tie-breaking order.

    Returns
    -------
    tuple
        (order, tie):
            order is a list indicating the final order of candidates,
            tie is an integer indicating the number of ties encountered.
    """
    tie = 0
    tiebreaking = list(reversed(tiebreaking))
    votes = []
    for v in vot:
        vvv = []
        for c in v:
            vvv.append(c)
        votes.append(vvv)
    not_deleted = list(range(m))
    order = [0] * m
    points = rule(vot, m)

    for i in range(m - 1):
        max_relevant = max([points[i] for i in not_deleted])
        cand_to_be_del = [i for i in not_deleted if points[i] == max_relevant]
        if len(cand_to_be_del) > 1:
            tie += 1
        for t in tiebreaking:
            if t in cand_to_be_del:
                delete = t
                break
        order[i] = delete
        not_deleted.remove(delete)
        for i in range(len(votes)):
            if delete in votes[i]:
                votes[i].remove(delete)
        points = rule(votes, m)
    order[m - 1] = not_deleted[0]
    return order, tie


def eliminate_bottom(vot, m, rule, tiebreaking):
    """
    Determine the final ranking order by iteratively eliminating the candidate with the lowest score.

    Parameters
    ----------
    vot : list
        A list where each element represents a voter's preference ranking.
    m : int
        The number of candidates.
    rule : function
        A function that computes the scores for the current ranking.
    tiebreaking : list
        A predefined order for breaking ties among candidates.

    Returns
    -------
    tuple
        A tuple containing the final ranking order and the number of ties encountered.
    """
    tie = 0
    votes = []
    for v in vot:
        vvv = []
        for c in v:
            vvv.append(c)
        votes.append(vvv)

    not_deleted = list(range(m))
    order = [0] * m
    points = rule(vot, m)
    print(points)
    for i in range(m - 1):
        min_relevant = min([points[i] for i in not_deleted])
        cand_to_be_del = [i for i in not_deleted if points[i] == min_relevant]
        if len(cand_to_be_del) > 1:
            tie += 1
        for t in tiebreaking:
            if t in cand_to_be_del:
                delete = t
                break
        order[m - i - 1] = delete
        not_deleted.remove(delete)
        for i in range(len(votes)):
            if delete in votes[i]:
                votes[i].remove(delete)
        points = rule(votes, m)
    order[0] = not_deleted[0]
    return order, tie


tie_breaking_order = None
tie = None


def compare(item1, item2):
    """
    Compare two items based on their scores and tie-breaking order.

    Parameters
    ----------
    item1 : tuple
        A tuple representing an item and its score.
    item2 : tuple
        A tuple representing another item and its score.

    Returns
    -------
    int
        1 if item1 should be ranked higher than item2, -1 otherwise.
    """
    if item1[0] > item2[0]:
        return 1
    elif item1[0] < item2[0]:
        return -1
    elif tie_breaking_order.index(item1[1]) < tie_breaking_order.index(item2[1]):
        global tie
        tie += 1
        return 1
    else:
        return -1


def score_ordering(m, points, tiebreaking):
    """
    Compute the final ranking order based on the scores and tie-breaking order.

    Parameters
    ----------
    m : int
        The number of items.
    points : list
        A list of scores for each item.
    tiebreaking : list
        A predefined order for breaking ties among items.

    Returns
    -------
    tuple
        A tuple containing the final ranking order and the number of ties encountered.
    """
    global tie
    tie = 0
    global tie_breaking_order
    tie_breaking_order = tiebreaking
    inversed_points = [-x for x in points]
    to_be_sorted = list(zip(inversed_points, list(range(m))))
    return [x for _, x in sorted(to_be_sorted, key=cmp_to_key(compare))], tie


def borda_score(input_file_path, output_file_path):
    """
    Calculate the Borda scores for items based on rankings provided by voters and write the results to a CSV file.

    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file containing voting data.
    output_file_path : str
        Path to the output CSV file where the results will be written.
    """
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    # Get unique Query values
    unique_queries = df['Query'].unique()

    # Create an empty DataFrame to store the results
    result = []
    for query in unique_queries:
        query_data = df[df['Query'] == query]
        int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists = wtf_map(query_data)

        full_input_lists, list_numofitems = partial_to_full(input_lists)

        # Call the function to get ranking information
        rank, tie = score_ordering(full_input_lists.shape[1], borda(full_input_lists),
                                   list(np.random.permutation(full_input_lists.shape[1])))
        # Add the results to result_df
        for i in range(len(rank)):
            item_code = int_to_item_map[rank[i]]
            item_rank = i + 1
            new_row = [query, item_code, item_rank]
            result.append(new_row)
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)
