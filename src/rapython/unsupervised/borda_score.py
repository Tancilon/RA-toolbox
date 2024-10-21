"""
Borda Score Algorithm

This module provides an implementation of the rank aggregation algorithm
using score rules. This implementation is based on the following reference:

Reference:
-----------
- Boehmer, N., Bredereck, R., & Peters, D. (2023, June). Rank aggregation using scoring rules. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 5, pp. 5515-5523).

Authors: 
    Shiwei Feng, Qi Deng
Date: 
    2024-7-25

Input Format:
-------------
The input to the algorithm should be in CSV file format with the following columns:
- Query: Does not require consecutive integers starting from 1.
- Voter Name: Allowed to be in string format.
- Item Code: Allowed to be in string format.
- Item Rank: Represents the rank given by each voter.

Output Format:
--------------
The final output of the algorithm will be in CSV file format with the following columns:
- Query: The same as the input.
- Item Code: The same as the input.
- Item Rank: The rank information (not the score information).
  - Note: The smaller the Item Rank, the higher the rank.
"""

from functools import cmp_to_key

import numpy as np

from src.rapython.datatools import *

__all__ = ['borda_score']

tie_breaking_order = None
tie = None


def borda(input_list):
    """
    Aggregate the Borda scores for the items provided by the voters.

    Parameters
    ----------
    input_list : numpy.ndarray
        A 2D array where each row represents a voter's scoring for different items.

    Returns
    -------
    numpy.ndarray
        An array containing the final scores of the items after aggregation.
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
        (order, tie_val):
            order is a list indicating the final order of candidates,
            tie_val is an integer indicating the number of ties encountered.
    """
    tie_val = 0
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
            tie_val += 1
        delete = None
        for t in tiebreaking:
            if t in cand_to_be_del:
                delete = t
                break
        order[i] = delete
        not_deleted.remove(delete)
        for k in range(len(votes)):
            if delete in votes[k]:
                votes[k].remove(delete)
        points = rule(votes, m)
    order[m - 1] = not_deleted[0]
    return order, tie_val


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
    tie_val = 0
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
            tie_val += 1
        delete = None
        for t in tiebreaking:
            if t in cand_to_be_del:
                delete = t
                break
        order[m - i - 1] = delete
        not_deleted.remove(delete)
        for k in range(len(votes)):
            if delete in votes[k]:
                votes[k].remove(delete)
        points = rule(votes, m)
    order[0] = not_deleted[0]
    return order, tie_val


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
    assert tie_breaking_order is not None, "tie_breaking_order must be initialized before comparison."

    # noinspection PyUnresolvedReferences
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
    df, unique_queries = csv_load(input_file_path)
    # Create an empty DataFrame to store the results
    result = []
    for query in unique_queries:
        query_data = df[df['Query'] == query]
        int_to_item_map, _, _, _, input_lists = wtf_map(query_data)

        full_input_lists, _ = partial_to_full(input_lists)

        # Call the function to get ranking information
        rank, _ = score_ordering(full_input_lists.shape[1], borda(full_input_lists).tolist(),
                                 list(np.random.permutation(full_input_lists.shape[1])))

        for i in range(len(rank)):
            item_code = int_to_item_map[rank[i]]
            item_rank = i + 1
            new_row = [query, item_code, item_rank]
            result.append(new_row)
    save_as_csv(output_file_path, result)
