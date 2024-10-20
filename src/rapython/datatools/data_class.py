class SingleQueryMappingResults:
    """
    Encapsulates the mapping relationships for a single query in the dataset.
    This class provides mappings between different representations of voters, items, and queries.

    Attributes
    ----------
    int_to_item_map : dict
        A dictionary mapping integer indices to Item Codes. Maps each item's integer index to its actual Item Code.

    int_to_voter_map : dict
        A dictionary mapping integer indices to Voter Names. Maps each voter's integer index to their actual name.

    item_to_int_map : dict
        A dictionary mapping Item Codes to integer indices. Converts Item Codes into their corresponding integer indices.

    voter_to_int_map : dict
        A dictionary mapping Voter Names to integer indices. Converts Voter Names into their corresponding integer indices.

    input_lists : numpy.ndarray
        A 2D numpy array representing the input data for the query, where rows correspond to voters and columns to items.

    int_to_query_map : dict
        A dictionary mapping integer indices to Query names or identifiers. Maps each query's integer index to its actual identifier or name.
    """

    def __init__(self, int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists,
                 int_to_query_map):
        self.int_to_item_map = int_to_item_map
        self.int_to_voter_map = int_to_voter_map
        self.item_to_int_map = item_to_int_map
        self.voter_to_int_map = voter_to_int_map
        self.input_lists = input_lists
        self.int_to_query_map = int_to_query_map
