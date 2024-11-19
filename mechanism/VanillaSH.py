import pandas as pd
import numpy as np
import argparse
import time
import opendp as dp
from data_structure.tree import OD_tree
from data_structure.utils import get_dataset_from_dict
from differential_privacy import make_stability_histogram
import ast


def VanillaSH(Tree: OD_tree, args: argparse.Namespace) -> tuple[pd.DataFrame, float]:
    start: float = time.time()

    # define privacy budget
    epsilon: float = float(args.epsilon)
    delta: float = float(args.delta)
    budget: tuple[float, float] = (epsilon, delta)

    # define the final level of the tree to reach
    final_level: int = vars(args).get('final_level', Tree.depth)

    # l2 sensitivity
    max_contribution = vars(args).get('max_contribution', 1)
    sensitivity: float = np.sqrt(2 * max_contribution)

    # instantiate the mechanism
    dp_mechanism: dp.Measurement = make_stability_histogram(d_in=sensitivity, budget=budget)
    # get dataset (use full histogram as we need also the zero counts)
    print("Querying the full histogram")
    data: pd.Series = Tree.stable_query_level(args.final_level)
    data_dict: dict = pre_process(data=data)
    # apply the mechanism
    print("Applying the DP mechanism")
    dp_data_dict: dict = dp_mechanism(data_dict)
    # round the counts
    dp_data_dict = {key: round(value) for key, value in dp_data_dict.items() if round(value) > 0}
    # post process
    # dp_data_dict = transform_key_type(dp_data_dict)
    dp_data_dict = convert_keys_to_tuples(dp_data_dict)
    # create dataset
    print("Creating the dataset")
    # the final constraint is a dictionary containing (leaf node, leaf node) as key and the flow as value
    geo_level = int(final_level / 2)
    dp_dataset: pd.DataFrame = get_dataset_from_dict(data_dict=dp_data_dict,
                                                     spine=Tree.spine,
                                                     geo_level=geo_level)
    end: float = time.time()
    print(f"Time taken to create the dataset: {end - start:.2f} seconds")
    return dp_dataset


def pre_process(data: pd.Series) -> dict:
    """
    Post process the data to be handled by the SH mechanism
    :param data:
    :return:
    """
    data_dict: dict = data.to_dict()
    # change keys from tuple[int, int] to a str
    return {str(k): v for k, v in data_dict.items()}


def transform_key_type(x: dict[str, int]) -> dict[tuple[any], int]:
    """
    Transform the key of a dictionary from string to a tuple with appropriate types
    :param x: dictionary with string keys
    :return: dictionary with tuple keys where elements are cast to correct types (str, int, float)
    """

    def cast_value(val):
        # Remove surrounding quotes for strings and try casting to int/float
        val = val.strip("'\"")  # Strip single or double quotes
        if val.isdigit():  # Check if it's an integer
            return int(val)
        try:
            return float(val)  # Check if it's a float
        except ValueError:
            return val  # Return as string if it's not a number

    transformed_dict = {}

    for k, v in x.items():
        # Remove the parentheses and split the string
        key_elements = k.strip('()').split(', ')
        # Map each element to the appropriate type
        transformed_key = tuple(map(cast_value, key_elements))
        transformed_dict[transformed_key] = v

    return transformed_dict


def convert_keys_to_tuples(d):
    """
    Convert string keys of the form "('item1', 'item2')" to tuple keys ('item1', 'item2').

    :param d: Dictionary with stringified tuple keys
    :return: Dictionary with actual tuple keys
    """
    new_dict = {}
    for key, value in d.items():
        # Convert string key to tuple using ast.literal_eval
        tuple_key = ast.literal_eval(key)
        new_dict[tuple_key] = value
    return new_dict
