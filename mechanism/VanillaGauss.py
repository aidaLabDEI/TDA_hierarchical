import pandas as pd
import numpy as np
import argparse
import time
import opendp as dp
from data_structure.tree import OD_tree
from data_structure.utils import get_dataset_from_dict
from differential_privacy import make_gaussian_noise
from differential_privacy import get_rho_from_budget


def VanillaGauss(Tree: OD_tree, args: argparse.Namespace) -> pd.DataFrame:
    start: float = time.time()

    # define privacy budget
    epsilon: float = float(args.epsilon)
    delta: float = float(args.delta)
    budget: tuple[float, float] = (epsilon, delta)
    rho: float = get_rho_from_budget(budget)

    # define the final level of the tree to reach
    final_level: int = vars(args).get('final_level', Tree.depth)

    # l2 sensitivity
    max_contribution = vars(args).get('max_contribution', 1)
    sensitivity: float = np.sqrt(2 * max_contribution)

    # instantiate the mechanism
    dp_mechanism: dp.Measurement = make_gaussian_noise(d_in=sensitivity, budget=rho, dtype=int)
    # get dataset (use full histogram as we need also the zero counts)
    print("Querying the full histogram")
    data: pd.Series = Tree.full_query_level(level=final_level)
    # apply the mechanism
    print("Applying the DP mechanism")
    dp_data_values = dp_mechanism(data.values)
    # post process (get a dictionary) and remove zero counts but keep negative values
    dp_data_dict: dict = {key: value for key, value in zip(data.index, dp_data_values) if value != 0}
    # create dataset
    print("Creating the dataset")
    # the final constraint is a dictionary containing (leaf node, leaf node) as key and the flow as value
    geo_level = int(final_level/2)
    dp_dataset: pd.DataFrame = get_dataset_from_dict(data_dict=dp_data_dict,
                                                     spine=Tree.spine,
                                                     geo_level=geo_level)
    end: float = time.time()
    print(f"Time took to create the dataset: {end - start:.2f} seconds")
    return dp_dataset

