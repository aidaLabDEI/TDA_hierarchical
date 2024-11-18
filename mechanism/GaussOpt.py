from collections.abc import ItemsView

import pandas as pd
import numpy as np
import argparse
import time
import tqdm
import opendp as dp
from .utils import split_rho_budget
from data_structure.tree import OD_tree
from data_structure.utils import get_dataset_from_dict
from optimization import check_constraint, int_opt, optimize_int_vector
from differential_privacy import make_gaussian_noise, get_rho_from_budget


def GaussOpt(Tree: OD_tree, args: argparse.Namespace) -> OD_tree:
    """
    This function implements the Gaussian mechanism with optimization. The Gaussian mechanism is implemented in the
    discrete domain, as the optimization step.
    :param args: argparse.Namespace containing input arguments.
    :param Tree: OD_tree object containing a tree structure.
    :return: differential private tree
    """

    start: float = time.time()

    # define privacy budget
    epsilon: float = float(args.epsilon)
    delta: float = float(args.delta)
    budget: tuple[float, float] = (epsilon, delta)
    rho: float = get_rho_from_budget(budget)

    # define the final level of the tree to reach
    final_level: int = vars(args).get('final_level', Tree.depth)

    # split the budget uniformly among the levels
    b = vars(args).get('b', None)  # feature to implement
    budget_list: list[float] = split_rho_budget(rho=rho,
                                                T=final_level * 2,
                                                method=args.split_method,
                                                b=b)

    # l2 sensitivity
    max_contribution = vars(args).get('max_contribution', 1)
    sensitivity: float = np.sqrt(2 * max_contribution)

    # get optimizer
    if args.optimizer == "int_opt":
        optimizer = int_opt
    elif args.optimizer == "int":
        optimizer = lambda x, y: optimize_int_vector(x, y, p=args.p)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    # get total number of users
    n: float = Tree.full_query_level(level=0).values[0]
    """ HERE to add for privacy by addition/removal, n needs to be DP"""
    c: dict = {(0, 0): int(n)}  # initial constraint
    count: int = 0  # initialize the counter
    for level in range(1, final_level + 1):
        print(f"Optimizing level {level}/{final_level}...")

        # dictionary to store the optimized constraint
        V_ell: dict = {}

        # instantiate the guassian mechanism
        dp_mechanism: dp.Measurement = make_gaussian_noise(d_in=sensitivity,
                                                           budget=budget_list[count],
                                                           dtype=int)

        # create the iterable
        if args.show_tqdm:
            constraint_items: ItemsView = tqdm.tqdm(c.items(), colour='green')
        else:
            constraint_items: ItemsView = c.items()

        for nodes, constraint in constraint_items:
            # Apply differential privacy mechanism
            q_c: pd.Series = Tree.full_child_query_level(level=level-1, nodes=nodes)
            dp_q_c_values: np.array = np.array(dp_mechanism(q_c.values)).astype(int)
            # Apply optimization
            bar_q_c_values: np.array = optimizer(dp_q_c_values, constraint)
            # Post process the data, remove zero values
            bar_q_c: dict = {key: int(value) for key, value in zip(q_c.index, bar_q_c_values) if value > 0}
            # Update the constraints
            V_ell.update(bar_q_c)

        # update the constraint
        c = V_ell
        count += 1

    # the final constraint is a dictionary containing (leaf node, leaf node) as key and the flow as value
    geo_level = int(final_level/2)
    dp_dataset: pd.DataFrame = get_dataset_from_dict(data_dict=c,
                                                     spine=Tree.spine,
                                                     geo_level=geo_level)

    end: float = time.time()
    print(f"Time taken to create the dataset: {end - start:.2f} seconds")
    print("Done!")
    return dp_dataset
