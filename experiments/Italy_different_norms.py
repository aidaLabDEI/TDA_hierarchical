import os
import sys
import pickle
import pandas as pd
import numpy as np
import tqdm as tqdm
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Add the 'Main' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics import analysis
from mechanism import GaussOpt
from data_structure import GeoSpine, OD_tree

"""
With this experiment, we compare the performance of the Gaussian mechanism with different norms and optimizer
on the Italy's dataset. We compare the L2, and Linf norms. The L2 norm is optimized using CLARABEL in the reals and then
is rounded. The L_inf norm is optimized using GLPK_MI and our IntOpt algorithm.
"""


def clear():
    # Clear the terminal
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For macOS and Linux
        os.system('clear')


def main(args: argparse.Namespace):
    def apply_mechanism(mech: callable, args: argparse.Namespace, num_mech: int, num_eps: int):
        absolute_error_distribution_epsilon = []
        for i in range(num_experiments):
            print(f"Experiment {i + 1}, mechanism {num_mech}, epsilon {args.epsilons[num_eps]}")
            start = time.time()
            data = mech(Tree, args)
            end = time.time()
            TIME[num_mech, num_eps, i] = end - start
            absolute_error_distribution = []
            for j, level in enumerate(tqdm.tqdm(levels, colour="green")):
                workload = [[f"LEVEL{level[0]}_ORIG", f"LEVEL{level[1]}_DEST"]]
                analysis_dict = analysis(data_true=data_true,
                                         dp_data=data,
                                         spine=Tree.spine,
                                         workload=workload)
                max_error[num_mech, e, i, j] = analysis_dict["max_absolute_error"]
                false_discovery_rate[num_mech, e, i, j] = analysis_dict["false_discovery_rate"]
                false_negative_rate[num_mech, e, i, j] = analysis_dict["false_negative_rate"]
                absolute_error_distribution.append(np.abs(analysis_dict["error_distribution"][0]))
            absolute_error_distribution_epsilon.append(absolute_error_distribution)
            # clear jupyter output on the terminal
            clear()
        # absolute_error_distribution_epsilon is a list of lists that needs to be concatenated
        error_to_add = np.zeros(len(levels))
        std_to_add = np.zeros(len(levels))
        for j in range(len(levels)):
            concantenation = []
            # concatenate the list of lists
            for i in range(num_experiments):
                concantenation.append(absolute_error_distribution_epsilon[i][j])
            # assert they are all the same length
            assert len(set([len(x) for x in concantenation])) == 1
            # flatten the list
            error_to_add[j] = np.mean(np.concatenate(concantenation))
            std_to_add[j] = np.std(np.concatenate(concantenation))
        MAE[num_mech, num_eps] = error_to_add
        std[num_mech, num_eps] = std_to_add

    # load the data
    folder_path: str = args.file_path
    with open(os.path.join(folder_path, "structure/geo_spine.pickle"), "rb") as f:
        geo_spine = pickle.load(f)
    df = pd.read_csv(os.path.join(folder_path, "data.csv"))
    spine = GeoSpine(geo_spine)
    Tree = OD_tree(df, spine)

    # get the data at the final level
    if args.final_level is None:
        args.final_level = Tree.depth
    data_true: pd.DataFrame = Tree.get_data_at_level(args.final_level)

    # get parameters
    epsilons: list = args.epsilons
    num_experiments: int = args.num_experiments
    # used to queries
    geo_level = int(args.final_level / 2)
    levels: list[tuple] = [(i, i) if i == j else (i, j) for i in range(geo_level + 1) for j in range(i, i + 2) if
                           j < geo_level + 1]
    # shape: mechanisms, epsilons, experiments, levels
    num_mechanisms = 3
    max_error = np.zeros((num_mechanisms, len(epsilons), num_experiments, len(levels)))
    false_discovery_rate = np.zeros((num_mechanisms, len(epsilons), num_experiments, len(levels)))
    false_negative_rate = np.zeros((num_mechanisms, len(epsilons), num_experiments, len(levels)))
    # shape: mechanism, epsilons, levels
    MAE = np.zeros((num_mechanisms, len(epsilons), len(levels)))
    std = np.zeros((num_mechanisms, len(epsilons), len(levels)))
    # shape: mechanism, epsilons, experiments
    TIME = np.zeros((num_mechanisms, len(epsilons), num_experiments))

    # RUN GAUSSOPT with L2 norm
    print("Running GaussOpt with L2 norm")
    num_mech = 0
    args.p = 2
    args.optimizer = "standard_int"
    for e, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        apply_mechanism(GaussOpt, args, num_mech, e)
        if len(epsilons) > 1: print("Running GaussOpt with L2 norm")

    # RUN GAUSSOPT with Linf norm (no IntOpt)
    print("Running GaussOpt with Linf norm")
    num_mech = 1
    args.p = "inf"
    args.optimizer = "standard_int"
    for e, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        apply_mechanism(GaussOpt, args, num_mech, e)
        if len(epsilons) > 1: print("Running GaussOpt with Linf norm")

    # RUN GAUSSOPT with Linf norm (IntOpt)
    print("Running GaussOpt with Linf norm and IntOpt")
    num_mech = 2
    args.p = "inf"
    args.optimizer = "fast_int_opt"
    for e, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        apply_mechanism(GaussOpt, args, num_mech, e)
        if len(epsilons) > 1: print("Running GaussOpt with Linf norm and IntOpt")

    # save TIME, MAE, etc...
    folder_path = args.save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # get name of the file
    today = datetime.now().strftime('%Y-%m-%d-%H-%M')
    counter = 1
    # Loop to find an available filename
    while os.path.exists(os.path.join(folder_path, f"results_{counter}.pickle")):
        counter += 1
    filename = f"results_{counter}.pickle"
    with open(os.path.join(folder_path, filename), "wb") as f:
        # save as a dictionary
        pickle.dump({"TIME": TIME,
                     "MAE": MAE,
                     "std": std,
                     "max_error": max_error,
                     "false_discovery_rate": false_discovery_rate,
                     "false_negative_rate": false_negative_rate,
                     "epsilons": epsilons,
                     "num_experiments": num_experiments,
                     "date": today}, f)


# Custom type function to parse a tuple of floats
def parse_float_tuple(value):
    try:
        return tuple(map(float, value.split(",")))  # Split by comma and convert to float
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {value}. Expected comma-separated floats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilons", type=parse_float_tuple, help="List of epsilons",
                        required=True)
    parser.add_argument("--delta", type=float, help="Delta", default=10 ** (-8))
    parser.add_argument("--num-experiments", type=int, help="Number of experiments", default=10)
    parser.add_argument("--show-tqdm", action="store_true", help="Show tqdm progress bar", default=True)
    parser.add_argument("--split-method", type=str, help="Split method", default="uniform")
    parser.add_argument("--final-level", type=int, help="Final level", default=None)
    parser.add_argument("--file-path", type=str, help="File path", default="../data/Italy/")
    parser.add_argument("--save-path", type=str, help="Save path", required=True)

    args = parser.parse_args()
    # clear the terminal
    clear()
    main(args)
