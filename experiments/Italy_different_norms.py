import os
import sys
import pickle
import pandas as pd
import numpy as np
import tqdm as tqdm
import argparse
import time
import datetime
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Add the 'Main' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics import analysis
from mechanism import GaussOpt
from data_structure import GeoSpine, OD_tree


def main(args: argparse.Namespace):
    def apply_mechanism(mech: callable, args: argparse.Namespace, num_mech: int, num_eps: int):
        absolute_error_distribution_epsilon = []
        for i in range(num_experiments):
            print(f"Experiment {i + 1}")
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
                max_error_sparse[num_mech, e, i, j] = analysis_dict["max_absolute_error"]
                false_discovery_rate_sparse[num_mech, e, i, j] = analysis_dict["false_discovery_rate"]
                false_negative_rate_sparse[num_mech, e, i, j] = analysis_dict["false_negative_rate"]
                absolute_error_distribution.append(np.abs(analysis_dict["error_distribution"][0]))
            absolute_error_distribution_epsilon.append(absolute_error_distribution)
            # clear jupyter output on the terminal
            clear_output(wait=True)
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
    folder_path = "../data/Italy/"
    with open(os.path.join(folder_path, "structure/geo_spine.pickle"), "rb") as f:
        geo_spine = pickle.load(f)
    df = pd.read_csv(os.path.join(folder_path, "data.csv"))
    spine = GeoSpine(geo_spine)
    Tree = OD_tree(df, spine)

    # get parameters
    epsilons = args.epsilons
    num_experiments = args.num_experiments
    # used to queries
    final_level = Tree.depth
    geo_level = int(final_level / 2)
    levels: list[tuple] = [(i, i) if i == j else (i, j) for i in range(geo_level + 1) for j in range(i, i + 2) if
                           j < geo_level + 1]
    # shape: mechanisms, epsilons, experiments, levels
    num_mechanisms = 4
    max_error_sparse = np.zeros((num_mechanisms, len(epsilons), num_experiments, len(levels)))
    false_discovery_rate_sparse = np.zeros((num_mechanisms, len(epsilons), num_experiments, len(levels)))
    false_negative_rate_sparse = np.zeros((num_mechanisms, len(epsilons), num_experiments, len(levels)))
    # shape: mechanism, epsilons, levels
    MAE = np.zeros((num_mechanisms, len(epsilons), len(levels)))
    std = np.zeros((num_mechanisms, len(epsilons), len(levels)))
    # shape: mechanism, epsilons, experiments
    TIME = np.zeros((num_mechanisms, len(epsilons), num_experiments))

    # RUN GAUSSOPT with L1 norm
    print("Running GaussOpt with L1 norm")
    num_mech = 0
    args.p = 1
    args.optimizer = "int"
    for e, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        apply_mechanism(GaussOpt, args, num_mech, e)

    # RUN GAUSSOPT with L2 norm
    print("Running GaussOpt with L2 norm")
    num_mech = 1
    args.p = 2
    args.optimizer = "int"
    for e, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        apply_mechanism(GaussOpt, args, num_mech, e)

    # RUN GAUSSOPT with Linf norm (no IntOpt)
    print("Running GaussOpt with Linf norm")
    num_mech = 2
    args.p = np.inf
    args.optimizer = "int"
    for e, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        apply_mechanism(GaussOpt, args, num_mech, e)

    # RUN GAUSSOPT with Linf norm (IntOpt)
    print("Running GaussOpt with Linf norm and IntOpt")
    num_mech = 3
    args.p = np.inf
    args.optimizer = "int_opt"
    for e, epsilon in enumerate(epsilons):
        args.epsilon = epsilon
        apply_mechanism(GaussOpt, args, num_mech, e)

    # save TIME, MAE, etc...
    folder_path = "../results/Italy"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # get today's date and time
    today = datetime.today().strftime('%Y-%m-%d-%H-%M')
    with open(folder_path, "wb") as f:
        # save as a dictionary
        pickle.dump({"TIME": TIME,
                     "MAE": MAE,
                     "std": std,
                     "max_error_sparse": max_error_sparse,
                     "false_discovery_rate_sparse": false_discovery_rate_sparse,
                     "false_negative_rate_sparse": false_negative_rate_sparse,
                     "epsilons": epsilons,
                     "num_experiments": num_experiments,
                     "date": today}, f)

    # plot
    mechanisms = ["GaussOpt_p1", "GaussOpt_p2", "GaussOpt_pinf", "GaussOpt_pinf_IntOpt"]
    markers = [["-s", "-*", "-v", "-8", "-P"], ["--s", "--*", "--v", "--8", "--P"]]
    colors = ["blue", "orange", "red", "green"]

    def plot(array: np.array, y_log: bool, title: str, save_to: str, name: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, mechanism in enumerate(mechanisms):
            for j, epsilon in enumerate(epsilons):
                # plot error bar line for each level using min max
                error = np.array([np.mean(array[i, j], axis=0) - np.min(array[i, j], axis=0),
                                  np.max(array[i, j], axis=0) - np.mean(array[i, j], axis=0)])
                ax.errorbar(range(len(levels)), np.mean(array[i, j], axis=0), yerr=error,
                            label=mechanism + f" eps: {epsilon}",
                            fmt=markers[j][i], color=colors[i])
        if y_log:
            ax.set_yscale("log")
        # Setting labels and ticks
        ax.set_ylabel(title, fontsize=15)
        ax.set_xlabel("Levels", fontsize=15)  # Optional: Add an x-label for clarity
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=45, ha='right')  # Set oblique labels
        # augment font size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Display the plot
        plt.grid(True)
        plt.tight_layout()
        file_path = os.path.join(save_to, name) + "_nolegend.pdf"
        plt.savefig(file_path, dpi=300)
        # put the legend outside
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # save
        plt.tight_layout()
        file_path = os.path.join(save_to, name) + ".pdf"
        plt.savefig(file_path, dpi=300)

    data_true = Tree.get_data_at_level(Tree.depth)

    folder_path = "../plots/Italy"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # plot maximum error
    plot(max_error_sparse, y_log=True, title="Max Absolute Error", save_to=folder_path, name="max_error")
    # plot false discovery rate
    plot(false_discovery_rate_sparse, y_log=False, title="False Discovery Rate", save_to=folder_path,
         name="false_discovery_rate")
    # plot false negative rate
    plot(false_negative_rate_sparse, y_log=False, title="False Negative Rate", save_to=folder_path,
         name="false_negative_rate")

    # plot MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mechanism in enumerate(mechanisms):
        for j, epsilon in enumerate(epsilons):
            ax.errorbar(range(len(levels)), MAE[i, j], yerr=std[i, j],
                        label=mechanism + f" eps: {epsilon}",
                        fmt=markers[j][i], color=colors[i], capsize=5)
    # Setting labels and ticks
    ax.set_ylabel("Absolute Error", fontsize=15)
    ax.set_xlabel("Levels", fontsize=15)  # Optional: Add an x-label for clarity
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=45, ha='right')  # Set oblique labels
    # augment font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(folder_path, "mean_absolute_error") + "_nolegend.pdf"
    plt.savefig(path, dpi=300)
    # put the legend outside
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # save
    plt.tight_layout()
    path = os.path.join(folder_path, "mean_absolute_error.pdf")
    plt.savefig(path, dpi=300)


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

    args = parser.parse_args()
    main(args)
