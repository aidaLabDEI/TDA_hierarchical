"""
The algorithm creates a synthetic hierarchical dataset, with relative hierarchical spine
"""
import sys

sys.path.append('../')

import os
import argparse
import pandas as pd
import numpy as np
import pickle
import random
import yaml
from data_structure import GeoSpine
import datetime
from itertools import product


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-level', type=int, default=4,
                        help='the depth of the hierarchy (default: 4)')
    parser.add_argument('--max-branching', type=int, default=20,
                        help='the maximum number of children for each node (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for the random number generator (default: 42)')
    parser.add_argument('--zero-probability', type=float, default=0.995,
                        help='probability of having a zero value in the dataset (default: 0.8)')
    parser.add_argument('--max-flow', type=int, default=1e3,
                        help='maximum flow value (default: 1e3)')
    parser.add_argument('--scale', type=float, default=1,
                        help='scale for the power law distribution (default: 1)')
    parser.add_argument('--random-branching', action='store_true',
                        help='if True, the branching factor is randomly chosen between 2 and max_branching'
                             ' (default: False), else the branching factor is max_branching')
    parser.add_argument('--save-to', type=str, default='../data/synthetic', )
    return parser.parse_args()


def create_consecutive_keys_dict(depth: int, max_branching: int, seed: int):
    """
    Create a synthetic hierarchical spine with a different key number for each node

    :param depth: the depth of the hierarchy
    :param max_branching: the maximum number of children for each node
    :param seed: seed for the random number generator

    Returns: a nested dictionary

    """

    def create_dict(level, key_index, internal_seed):
        if level >= depth:
            return {}, key_index  # Return the updated key_index along with the base case

        random.seed(internal_seed)

        if args.random_branching:
            # uniform random branching factor between 2 and max_branching
            branching_factor = random.randint(2, max_branching)
        else:
            branching_factor = max_branching

        level_dict = {}
        for _ in range(branching_factor):
            # Pass the current key_index as a key and increment it for the next iteration
            new_dict, key_index = create_dict(level + 1, key_index, internal_seed)  # the seed is updated by one
            level_dict[key_index] = new_dict
            key_index += 1  # Increment key_index after adding to ensure uniqueness
            internal_seed += 1  # Increment seed to ensure randomness

        return level_dict, key_index

    # Start the recursive creation with initial key_index 0
    spine_dict, _ = create_dict(0, 0, seed)
    return spine_dict


def combine_tuples(list1: list[tuple], list2: list[tuple]) -> list[tuple]:
    # Use product to get all combinations and concatenate tuples
    return [t1 + t2 for t1, t2 in product(list1, list2)]


args = parse_arguments()
final_level = args.final_level
max_branching = args.max_branching
seed = args.seed
np.random.seed(seed)
zero_probability = args.zero_probability
max_flow = int(args.max_flow)
scale = args.scale

# create the geo_spine
print("Creating the geographical spine...")
geo_spine = create_consecutive_keys_dict(final_level, max_branching, seed)

# save the geo_spine
folder_path = args.save_to
folder_path_2 = f"{folder_path}/structure"
# create the folder if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
if not os.path.exists(folder_path_2):
    os.makedirs(folder_path_2)
file_name = f"{folder_path_2}/geo_spine.pickle"
with open(file_name, "wb") as f:
    pickle.dump(geo_spine, f)
print("Geographical spine saved successfully")

# create GeoSpine object
geo_spine = GeoSpine(geo_spine)

print("Creating the synthetic dataset...")
# get all possible root to leaf paths
all_paths = geo_spine.get_all_paths()
# all_combinations is the cartesian product of the number of paths
all_combinations = combine_tuples(all_paths, all_paths)
paths_to_sample = int(len(all_combinations) * (1 - zero_probability))
random.seed(seed)
orig_dest_paths = random.sample(all_combinations, paths_to_sample)

# sample the flows using a power-law distribution
random_flows = np.random.pareto(scale, paths_to_sample)
# Scale and shift the power-law samples to fit the range [1, max_flow]
# First, normalize the samples to the range [0, 1]
random_flows = random_flows / np.max(random_flows)
# Scale to [0, max_flow-1] and shift by 1 to ensure the range is [1, max_flow]
random_flows = 1 + random_flows * (max_flow - 1)
# Since we want integers
random_flows = np.floor(random_flows).astype(int)

# generate the dataset
orig_columns = [f"LEVEL{i}_ORIG" for i in range(final_level + 1)]
dest_columns = [f"LEVEL{i}_DEST" for i in range(final_level + 1)]
synthetic_df = pd.DataFrame(orig_dest_paths, columns=orig_columns + dest_columns)
synthetic_df["COUNT"] = random_flows

# save the dataset
file_name = f"{folder_path}/data.csv"
synthetic_df.to_csv(file_name, index=False)

# save orig_dest and layers_column
file_name = f"{folder_path_2}/orig_dest.pickle"
with open(file_name, "wb") as f:
    pickle.dump(["_ORIG", "_DEST"], f)

file_name = f"{folder_path_2}/layers_column.pickle"
with open(file_name, "wb") as f:
    pickle.dump([f"{i}" for i in range(final_level + 1)], f)

# save flow name
file_name = f"{folder_path_2}/flow_name.pickle"
with open(file_name, "wb") as f:
    pickle.dump("COUNT", f)

print("Synthetic dataset saved successfully")

# generate a yaml file containing the Namespace arguments, insert also the date and time
args_dict = vars(args)
args_dict["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_name = f"{folder_path}/args.yaml"
with open(file_name, "w") as f:
    yaml.dump(args_dict, f)
