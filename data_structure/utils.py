from .geospine import GeoSpine
import pandas as pd


def get_dataset_from_dict(data_dict: dict,
                          spine: GeoSpine,
                          geo_level: int) -> pd.DataFrame:
    """
    WORKS ONLY FOR EVEN FINAL LEVEL

    Get a dataset from a dictionary
    :param data_dict: dict, dictionary with the data
    :param spine: GeoSpine, spine of the dataset
    :param geo_level: int, final level of the geo partition

    :return: pd.DataFrame, dataset
    """
    data_to_add = []
    # compute all the possible paths, so the set of the nodes in the first and second keys
    all_nodes = set()
    for nodes in data_dict.keys():
        all_nodes.add(nodes[0])
        all_nodes.add(nodes[1])
    path_dict = {}
    for node in all_nodes:
        path = spine.get_path(level=geo_level, node=node)
        path_dict[node] = path

    # get origin, destination and flow
    for nodes, value in data_dict.items():
        orig_node = nodes[0]
        dest_node = nodes[1]
        data_to_add.append(path_dict[orig_node] + path_dict[dest_node] + (value,))

    orig_column = tuple(["LEVEL" + str(level) + "_ORIG" for level in range(geo_level + 1)])
    dest_column = tuple(["LEVEL" + str(level) + "_DEST" for level in range(geo_level + 1)])
    dataset = pd.DataFrame(data_to_add, columns=orig_column + dest_column + ("COUNT",))
    return dataset
