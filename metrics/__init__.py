import pandas as pd
from .metrics import (false_discovery_rate, false_negative_rate, CPC, MAE, RMSE, L1, L2, max_absolute_error,
                      error_distribution, standard_deviation)
from .utils import match_index
from data_structure import GeoSpine

METRICS = [false_discovery_rate, false_negative_rate, CPC, MAE, RMSE, L1, L2, max_absolute_error,
           error_distribution, standard_deviation]


def analysis(data_true: pd.DataFrame,
             dp_data: pd.DataFrame,
             spine: GeoSpine,
             metrics_to_use: list = None,
             workload: list = None) -> dict:
    """
    Return all the metrics
    :param data_true: pd.DataFrame containing the true data in tabular format at the final level.
    :param dp_data: pd.DataFrame containing the differentially private data in tabular format at the final level.
    :param spine: GeoSpine object containing the spine of the tree.
    :param metrics_to_use: list of metrics to use.
    :param workload: list of workload to use, list of queries. (Our experiments use only one query)

    :return: A dictionary containing the metrics.
    """
    if metrics_to_use is None:
        metrics_to_use = METRICS
    output = {}
    # match the index, as the data_true and dp_data might have different indexes
    data_true, dp_data = match_index(data_true, dp_data)
    for metric in metrics_to_use:
        output[metric.__name__] = metric(data_true=data_true,
                                         dp_data=dp_data,
                                         workload=workload,
                                         spine=spine)
    return output
