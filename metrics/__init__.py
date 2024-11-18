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
    :param data_true:
    :param dp_data:
    :param spine:
    :param metrics_to_use:
    :param workload:
    :return:
    """
    if metrics_to_use is None:
        metrics_to_use = METRICS
    output = {}
    # match the index
    data_true, dp_data = match_index(data_true, dp_data)
    for metric in metrics_to_use:
        output[metric.__name__] = metric(data_true=data_true,
                                         dp_data=dp_data,
                                         workload=workload,
                                         spine=spine)
    return output
