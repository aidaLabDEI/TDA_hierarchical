import numpy as np
import pandas as pd
from .utils import get_levels_from_query
from data_structure import GeoSpine


def false_discovery_rate(data_true: pd.DataFrame, dp_data: pd.DataFrame, workload=None, spine=None) -> int:
    """
    Return the percentage (in the released data) of false positive in the released data
    :param data_true: sensitive data
    :param dp_data: private data
    :return: percentage of false positive
    """
    count_str = data_true.columns[-1]
    false_discovery_rate_list = []
    for query in workload:
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        # drop elements that are equal to zero or negative
        data_true = data_true[data_true > 0]
        dp_data = dp_data[dp_data > 0]
        # search indices of dp_data that are not in data_true
        false_positive_indices = np.logical_not(dp_data.index.isin(data_true.index))
        # return the percentage of false positive in the released data
        false_discovery_rate_list.append((np.sum(false_positive_indices) / len(dp_data)) * 100)
    output = np.mean(false_discovery_rate_list)
    return output


def false_negative_rate(data_true: pd.DataFrame, dp_data: pd.DataFrame, workload=None, spine=None) -> int:
    """
    Return the percentage (in the true data) of false negative that we missed
    :param data_true: sensitive data
    :param dp_data: private data
    :return: percentage of false negative
    """
    count_str = data_true.columns[-1]
    false_negative_rate_list = []
    for query in workload:
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        # drop elements that are equal to zero or negative
        data_true = data_true[data_true > 0]
        dp_data = dp_data[dp_data > 0]
        # search indices of data_true that are not in dp_data
        false_negative_indices = np.logical_not(data_true.index.isin(dp_data.index))
        # return the percentage of false negative that we missed
        false_negative_rate_list.append((np.sum(false_negative_indices) / len(data_true)) * 100)
    output = np.mean(false_negative_rate_list)
    return output


def CPC(data_true: pd.DataFrame, dp_data: pd.DataFrame, workload=None, spine=None) -> float:
    """
    Return the CPC (Common part of Commuter) of the released data
    :param data_true: sensitive data
    :param dp_data: private data
    """
    count_str = data_true.columns[-1]
    CPC_list = []
    for query in workload:
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        CPC_list.append(2 * (np.minimum(data_true, dp_data).sum()) / (
                data_true.sum() + dp_data.sum()))
    # calculate the MAE
    output = np.mean(CPC_list)
    return output


def MAE(data_true: pd.DataFrame, dp_data: pd.DataFrame, spine: GeoSpine, workload=None) -> float:
    """
    Return the MAE (Mean Absolute Error) of the released data
    :param spine:
    :param data_true: sensitive data
    :param dp_data: private data
    """
    count_str = data_true.columns[-1]
    MAE_list = []
    for query in workload:
        levels = get_levels_from_query(query)
        number_of_orig_nodes = len(spine.get_nodes(levels[0]))
        number_of_dest_nodes = len(spine.get_nodes(levels[1]))
        normalization = number_of_orig_nodes * number_of_dest_nodes
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        MAE_list.append(np.abs(data_true - dp_data).sum() * (1 / normalization))
    # calculate the MAE
    output = np.mean(MAE_list)
    return output


def max_absolute_error(data_true: pd.DataFrame, dp_data: pd.DataFrame, spine: GeoSpine, workload=None) -> float:
    """
    Return the max absolute error of the released data
    :param spine:
    :param data_true: sensitive data
    :param dp_data: private data
    :param workload: list of queries
    """
    count_str = data_true.columns[-1]
    MAE_list = []
    for query in workload:
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        MAE_list.append(np.abs(data_true - dp_data).max())
    # calculate the MAE
    output = np.mean(MAE_list)
    return output


def RMSE(data_true: pd.DataFrame, dp_data: pd.DataFrame, spine: GeoSpine, workload=None) -> float:
    """
    Return the RMSE (Root Mean Squared Error) of the released data
    :param spine:
    :param data_true: sensitive data
    :param dp_data: private data
    """
    count_str = data_true.columns[-1]
    RMSE_list = []
    for query in workload:
        levels = get_levels_from_query(query)
        number_of_orig_nodes = len(spine.get_nodes(levels[0]))
        number_of_dest_nodes = len(spine.get_nodes(levels[1]))
        normalization = number_of_orig_nodes * number_of_dest_nodes
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        RMSE_list.append(np.sqrt(np.pow(data_true - dp_data, 2).sum() * (1 / normalization)))
    # calculate the MAE
    output = np.mean(RMSE_list)
    return output


def L1(data_true: pd.DataFrame, dp_data: pd.DataFrame, spine: GeoSpine, workload=None) -> float:
    """
    Return the L1 of the released data
    :param spine:
    :param data_true: sensitive data
    :param dp_data: private data
    """
    count_str = data_true.columns[-1]
    L1_list = []
    for query in workload:
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        L1_list.append(np.abs(data_true - dp_data).sum())
    # calculate the MAE
    output = np.mean(L1_list)
    return output


def L2(data_true: pd.DataFrame, dp_data: pd.DataFrame, spine: GeoSpine, workload=None) -> float:
    """
    Return the L2 of the released data
    :param spine:
    :param data_true: sensitive data
    :param dp_data: private data
    """
    count_str = data_true.columns[-1]
    L2_list = []
    for query in workload:
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        L2_list.append(np.sqrt(np.pow(data_true - dp_data, 2).sum()))
    # calculate the MAE
    output = np.mean(L2_list)
    return output


def error_distribution(data_true: pd.DataFrame, dp_data: pd.DataFrame, spine: GeoSpine,
                       workload=None) -> float:
    """
    Return the distribution of the error of the released data, accounting also for the zero values not present in
    the true data
    :param spine:
    :param data_true: sensitive data
    :param dp_data: private data
    :param workload: list of queries
    """
    count_str = data_true.columns[-1]
    output = []
    for query in workload:
        # get the total number of data points
        levels = get_levels_from_query(query)
        number_of_orig_nodes = len(spine.get_nodes(levels[0]))
        number_of_dest_nodes = len(spine.get_nodes(levels[1]))
        total_data_points = number_of_orig_nodes * number_of_dest_nodes
        # get pandas series
        data_true = data_true.groupby(query)[count_str].sum()
        dp_data = dp_data.groupby(query)[count_str].sum()
        absolute_error_distribution_list = list(data_true - dp_data)
        # add zero counts
        zero_elements = total_data_points - len(absolute_error_distribution_list)
        if zero_elements > 0:
            absolute_error_distribution_list += [0] * zero_elements
        output.append(absolute_error_distribution_list)
    return output


def standard_deviation(data_true: pd.DataFrame, dp_data: pd.DataFrame, spine: GeoSpine, workload=None) -> float:
    """
    Return the standard deviation of the error of the released data
    :param spine:
    :param data_true: sensitive data
    :param dp_data: private data
    """
    # works for workload with only one query
    assert len(workload) == 1, "The workload must have only one query"
    # get pandas series
    error_list = error_distribution(data_true, dp_data, spine, workload)[0]
    output = np.std(error_list)
    return output
