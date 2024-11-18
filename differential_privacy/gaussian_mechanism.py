from typing import Union
import numpy as np
import opendp as dp
import pandas as pd
from opendp.domains import vector_domain, atom_domain
from opendp.metrics import l2_distance
from opendp.measurements import make_gaussian
from .utils import get_rho_from_budget


def make_gaussian_noise(d_in: Union[int, float],
                        budget: float,
                        dtype: type) -> dp.Measurement:
    """
    Return a Gaussian mechanism with rho (from zCDP)
    :param d_in: l2 sensitivity
    :param budget: privacy budget (rho in zCDP)
    :param dtype: input data type
    :return: mechanism
    """
    # assert dtype is int or float
    assert budget > 0, f"Invalid budget: {budget}, must be > 0"
    assert dtype in [int, float], f"Invalid dtype: {dtype}, must be int or float"
    if dtype == int:
        input_space = vector_domain(atom_domain(T=np.int64)), l2_distance(T=np.int64)
    else:
        input_space = vector_domain(atom_domain(T=float)), l2_distance(T=float)
    mechanism = make_gaussian(*input_space, scale=d_in / np.sqrt(2 * budget))
    return mechanism


def gaussian_mechanism(data_sensitive: pd.DataFrame,
                       sensitivity: Union[int, float],
                       budget: tuple[float, float],
                       count_str: str,
                       dtype: type = int) -> pd.DataFrame:
    """
    Return a differential private dataset using the Gaussian mechanism
    :param data_sensitive: the sensitive dataset (it has to contains also the rows with 0 values)
    :param sensitivity: l2 sensitivity
    :param budget: (epsilon, delta)
    :param dtype: data type
    :param count_str: the column containing the counts
    """
    if dtype == int:
        data_sensitive.astype({count_str: int})
    elif dtype == float:
        data_sensitive.astype({count_str: float})

    rho = get_rho_from_budget(budget)  # get rho privacy budget
    # pre-process the dataset
    data_dp: pd.DataFrame = data_sensitive[data_sensitive.columns[:-1]].copy()
    data_values = data_sensitive[count_str].values
    # create the mechanism
    dp_mechanism = make_gaussian_noise(sensitivity, rho, dtype)
    # apply the mechanism
    dp_data_values: list = dp_mechanism(data_values)
    # clamp to zero the negative values
    dp_data_values = np.maximum(dp_data_values, 0)
    # add the values to the dataset
    data_dp[count_str] = dp_data_values
    # post-process the dataset (types)
    data_dp = data_dp.astype({col: str for col in data_dp.columns[:-1]})
    data_dp = data_dp.astype({count_str: int})
    return data_dp
