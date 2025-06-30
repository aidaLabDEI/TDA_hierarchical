from typing import Union

import opendp as dp
import pandas as pd
from opendp.domains import atom_domain, map_domain
from opendp.metrics import l1_distance
from opendp.measurements import make_laplace_threshold
from opendp.mod import binary_search


def make_stability_histogram(d_in: float, budget: tuple[float, float], verbose: bool = False) -> dp.Measurement:
    """
    Make a stability histogram given a budget
    :param d_in: input sensitivity
    :param budget: privacy budget
    :param verbose: print threshold
    """

    input_space = map_domain(atom_domain(T=str), atom_domain(T=float)), l1_distance(T=float)

    def privatize(s, t=1e8):
        return make_laplace_threshold(*input_space, scale=s, threshold=t)

    s = binary_search(lambda s: privatize(s=s).map(d_in)[0] <= budget[0])
    t = binary_search(lambda t: privatize(s=s, t=t).map(d_in)[1] <= budget[1])
    if verbose: print(f"\nThreshold used: ", t, "\n")
    return privatize(s=s, t=t)


def get_dict(df: pd.Series, key_as_str: bool = True) -> dict[str, float]:
    """
        Returns a dictionary of the Series
        :param df: pd.Series
        :param key_as_str: type, type of the keys, int or str
    """
    # transform df into a hash map
    df_dict = df.to_dict()
    if key_as_str:
        # change keys from tuple[int, int] to a str
        df_dict = {str(k): v for k, v in df_dict.items()}
    return df_dict


def transform_key_type(x: dict[str, int]) -> dict[tuple[any], int]:
    """
    Transform the key of a dictionary from string to a tuple with appropriate types
    :param x: dictionary with string keys
    :return: dictionary with tuple keys where elements are cast to correct types (str, int, float)
    """

    def cast_value(val):
        # Remove surrounding quotes for strings and try casting to int/float
        val = val.strip("'\"")  # Strip single or double quotes
        if val.isdigit():  # Check if it's an integer
            return int(val)
        try:
            return float(val)  # Check if it's a float
        except ValueError:
            return val  # Return as string if it's not a number

    transformed_dict = {}

    for k, v in x.items():
        # Remove the parentheses and split the string
        key_elements = k.strip('()').split(', ')
        # Map each element to the appropriate type
        transformed_key = tuple(map(cast_value, key_elements))
        transformed_dict[transformed_key] = v

    return transformed_dict


def dict_to_dataset(data_dict, column_names=None):
    """
    Converts a dictionary with tuple or simple keys into a pandas DataFrame.

    Parameters:
    - data_dict (dict): Dictionary where keys are either tuples or single values.
                        If keys are tuples, their elements will be split into separate columns.
                        The values are numeric or categorical data.
    - column_names (list): Optional list of column names for the DataFrame. If not provided,
                           the function will generate generic names like 'Column_1', 'Column_2', etc.
                           followed by 'Value' for the last column.

    Returns:
    - pd.DataFrame: A pandas DataFrame with columns based on the keys and values.
    """
    # Determine if keys are tuples or single values
    first_key = next(iter(data_dict))

    # If the keys are tuples, determine how many elements are in each tuple
    if isinstance(first_key, tuple):
        num_columns = len(first_key)  # Number of elements in the tuple keys
    else:
        num_columns = 1  # If keys are not tuples, treat them as single values

    # Generate column names if not provided
    if column_names is None:
        column_names = [f'Column_{i + 1}' for i in range(num_columns)] + ['Value']

    # Convert the dictionary to a list of rows
    data_rows = [(list(key) if isinstance(key, tuple) else [key]) + [value] for key, value in data_dict.items()]

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)

    return df


def stability_histogram(data_sensitive: pd.DataFrame, sensitivity: Union[int, float],
                        budget: tuple[float, float], count_str: str) -> pd.DataFrame:
    """
    Return a differential private dataset using the Stability Histogram mechanism
    :param data_sensitive: the sensitive dataset
    :param sensitivity: the l1 sensitivity
    :param budget: (epsilon, delta)
    :param count_str: the column containing the counts
    :return: pd.DataFrame, dataset privatize with stability histogram
    """
    # remove counts equal to zero
    data_sensitive = data_sensitive[data_sensitive[count_str] > 0]

    # pre-process input, as currently, the mechanism works only with float
    sensitivity = float(sensitivity)

    # pre-process dataset
    attribute_columns = list(data_sensitive.columns[:-1])
    data_series = data_sensitive.set_index(attribute_columns)[count_str]
    data_series = data_series.astype(float)
    data_dict = get_dict(data_series)
    # create the mechanism
    dp_mechanism = make_stability_histogram(sensitivity, budget)

    # apply the mechanism
    dp_data_dict = dp_mechanism(data_dict)
    # transform the key type
    dp_data_dict = transform_key_type(dp_data_dict)
    # get the dataset
    data_dp = dict_to_dataset(dp_data_dict, column_names=data_sensitive.columns)
    # round the counts
    data_dp[count_str] = data_dp[count_str].round()
    # set the count_str_column to float
    data_dp = data_dp.astype({col: str for col in data_dp.columns[:-1]})
    data_dp = data_dp.astype({count_str: int})
    return data_dp
