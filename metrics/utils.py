import pandas as pd
import re


def match_index(data_true: pd.DataFrame, dp_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the index of the two dataframes adding zero counts where necessary
    :param data_true: pd.DataFrame, true data
    :param dp_data: pd.DataFrame, private data
    :return: tuple[pd.DataFrame, pd.DataFrame], ordered data
    """
    # get pandas series
    data_true = data_true.set_index(data_true.columns[:-1].tolist())
    dp_data = dp_data.set_index(dp_data.columns[:-1].tolist())
    # add zero counts
    index = data_true.index.union(dp_data.index)
    data_true = data_true.reindex(index, fill_value=0)
    dp_data = dp_data.reindex(index, fill_value=0)
    # return the dataset
    return data_true.reset_index(), dp_data.reset_index()


def get_levels_from_query(query: list[str]) -> tuple[int, int]:
    """
    Return the levels from the query, is the last element of the query
    :param query: list of columns
    :return: tuple of two levels
    """
    assert len(query) == 2, "The query must have two elements"
    levels = [int(re.findall(r'\d+', string.split("_")[0])[0]) for string in query]
    return tuple(levels)
