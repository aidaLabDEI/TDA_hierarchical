import pandas as pd
import re


def match_index(data_true: pd.DataFrame, dp_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the index of the two dataframes adding zero counts where necessary
    :param data_true: pd.DataFrame, true data
    :param dp_data: pd.DataFrame, private data
    :return: tuple[pd.DataFrame, pd.DataFrame], ordered data

    First transform the dataframes into pandas series, then reindex them on the union of the indexes and fill the missing
    values with 0. Finally, return the dataframes in standard tabular format by resetting the index.
    """
    # get pandas series, use the columns except the last one (which is the COUNT) as the index
    data_true: pd.Series = data_true.set_index(data_true.columns[:-1].tolist())
    dp_data: pd.Series = dp_data.set_index(dp_data.columns[:-1].tolist())
    # get the union of the indexes
    index = data_true.index.union(dp_data.index)
    # reindex the dataframes on the union of the indexes and fill the missing values with 0
    data_true = data_true.reindex(index, fill_value=0)
    dp_data = dp_data.reindex(index, fill_value=0)
    # return the dataset in standard tabular format
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
