import pandas as pd
from .geospine import GeoSpine


class OD_tree:

    def __init__(self,
                 data: pd.DataFrame,
                 spine: GeoSpine,
                 orig_dest: tuple[str] = None,
                 flow_column: str = None,
                 mode: str = "destination",
                 final_level: int = None):
        if final_level is None:
            final_level = spine.depth
        if orig_dest is None:
            orig_dest = ("_ORIG", "_DEST")
        if flow_column is None:
            flow_column = "COUNT"
        if mode == "destination":
            od_to_level = [(i, i) if i == j else (i, j) for i in range(final_level + 1) for j in range(i, i + 2)]
            bool_od = [(False, True) if i % 2 == 0 else (True, False) for i in range(final_level * 2)]
            # drop the last element
            od_to_level = od_to_level[:-1]
        else:
            raise ValueError("Mode not recognized")
        self.data = data
        self.spine = spine
        self.orig_dest = orig_dest
        self.flow_column = flow_column
        self.od_to_level = od_to_level
        self._bool_od = bool_od
        self.depth = 2 * final_level

    def _get_od_levels(self, level: int) -> tuple[int, int]:
        """
        Get the levels of the origin and destination nodes
        :param od: tuple[int, int], origin and destination nodes
        :return: tuple[int, int], levels of the origin and destination nodes
        """
        return self.od_to_level[level]

    def get_data_at_level(self, level: int) -> pd.DataFrame:
        od_level: tuple = self._get_od_levels(level)
        df = self.data
        orig_column = ["LEVEL" + str(x) + self.orig_dest[0] for x in range(od_level[0] + 1)]
        dest_column = ["LEVEL" + str(x) + self.orig_dest[1] for x in range(od_level[1] + 1)]
        return df.groupby(orig_column + dest_column, as_index=False)[self.flow_column].sum()

    def stable_query_level(self, level: int) -> pd.Series:
        """
        Query the level of the tree, only return non-zero values
        :param level: int, level to query
        :return: pd.Series, data at the level
        """

        # get the data at the level
        df = self.data
        od_level: tuple = self._get_od_levels(level)
        orig_column = "LEVEL" + str(od_level[0]) + self.orig_dest[0]
        dest_column = "LEVEL" + str(od_level[1]) + self.orig_dest[1]
        query = df.groupby([orig_column] + [dest_column])[self.flow_column].sum()
        return query

    def full_query_level(self, level: int) -> pd.Series:
        """
        Query the level of the tree, return all possible OD pairs
        :param level: int, level to query
        :return: pd.Series, data at the level
        """
        od_level: tuple = self._get_od_levels(level)
        stable_data = self.stable_query_level(level)
        # get all possible OD pairs from spine
        orig_nodes = self.spine.get_nodes(level=od_level[0])
        dest_nodes = self.spine.get_nodes(level=od_level[1])
        all_nodes = [(orig, dest) for orig in orig_nodes for dest in dest_nodes]
        # reindex stable data by adding zero
        query = stable_data.reindex(all_nodes, fill_value=0)
        return query

    def stable_child_query_level(self, level: int, nodes: tuple[any, any]) -> pd.Series:
        """
        Return the attributes of the children of a node, only return non-zero values
        :param nodes:
        :param level:
        :return:
        """
        df = self.data
        od_level: tuple = self._get_od_levels(level)
        bool_od = self._bool_od[level]
        orig_node, dest_node = nodes
        # get the children
        if bool_od[0]:
            orig_node_children = self.spine.get_children(level=od_level[0], father_node=orig_node)
        else:
            orig_node_children = [orig_node]
        if bool_od[1]:
            dest_node_children = self.spine.get_children(level=od_level[1], father_node=dest_node)
        else:
            dest_node_children = [dest_node]
        # get attributes
        od_level: tuple = self._get_od_levels(level + 1)
        orig_column = "LEVEL" + str(od_level[0]) + self.orig_dest[0]
        dest_column = "LEVEL" + str(od_level[1]) + self.orig_dest[1]
        query = df[df[orig_column].isin(orig_node_children) & df[dest_column].isin(dest_node_children)][
            [orig_column, dest_column, self.flow_column]]
        query = query.groupby([orig_column, dest_column])[self.flow_column].sum()
        return query

    def full_child_query_level(self, level: int, nodes: tuple[any, any]) -> pd.Series:
        """
        Return the attributes of the children of a node
        :param level:
        :return:
        """
        od_level: tuple = self._get_od_levels(level)
        bool_od = self._bool_od[level]
        orig_node, dest_node = nodes
        # get the children
        if bool_od[0]:
            orig_node_children = self.spine.get_children(level=od_level[0], father_node=orig_node)
        else:
            orig_node_children = [orig_node]
        if bool_od[1]:
            dest_node_children = self.spine.get_children(level=od_level[1], father_node=dest_node)
        else:
            dest_node_children = [dest_node]
        # get all possible OD pairs from spine
        all_nodes = [(orig, dest) for orig in orig_node_children for dest in dest_node_children]
        # get stable query
        stable_data = self.stable_child_query_level(level, nodes)
        # reindex stable data by adding zero
        query = stable_data.reindex(all_nodes, fill_value=0)
        return query
