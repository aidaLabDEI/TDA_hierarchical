class GeoSpine:
    """
    This class is a wrapper for a nested dictionary that represents a hierarchical spine.
    Root node always indicated with 0.
    :param nested_dictionary: dict, nested dictionary representing the hierarchical spine
    """

    def __init__(self, nested_dictionary: dict):
        self.original_spine = nested_dictionary
        self.geo_spine = nested_dictionary
        self.depth = self.get_depth()

    def access_level(self, level: int) -> dict:
        """
        Access to a level of the nested dictionary
        :param level:int level of the nested dictionary, zero level is the root
        """

        # Helper function for recursive traversal
        def traverse(d, current_level):
            if current_level == level:
                return d
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result.update(traverse(value, current_level + 1))
            return result

        return traverse(self.geo_spine, 1)

    def update_spine_to_level(self, final_level: int):
        """
        Recursively extract the nested dictionary representation of the tree up to the given level. It updates the
        `geo_spine` attribute with the new tree.

        :param level: The level up to which the tree should be returned.
        """

        def _get_tree_to_level(tree: dict, final_level: int, current_level: int = 0) -> dict:
            """
            Recursively extract the nested dictionary representation of the tree up to the given level.

            :param tree: The nested dictionary representing the tree.
            :param level: The level up to which the tree should be returned.
            :param current_level: The current level in the recursive traversal (used internally).
            :return: A dictionary representing the tree up to the specified level.
            """
            # Base case: if current level is greater than or equal to the target level, return an empty dict
            if current_level >= final_level:
                return {}

            # Create a new dictionary to store the tree up to the specified level
            tree_subset = {}

            for node, subtree in tree.items():
                # Recursively include children only if we are below the target level
                tree_subset[node] = _get_tree_to_level(subtree, final_level, current_level + 1)

            return tree_subset

        self.geo_spine = _get_tree_to_level(tree=self.original_spine, final_level=final_level)
        self.depth = final_level

    def get_nodes(self, level: int) -> list[any]:
        """
        Returns the list of nodes in a level
        :param level: int, level of the nested dictionary, zero level is the root
        """

        if level > self.depth:
            raise Exception("Level out of range, max level is: ", self.depth - 1)
        if level == 0:
            return [0]
        try:
            return list(self.access_level(level).keys())
        except KeyError:
            print("Level not found")

    def get_children(self, level: int, father_node: any) -> list[any]:
        """
        Returns a list of nodes at level=a+1 children of node at level=a
        :param level:int, level of the nested dictionary, zero level is the root
        :param node:any, node of the graph at level=a, if level=0 unique node is 0
        """
        if level == 0 and father_node == 0:
            return list(self.geo_spine.keys())
        if level >= self.depth:
            raise Exception("Level out of range, max level is: ", self.depth - 1)
        try:
            return list(self.access_level(level)[father_node].keys())
        except KeyError:
            print("Node not found")

    def get_father(self, level: int, child_node: any) -> any:
        """
        Returns the father node of the given node at the specified level
        :param level: int, this is the level of the child_node
        :param child_node: any, node of the graph at level
        """
        assert (0 < level <= self.depth), "Level out of range"
        if level == 1:
            # return the root
            return 0
        for node in self.get_nodes(level - 1):
            if child_node in self.get_children(level - 1, node):
                return node

    def get_depth(self) -> int:
        """
        Returns how many layers has the hierarchical spine
        """

        # Helper function for recursive depth calculation
        def depth(d):
            if not isinstance(d, dict) or not d:
                return 0
            return 1 + max(depth(v) for v in d.values())

        return depth(self.geo_spine)

    def get_descendants(self, level: int, node: any, alpha: int) -> list[any]:
        """
        Returns a list of descendants of a node at level alpha
        Each descendant is represented as a tuple: (layer, node).

        :param level: int, the layer of the start node, zero level is the root.
        :param node: any, the start node from which to search for descendants.
        :param alpha: int, the level of the descendants.
        :return: list of descendants of the node at level alpha.
        """
        # assert that alpha > level
        assert level < alpha <= self.depth, "alpha must be greater than level and less than the depth of the spine."

        descendants = []

        # Helper function to recursively find descendants
        def find_descendants(current_layer, current_node, beta):
            if beta == current_layer:
                # Base case: h levels down
                descendants.append(current_node)
            else:
                children = self.get_children(current_layer, current_node)
                for child in children:
                    find_descendants(current_layer + 1, child, beta)

        find_descendants(level, node, alpha)
        return descendants

    def get_ancestors(self, level: int, node: any, alpha: int) -> list[any]:
        """
        Returns the ancestor of a node at level alpha.
        :param level: int, the layer of the start node, zero level is the root.
        :param node: any, the start node from which to search for ancestors.
        :param alpha: int, the number of levels to go up in the tree.
        :return: the ancestor of the node, in a list.
        """
        assert 0 <= alpha < level, "alpha must be less than level and greater than or equal to 0."
        if alpha == 0:
            # return the root
            return [0]

        ancestors = []

        # Helper function to recursively find ancestors
        def find_ancestors(current_layer, current_node, beta):
            if beta == current_layer:
                # Base case: h levels up
                ancestors.append(current_node)
            else:
                father = self.get_father(current_layer, current_node)
                find_ancestors(current_layer - 1, father, beta)

        find_ancestors(level, node, alpha)
        return ancestors

    def get_paths_from(self, level: int, node: any) -> list[tuple]:
        """
        Return all the paths from node (level, node) to leaf nodes.
        Args:
            level (int): The level of the node to find paths containing it.
            node (any): The node to find paths containing it.
        Returns:
            list[tuple] - List of paths as tuples of nodes containing the specified node.
        """
        assert 0 <= level < self.depth, "Level out of range"

        # Helper function to recursively find paths
        def find_paths(current_layer, current_node, path):
            if current_layer == self.depth:
                # Base case: we've reached the leaf nodes (genders)
                paths.append(tuple(path))  # Add the accumulated path as a tuple
            else:
                children = self.get_children(current_layer, current_node)
                for child in children:
                    # Add the current node to the path and recurse
                    find_paths(current_layer + 1, child, path + [(current_node, child)])

        def flatten_tuple(tpl):
            """
            Flatten a tuple of tuples into a single tuple.

            Parameters:
            - tpl: A tuple of tuples.

            Returns:
            - A flat tuple with all elements from the nested tuples.
            """
            flat_list = [node]  # add starting node

            # Iterate over each nested tuple and extend the flat_list
            for sub_tpl in tpl:
                flat_list.extend([sub_tpl[1]])

            # Convert the list to a tuple
            return tuple(flat_list)

        paths = []
        find_paths(level, node, [])
        output = [flatten_tuple(t) for t in paths]
        return output

    def get_path(self, level: int, node: any) -> tuple:
        """
        Return the path from the root to the specified node.

        :param level: The level of the target node.
        :param node: The target node.
        :return: A tuple representing the path from the root to the node.
        """
        assert 0 <= level <= self.depth, "Level out of range"

        path = []

        # Traverse upwards to find the path to the root
        while level > 0:
            path.append(node)
            node = self.get_father(level, node)
            level -= 1

        # Add the root node at the end
        path.append(0)

        # Reverse the path to show it from root to the target node
        return tuple(reversed(path))

    def get_all_paths(self):
        """
        Return all the paths from the root to the leaf nodes.
        :return:
        """
        return self.get_paths_from(0, 0)

    # def get_paths(self, nodes: list[tuple[int, any]]) -> list[tuple]:
    #
    #     """
    #     Return the paths from the root to the specified nodes for a batch of nodes.
    #
    #     :param nodes: A list of tuples, where each tuple contains (level, node).
    #     :return: A list of tuples, where each tuple represents the path from the root to the corresponding node.
    #     """
    #     # Ensure that all levels are within the valid range
    #     for level, node in nodes:
    #         assert 0 <= level <= self.depth, "Level out of range"
    #
    #     # Memoization cache for get_father results
    #     father_cache = {}
    #
    #     def get_father_cached(level, node):
    #         """
    #         Helper function to get the father of a node with memoization.
    #         """
    #         if (level, node) not in father_cache:
    #             father_cache[(level, node)] = self.get_father(level, node)
    #         return father_cache[(level, node)]
    #
    #     results = []
    #
    #     # For each (level, node) in the input list
    #     for level, node in nodes:
    #         path = []
    #
    #         # Traverse upwards to find the path to the root
    #         current_level = level
    #         current_node = node
    #         while current_level > 0:
    #             path.append(current_node)
    #             current_node = get_father_cached(current_level, current_node)
    #             current_level -= 1
    #
    #         # Add the root node at the end
    #         path.append(0)
    #
    #         # Reverse the path to show it from root to the target node
    #         results.append(tuple(reversed(path)))
    #
    #     return results
