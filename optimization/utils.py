import numpy as np
from numba import njit


def check_constraint(x: np.array(float), y: float, precision=1e-2):
    """
    Check if the constraint is satisfied, hence the sum of the elements in x is equal to y
    :param x: np.array to check the sum
    :param y: constraint
    :param precision: precision of the check
    :return: None
    """
    assert abs(sum(x) - y) < precision, f"Constraint not satisfied, sum(x) = {sum(x)} != {y} with x = {x}"


# @njit
# def get_sorted_indices(vector: list[int], threshold: int = 0, descending=False) -> list[int]:
#     # # Convert to a numpy array for efficient indexing
#     # vector = np.array(vector)
#
#     # Get indices where elements are above the threshold
#     indices = np.where(vector > threshold)[0]
#
#     # Sort indices by the corresponding values in the vector
#     sorted_indices = indices[np.argsort(vector[indices], kind='stable')]
#
#     # Reverse order if descending is True
#     if descending:
#         sorted_indices = sorted_indices[::-1]
#
#     return sorted_indices.tolist()

@njit
def get_sorted_indices(vector, threshold=0, descending=False):
    # # Ensure input is a NumPy array for efficient computation
    # vector = np.asarray(vector)

    # Get indices where elements are above the threshold
    indices = np.where(vector > threshold)[0]

    # Sort indices based on values in the vector
    sorted_indices = indices[np.argsort(vector[indices])]

    # Reverse order if descending is True
    if descending:
        sorted_indices = sorted_indices[::-1]

    return sorted_indices  # Returns as a NumPy array, compatible with Numba

# @njit
# def remove_indices_to_zero(I: list[int], x: np.array) -> np.array:
#     """
#     Filters out indices from I where corresponding values in x are zero
#     :param I: list of indices
#     :param x: np.array of floats
#     :return: np.array of remaining indices
#     """
#     # Use list comprehension to filter out indices where x[i] == 0
#     return [i for i in I if x[i] != 0]


# def remove_indices_to_zero(indices, y):
#     return indices[y[indices] > 0]
