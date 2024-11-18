from typing import Union

import numpy as np
import cvxpy as cp
import argparse
from .utils import get_sorted_indices
from numba import njit


def optimize_int_vector(y: np.array(int),
                        constraint: int,
                        p: Union[str, int]) -> np.array(int):
    """
    Solves the vector optimization problem with non-negative elements.

    :param y: A vector of floats.
    :param constraint: A number representing the sum of the elements in the solution vector.
    :param p: The norm to use for the optimization. Can be "inf", 1, or 2.

    Returns:
    np.array: A float vector x that minimizes the L2 distance to y,
              with the sum of its elements equal to edge_constraint, and all elements non-negative.
    """
    # get the solver
    if p == 1 or p == "inf":
        solver = cp.GLPK_MI
        int_type = True
    elif p == 2:
        # get solver for real convex optimization
        solver = cp.CLARABEL
        int_type = False  # solve in the continuous domain
    else:
        raise ValueError("Only L2, L1, and Linf norms are supported for integer optimization")

    # make sure the constraint is an integer
    assert isinstance(constraint, int), f"Constraint must be an integer, got {type(constraint)}"
    # make sure the input is an integer
    assert np.all(type(x) == int for x in y), f"Input vector must be an integer vector, got {y.dtype}"

    # Define the optimization problem
    x = cp.Variable(len(y), integer=int_type)
    objective = cp.Minimize(cp.norm(x - y, p))
    constraints = [cp.sum(x) == constraint, x >= 0]
    problem = cp.Problem(objective, constraints)

    # solve th problem
    problem.solve(solver=solver)

    # if the solution is not integer, round it and adjust it to satisfy the constraint
    if not int_type:
        if problem.status in ["optimal", "optimal_inaccurate"]:
            x = np.round(x.value).astype(int)

            # Check if the sum constraint is met
            current_sum = np.sum(x)
            diff = constraint - current_sum

            # Adjust the solution to satisfy the constraint
            if diff != 0:
                if diff > 0:
                    # Distribute excess needed (+diff) across indices with higher values
                    indices = np.argsort(-x)[:diff]  # Largest values first
                    x[indices] += 1
                elif diff < 0:
                    # Remove excess (-diff) across indices with smaller values
                    nonzero_indices = np.nonzero(x)[0]  # Avoid zero values
                    indices = np.argsort(x[nonzero_indices])[:abs(diff)]  # Smallest value first
                    x[nonzero_indices[indices]] -= 1
        else:
            raise ValueError(f"Solver did not find a valid solution. Status: {problem.status}")
    else:
        # if the solution is integer, round it to make sure it is integer
        x = np.round(x.value).astype(int)

    # check the constraint
    assert np.sum(x) == constraint, f"Sum of x: {np.sum(x)}, constraint: {constraint}"
    return x

@njit
def int_opt(x: np.array(int),
            constraint: int) -> np.array:
    """
    Solves an integer optimization problem using the Algorithm 3 from the paper:
    :param x: A vector of integers.
    :param constraint: A number representing the sum of the elements in the solution vector.
    :return: A vector of integers that minimizes the L_inf distance to x, with the sum of its elements equal to
             constraint.
    """
    # first map to zero negative elements of y into the solution
    y = np.maximum(x, 0)
    # compute the linf distance between x and y
    t = max(np.max(np.abs(x - y)), 1)
    # get the sum
    y_sum = y.sum()
    # check constraint
    if y_sum == constraint:
        # we have a solution!
        return y
    elif y_sum > constraint:
        return subtraction(y=y, c=constraint, t=t, y_sum=y_sum)
    else:
        return addition(y=y, x=x, c=constraint, t=t, y_sum=y_sum)

@njit
def subtraction(y: np.array(int), c: int, t: int, y_sum: int) -> np.array:
    I = get_sorted_indices(y)
    i = 0
    while y_sum > c:
        z = y_sum - c
        y_new = max(y[I[i]] - min(z, t), 0)
        reduction = y[I[i]] - y_new
        y[I[i]] = y_new
        y_sum -= reduction
        i += 1
        if i == len(I):
            t = 1
            I = I[y[I] > 0]
            i = 0
    # check the constraint
    assert y.sum() == c, f"Sum of y: {y.sum()}, constraint: {c}"
    return y

@njit
def addition(y: np.array(int), x: np.array(int), c: int, t: int, y_sum: int) -> np.array:
    x_adjusted = x + t
    I = get_sorted_indices(x_adjusted, threshold=-1, descending=True)
    i = 0
    while y_sum < c:
        z = c - y.sum()
        y_new = min(y[I[i]] + z, x_adjusted[I[i]])
        increase = y_new - y[I[i]]
        y[I[i]] = y_new
        y_sum += increase
        i += 1
        if i == len(I):
            x_adjusted += 1
            i = 0
    # check the constraint
    assert y.sum() == c, f"Sum of y: {y.sum()}, constraint: {c}"
    return y
