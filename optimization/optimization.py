from typing import Union

import numpy as np
import cvxpy as cp


def standard_int_opt(y: np.array(int),
                     c: int,
                     p: Union[str, int]) -> np.array(int):
    """
    Solves the vector optimization problem with non-negative elements.

    :param y: A vector of floats.
    :param c: A number representing the sum of the elements in the solution vector.
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
    assert isinstance(c, int), f"Constraint must be an integer, got {type(c)}"
    # make sure the input is an integer
    assert np.all(type(x) == int for x in y), f"Input vector must be an integer vector, got {y.dtype}"

    # Define the optimization problem
    x = cp.Variable(len(y), integer=int_type)
    objective = cp.Minimize(cp.norm(x - y, p))
    constraints = [cp.sum(x) == c, x >= 0]
    problem = cp.Problem(objective, constraints)

    # solve th problem
    problem.solve(solver=solver)

    # if the solution is not integer, round it and adjust it to satisfy the constraint
    if not int_type:
        if problem.status in ["optimal", "optimal_inaccurate"]:
            x = np.round(x.value).astype(int)

            # Check if the sum constraint is met
            current_sum = np.sum(x)
            diff = c - current_sum

            if diff != 0:
                if diff > 0:
                    # Distribute excess needed (+diff) across indices with higher values
                    I = np.argsort(-x)[:diff]  # Largest values first
                    x[I] += 1
                elif diff < 0:
                    # Remove excess (-diff) across indices with smaller values
                    I = np.argsort(x)  # Smallest values first
                    I = I[x[I] > 0][:abs(diff)]  # Avoid zero values
                    x[I] -= 1
        else:
            raise ValueError(f"Solver did not find a valid solution. Status: {problem.status}")
    else:
        # if the solution is integer, round it to make sure it is integer
        x = np.round(x.value).astype(int)

    # check the constraint
    assert np.sum(x) == c, f"Sum of x: {np.sum(x)}, constraint: {c}"
    return x


def fast_int_opt(x: np.array(int), c: int) -> np.array:
    # get the best solution in the reals
    d = len(x)
    z = (c - x.sum()) / d
    # get the integer rounding
    z_int = int(np.floor(z))
    y: np.array = x + z_int
    # get the offsets
    d_tilde = c - x.sum() - d * z_int
    # get indices of y in ascending order
    I = np.argsort(-y)[:d_tilde]
    # add ones
    y[I] += 1
    assert y.sum() == c, f"Sum of y: {y.sum()}, constraint: {c}"
    # get maximum allowable subtraction
    t = max(-min(y) + 1, 1)
    # add non-negativity constraint
    y = np.maximum(y, 0)
    if y.sum() == c:
        return y
    # apply subtraction
    I = np.argsort(y)  # get indices of y in ascending order
    I = I[y[I] > 0]
    i = 0
    y_sum = y.sum()
    while y_sum != c:
        z = y_sum - c
        y_new = max(y[I[i]] - min(z, t), 0)
        reduction = y[I[i]] - y_new
        y[I[i]] = y_new
        y_sum -= reduction
        i += 1
        if i == len(I):
            I = I[y[I] > 0]
            g = len(I)
            r = int((1 / g) * (y_sum - c))
            y_min = min(y[I])
            t = min(max(r, 1), y_min)
            while t > 1:
                y = np.maximum(y - t, 0)
                y_sum = y.sum()
                I = I[y[I] > 0]
                g = len(I)
                r = int((1 / g) * (y.sum() - c))
                y_min = min(y[I])
                t = min(max(r, 1), y_min)
            i = 0

    assert y.sum() == c, f"Sum of y: {y.sum()}, constraint: {c}"
    return y


def int_opt(x: np.array(int), c: int) -> np.array:
    # get the best solution in the reals
    d = len(x)
    z = (c - x.sum()) / d
    # get the integer rounding
    z_int = int(np.floor(z))
    y: np.array = x + z_int
    # get the offsets
    d_tilde = c - x.sum() - d * z_int
    # get indices of y in ascending order
    I = np.argsort(-y)[:d_tilde]
    # add ones
    y[I] += 1
    assert y.sum() == c, f"Sum of y: {y.sum()}, constraint: {c}"
    # get maximum allowable subtraction
    t = max(-min(y) + 1, 1)
    # add non-negativity constraint
    y = np.maximum(y, 0)
    if y.sum() == c:
        return y
    # apply subtraction
    I = np.argsort(y)  # get indices of y in ascending order
    I = I[y[I] > 0]
    i = 0
    y_sum = y.sum()
    while y_sum != c:
        z = y_sum - c
        y_new = max(y[I[i]] - min(z, t), 0)
        reduction = y[I[i]] - y_new
        y[I[i]] = y_new
        y_sum -= reduction
        i += 1
        if i == len(I):
            I = I[y[I] > 0]
            t = 1
            i = 0

    assert y.sum() == c, f"Sum of y: {y.sum()}, constraint: {c}"
    return y

