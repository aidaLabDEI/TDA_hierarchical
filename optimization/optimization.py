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
    """
    From Algorithm 3 in the Appendix
    Args:
        x: np.array: array of integer
        c: int: constraint

    Returns: y: np.array: array of integer

    """
    a = (c - x.sum()) / len(x)
    constrait = int(c - x.sum())
    # round
    a_int = int(np.ceil(a))
    z = np.ones(len(x)) * a_int
    # clip
    z = np.maximum(z, -x).astype(int)
    t = max(abs(z))  # smallest value in z that is allowed
    I = np.argsort(x)  # get indices of x in ascending order
    I = I[z[I] > -x[I]]
    z_sum = z.sum()
    i = 0
    while z_sum > constrait:
        R = int(z.sum() - c + x.sum())
        z_new = max(z[I[i]] - R, -x[I[i]], -t)
        reduction = z[I[i]] - z_new
        z[I[i]] = z_new
        z_sum -= reduction
        i += 1
        if i == len(I):
            I = I[z[I] > -x[I]]
            g = len(I)
            r = int((1 / g) * (z_sum - constrait))
            t += max(1, r)
            i = 0
    return z + x
