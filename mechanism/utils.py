import numpy as np


def split_rho_budget(rho: float, T: int, method: str = "uniform", b: int = None):
    if method == "uniform":
        output = [rho / T for _ in range(T)]
        # check if the sum of the output is equal to the budget
        assert np.abs(sum(output) - rho) < 1e-6, f"Sum of output: {sum(output)}, budget: {rho}"
        return output

    elif method == "smart":
        if b is None:
            raise ValueError("b must be set when using smart method")
        coeff = (np.pow(b, 2 / 3) - 1) / (np.pow(b, 2 / 3) - np.pow(b, -(2 / 3) * (T - 1)))
        output = [rho * coeff * np.pow(b, -(2 / 3) * (T - l - 1)) for l in range(T)]
        # check if the sum of the output is equal to the budget
        assert np.abs(sum(output) - rho) < 1e-6, f"Sum of output: {sum(output)}, budget: {rho}"
        return output

    else:
        raise ValueError(f"Invalid method: {method}")
