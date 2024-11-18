import numpy as np


def get_rho_from_budget(budget: tuple[float, float]) -> float:
    """
    Return rho (from zCDP) given epsilon and delta.
    :param budget: (epsilon, delta)
    :return: rho
    """
    epsilon = budget[0]
    delta = budget[1]
    return np.log(1 / delta) * (np.sqrt(1 + epsilon / np.log(1 / delta)) - 1) ** 2
