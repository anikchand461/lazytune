from itertools import product


def generate_param_combinations(param_grid):
    """
    Convert parameter grid to list of combinations
    """

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)

    return combinations
