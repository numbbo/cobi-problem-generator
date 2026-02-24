import numpy as np
from typing import Union, Tuple, Dict, Optional
from cobi import CobiProblem


def create_random_hessian(n_var):
    """ Returns a random symmetric positive definite Hessian matrix of shape (n_var, n_var). """
    A = np.random.randn(n_var, n_var)
    symmetric_matrix = np.dot(A, A.T)
    positive_definite_matrix = symmetric_matrix + n_var * np.eye(n_var)
    return positive_definite_matrix


def create_random_hessian_with_condition_number(n_var, cond):
    """ Returns a random Hessian matrix of shape (n_var, n_var) with a condition number in the specified range. """
    low, high = cond
    cond = np.exp(np.random.uniform(np.log(low), np.log(high)))
    lambda_min = 1
    lambda_max = lambda_min * cond
    middle_eigenvalues = np.exp(np.random.uniform(np.log(lambda_min), np.log(lambda_max), n_var - 2))
    eigenvalues = np.concatenate([[lambda_max], middle_eigenvalues, [lambda_min]])
    np.random.shuffle(eigenvalues)
    Q, _ = np.linalg.qr(np.random.randn(n_var, n_var))
    D = np.diag(eigenvalues)
    A = Q @ D @ Q.T
    return A


def get_n_var(val):
    """
    Validates the input for n_var.
    """
    if isinstance(val, int):
        return val
    else:
        raise ValueError(
            f'Unsupported input for n_var: {val}. Expected an int.')


def choose_number(val, name, min):
    """
    Validates the input val:
    - If val is a number greater than or equal to min, returns it.
    - If val is a tuple (a, b) where min <= a <= b, returns a random integer v such that a <= v <= b.
    """
    if isinstance(val, int) and min <= val:
        return val
    elif (isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], int) and isinstance(val[1], int) and min <= val[0] <= val[1]):
        return np.random.randint(val[0], val[1] + 1)
    else:
        raise ValueError(
            f'Unsupported input for {name}: {val}. Expected an integer greater than or equal to {min} or a tuple '
            f'of two integers (a, b), where {min} <= a <= b.')


def get_peaks_alphas(val):
    """
    Validates the input for peaks_value_shift and returns a proper range.
    """
    if isinstance(val, int) or isinstance(val, float):
        return (val, val)
    elif isinstance(val, tuple) and len(val) == 2 and 0 < val[0] <= val[1]:
        return val
    else:
        raise ValueError(f'Unsupported input for peaks_alphas: {val}. Expected a positive float or a tuple of positive floats (min, max).')
    

def get_shift(val):
    """
    Validates the input for peaks_value_shift and returns a proper range.
    """
    if isinstance(val, int) or isinstance(val, float):
        return (-val, val)
    elif isinstance(val, tuple) and len(val) == 2 and (isinstance(val[0], int) or isinstance(val[0], float)) and (
            isinstance(val[1], int) or isinstance(val[1], float)) and val[0] <= val[1]:
        return val
    else:
        raise ValueError(
            f'Unsupported input for peaks_value_shift: {val}. Expected a float or a tuple of two floats (min, max).')


def get_condition_number(val, name):
    """
    Validates the input for condition number and returns a proper range.
    """
    if (isinstance(val, int) or isinstance(val, float)) and 1 <= val:
        return (1, val)
    elif isinstance(val, tuple) and len(val) == 2 and (isinstance(val[0], int) or isinstance(val[0], float)) and (
            isinstance(val[1], int) or isinstance(val[1], float)) and 1 <= val[0] <= val[1]:
        return val
    else:
        raise ValueError(
            f'Unsupported input for {name}: {val}. Expected a float greater than 1 or a tuple of two '
            f'floats greater than 1 (min, max).')


def get_quadratic_constraint_size(val):
    """
    Validates the input for quadratic_constraints_size and returns a proper range.
    """
    if (isinstance(val, int) or isinstance(val, float)) and 0 < val:
        return (val, val)
    elif isinstance(val, tuple) and len(val) == 2 and (isinstance(val[0], int) or isinstance(val[0], float)) and (
            isinstance(val[1], int) or isinstance(val[1], float)) and 0 < val[0] <= val[1]:
        return val
    else:
        raise ValueError(
            f'Unsupported input for quadratic_constraints_size: {val}. Expected a positive float or a tuple of two '
            f'positive floats (min, max).')


def get_alpha(val):
    """
    Validates the input for alpha and returns a proper vector.
    """
    if (isinstance(val, int) or isinstance(val, float)) and 0 < val:
        return (val, val)
    elif isinstance(val, tuple) and len(val) == 2 and (isinstance(val[0], int) or isinstance(val[0], float)) and (
            isinstance(val[1], int) or isinstance(val[1], float)) and 0 < val[0] and 0 < val[1]:
        return val
    else:
        raise ValueError(
            f'Unsupported input for alpha: {val}. Expected a positive float or a tuple of two positive floats.')


def create_linear_constraint(n_var, domain, feasible_pt=None, perpendicular=False):
    """
    Creates a random linear constraint. If feasible_pt is set, ensures that this point is feasible. If perpendicular is
    True, the constraint is perpendicular to the x1-x2 plane.
    """
    p = np.random.uniform(domain[0], domain[1], n_var)
    if perpendicular:
        n = np.concatenate([np.random.uniform(-1, 1, 2), np.zeros(n_var - 2)])
    else:
        n = np.random.uniform(-1, 1, n_var)
    if feasible_pt is not None and np.dot(feasible_pt - p, n) > 0:
        n *= -1
    return {'P': p, 'n': n}


def create_quadratic_constraint(n_var, domain, quadratic_constraints_size, quadratic_constraints_condition_number,
                                feasible_pt=None):
    """
    Creates a random quadratic constraint. If feasible_pt is set, ensures that this point is feasible.
    """
    if quadratic_constraints_condition_number is None:
        H = create_random_hessian(n_var)
    else:
        H = create_random_hessian_with_condition_number(n_var, quadratic_constraints_condition_number)

    low, high = quadratic_constraints_size
    b = np.exp(np.random.uniform(np.log(low), np.log(high))) ** 2

    if feasible_pt is not None:
        direction = np.random.randn(n_var)
        direction /= np.linalg.norm(direction)
        max_distance = np.sqrt(b / (direction.T @ H @ direction))
        distance = np.random.uniform(0, max_distance)
        c = feasible_pt + distance * direction
    else:
        c = np.random.uniform(domain[0], domain[1], n_var)

    return {'H': H, 'c': c, 'b': b}


def set_n_digits(x, n_digits):
    """
    Rounds all numbers in x to n_digits digits.
    """
    if isinstance(x, np.ndarray):
        return np.round(x, n_digits)
    elif isinstance(x, list):
        return [set_n_digits(e, n_digits) for e in x]
    elif isinstance(x, tuple):
        return tuple(set_n_digits(e, n_digits) for e in x)
    elif isinstance(x, dict):
        return {k: set_n_digits(v, n_digits) for k, v in x.items()}
    elif isinstance(x, (int, float)):
        return round(x, n_digits)
    else:
        return x


def create_random_problem(
    n_var: int = 2,
    seed: Optional[int] = None,
    domain: Tuple[float, float] = (-5, 5),
    n_peaks: Tuple[
        Union[int, Tuple[int, int]],
        Union[int, Tuple[int, int]]
    ] = ((2, 5), (2, 5)),
    peaks_value_shift: Union[float, Tuple[float, float]] = 10,
    peaks_condition_number: Optional[Union[float, Tuple[float, float]]] = None,
    peaks_alphas: Union[float, Tuple[float, float]] = 1,
    alpha: Union[float, Tuple[float, float]] = (1, 1),
    n_constraints: Optional[Dict[str, Union[int, Tuple[int, int]]]] = None,
    boundary_constraints: bool = True,
    quadratic_constraints_size: Union[float, Tuple[float, float]] = 10,
    quadratic_constraints_condition_number: Optional[Union[float, Tuple[float, float]]] = None,
    n_multi_constraints_groups: Union[int, Tuple[int, int]] = 2,
    n_multi_constraints_group_linear: Union[int, Tuple[int, int]] = (0, 1),
    n_multi_constraints_group_quadratic: Union[int, Tuple[int, int]] = (2, 3),
    constraints_feasible: bool = True,
    perpendicular_linear_constraints: bool = False,
    n_digits: Optional[int] = None,
    print_seed: bool = True
) -> CobiProblem:
    """
    Generates a random CobiProblem.

    Parameters:
    - n_var (int): Dimension of the decision/search space.
    - seed (int): Random seed for reproducibility.
    - domain (tuple of float): Lower and upper bounds for all decision variables.
    - n_peaks (tuple of int or tuple of tuples of int): Number of peaks for each objective function. If a tuple (min, max), a random integer in that range is chosen.
    - peaks_value_shift (float, or tuple of float): Range from which the f-value shifts b1_i, b2_j for peaks are sampled. If a single number x, shifts are sampled uniformly from [-x, x].
    If a tuple (min, max), shifts are sampled uniformly from [min, max].
    - peaks_condition_number (float, tuple of float, or None): Condition number for Hessian matrices of objective functions. If a number x, the actual condition number is sampled logarithmically
    from [1, x]. If a tuple (min, max), it is sampled from [min, max]. If None, Hessians with random condition numbers are generated.
    - peaks_alphas (float, or tuple of float): Range from which the alphas for peaks are sampled. If a single number x, all alphas are x.
    If a tuple (min, max), alphas are sampled uniformly from [min, max].
    - alpha (float or tuple of float): Exponents used to transform the objective functions. If a single number x, alpha_1 = alpha_2 = x.
    - n_constraints (dict or None): Dictionary with the number of constraints of each type. Must have the form {'Linear': number of linear constraints, 'Quadratic': number of quadratic constraints,
    'Multi': number of multi-constraints}. If None, default dictionary {'Linear': 1, 'Quadratic': 1, 'Multi': 1} is used.
    - boundary_constraints (bool): If True, automatically adds boundary constraints for each decision variable, ensuring that constraint violations reflect the domain.
    - quadratic_constraints_size (float, or tuple of float): Range from which sizes of quadratic constraints are sampled. If a single number x, all sizes are x.
    If a tuple (min, max), sizes are sampled logarithmically from [min, max].
    - quadratic_constraints_condition_number (float, tuple of float, or None): Condition number for Hessian matrices of quadratic constraints. Follows the same rules as peaks_condition_number.
    - n_multi_constraints_groups (int or tuple of int): Number of groups in each multi-constraint. A multi-constraint has the form min_k [max_l [g_{k,l}]] <= 0, where g_{k,l} are linear or
    convex-quadratic constraints and k runs from 1 to the number of groups. If a tuple (min, max) is provided, a random integer in that range is chosen for each multi-constraint.
    - n_multi_constraints_group_linear (int or tuple of int): Number of linear constraints in each group of a multi-constraint. For group k of a multi-constraint, this defines the number of linear
    constraints in {g_{k,1}, ..., g_{k,m_k}}. If a tuple (min, max) is provided, a random integer in that range is chosen for each group.
    - n_multi_constraints_group_quadratic (int or tuple of int): Number of quadratic constraints in each group of a multi-constraint. For group k of a multi-constraint, this defines the number of
    quadratic constraints in {g_{k,1}, ..., g_{k,m_k}}. If a tuple (min, max) is provided, a random integer in that range is chosen for each group.
    - constraints_feasible (bool): If True, all constraints are generated so that some randomly sampled point is feasible.
    - perpendicular_linear_constraints (bool): If True, linear constraints are generated perpendicular to the x1-x2 plane.
    - n_digits (int or None): If not None, rounds all generated numbers to this number of digits.
    - print_seed (bool): If True, prints the seed used for generating the problem.

    Returns a randomly generated CobiProblem instance.
    """
    if n_constraints is None:
        n_constraints = {'Linear': 1, 'Quadratic': 1, 'Multi': 1}
    if not seed:
        seed = np.random.randint(1, 1000000)

    np.random.seed(seed)
    if print_seed:
        print(f'Random seed: {seed}')

    n_var = get_n_var(n_var)

    n_peaks_f1 = choose_number(n_peaks[0], 'n_peaks_f1', 1)
    n_peaks_f2 = choose_number(n_peaks[1], 'n_peaks_f2', 1)
    
    peaks_value_shift = get_shift(peaks_value_shift)
    peaks_alphas = get_peaks_alphas(peaks_alphas)

    if peaks_condition_number is not None:
        peaks_condition_number = get_condition_number(peaks_condition_number, 'peaks_condition_number')

    # Objectives
    centers_f1 = np.random.uniform(domain[0], domain[1], (n_peaks_f1, n_var))
    v_shifts_f1 = np.random.uniform(peaks_value_shift[0], peaks_value_shift[1], n_peaks_f1)
    Hessians_f1 = [create_random_hessian(n_var) if peaks_condition_number is None
                   else create_random_hessian_with_condition_number(n_var, peaks_condition_number) for _ in range(n_peaks_f1)]
    alphas_f1 = np.random.uniform(peaks_alphas[0], peaks_alphas[1], n_peaks_f1)
    centers_f2 = np.random.uniform(domain[0], domain[1], (n_peaks_f2, n_var))
    v_shifts_f2 = np.random.uniform(peaks_value_shift[0], peaks_value_shift[1], n_peaks_f2)
    Hessians_f2 = [create_random_hessian(n_var) if peaks_condition_number is None
                   else create_random_hessian_with_condition_number(n_var, peaks_condition_number) for _ in range(n_peaks_f2)]
    alphas_f2 = np.random.uniform(peaks_alphas[0], peaks_alphas[1], n_peaks_f2)
    
    feasible_pt = np.random.uniform(domain[0], domain[1], n_var) if constraints_feasible else None

    n_linear_constraints = n_constraints['Linear']
    n_quadratic_constraints = n_constraints['Quadratic']
    n_multi_constraints = n_constraints['Multi']

    # Linear constraints
    n_linear_constraints = choose_number(n_linear_constraints, 'n_linear_constraints', 0)
    linear_constraints = []
    for _ in range(n_linear_constraints):
        linear_constraint = create_linear_constraint(n_var, domain, feasible_pt=feasible_pt,
                                                     perpendicular=perpendicular_linear_constraints)
        linear_constraints.append(linear_constraint)

    # Quadratic constraints
    quadratic_constraints_size = get_quadratic_constraint_size(quadratic_constraints_size)
    if quadratic_constraints_condition_number is not None:
        quadratic_constraints_condition_number = get_condition_number(quadratic_constraints_condition_number, 'quadratic_constraints_condition_number')
    n_quadratic_constraints = choose_number(n_quadratic_constraints, 'n_quadratic_constraints', 0)
    quadratic_constraints = []
    for _ in range(n_quadratic_constraints):
        quadratic_constraint = create_quadratic_constraint(n_var, domain, quadratic_constraints_size,
                                                           quadratic_constraints_condition_number,
                                                           feasible_pt=feasible_pt)
        quadratic_constraints.append(quadratic_constraint)

    # Multi constraints
    n_multi_constraints = choose_number(n_multi_constraints, 'n_multi_constraints', 0)
    multi_constraints = []
    for _ in range(n_multi_constraints):
        multi_constraint = []
        n_groups = choose_number(n_multi_constraints_groups, 'n_multi_constraints_groups', 0)
        for i in range(n_groups):
            feasible_pt_group = feasible_pt if i == 0 else (
                np.random.uniform(domain[0], domain[1], n_var) if constraints_feasible else None)
            group = {'Linear': [], 'Quadratic': []}
            n_group_linear = choose_number(n_multi_constraints_group_linear, 'n_multi_constraints_group_linear', 0)
            for _ in range(n_group_linear):
                linear_constraint = create_linear_constraint(n_var, domain, feasible_pt=feasible_pt_group,
                                                             perpendicular=perpendicular_linear_constraints)
                group['Linear'].append(linear_constraint)
            n_group_quadratic = choose_number(n_multi_constraints_group_quadratic,
                                              'n_multi_constraints_group_quadratic', 0)
            for _ in range(n_group_quadratic):
                quadratic_constraint = create_quadratic_constraint(n_var, domain, quadratic_constraints_size,
                                                                   quadratic_constraints_condition_number,
                                                                   feasible_pt=feasible_pt_group)
                group['Quadratic'].append(quadratic_constraint)
            multi_constraint.append(group)
        multi_constraints.append(multi_constraint)

    alpha = get_alpha(alpha)
    objectives = ({'H': Hessians_f1, 'c': centers_f1, 'b': v_shifts_f1, 'alphas': alphas_f1}, {'H': Hessians_f2, 'c': centers_f2, 'b': v_shifts_f2, 'alphas': alphas_f2})
    constraints = {'Linear': linear_constraints, 'Quadratic': quadratic_constraints, 'Multi': multi_constraints}

    if n_digits is not None:
        objectives = set_n_digits(objectives, n_digits)
        constraints = set_n_digits(constraints, n_digits)

    problem = CobiProblem(n_var, objectives, constraints, domain=domain, alpha=alpha, boundary_constraints=boundary_constraints)
    return problem