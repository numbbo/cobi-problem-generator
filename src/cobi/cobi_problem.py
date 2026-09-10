from cobi import transform_objective, transform_constraint
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import itertools
from moarchiving import get_mo_archive
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from qpsolvers import solve_qp
import pickle
import cvxpy as cp
import copy
from sklearn.cluster import AgglomerativeClustering
from .utils import CMAP, plot_linear_constraints, plot_quadratic_constraints, plot_multi_constraints
from queue import PriorityQueue
import hashlib
from typing import Tuple, Union
from scipy.optimize import minimize
import warnings


def check_spd(H, tol=1e-8):
    """ Checks if the matrix H is symmetric positive definite. """
    if not np.allclose(H, H.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(H)
    return np.all(eigvals > 0)
    

def peak_function(x, c, H, b=0):
    """ Evaluates a single peak function at the point x. """
    x_diff = x - c
    return b + 0.5 * np.dot(x_diff.T, np.dot(H, x_diff))


def multi_peak_function(x, centers, Hessians, shifts=None):
    """ Evaluates the multi-peak function at the point x. """
    if shifts is None:
        shifts = np.zeros(len(centers))
    values = [peak_function(x, c, H, b) for c, H, b in zip(centers, Hessians, shifts)]
    return np.min(values)


def evaluate_linear_constraint(x, linear_constraint):
    """ Evaluates the given linear constraint at the point x. """
    return transform_constraint(np.dot(x - linear_constraint['P'], linear_constraint['n']), linear_constraint['transformation'])


def evaluate_quadratic_constraint(x, quadratic_constraint):
    """ Evaluates the given quadratic constraint at the point x. """
    H, c, b = quadratic_constraint['H'], quadratic_constraint['c'], quadratic_constraint['b']
    return transform_constraint((x - c).T @ H @ (x - c) - b, quadratic_constraint['transformation'])


def evaluate_linear_quadratic_constraints(x, linear_constraints, quadratic_constraints):
    """ Evaluates the given linear and quadratic constraints at the point x. """
    all_constraints = []
    for constraint in linear_constraints:
        all_constraints.append(evaluate_linear_constraint(x, constraint))
    for constraint in quadratic_constraints:
        all_constraints.append(evaluate_quadratic_constraint(x, constraint))
    return all_constraints


def evaluate_multi_constraint(x, multi_constraint):
    """ Evaluates the given multi-constraint at the point x. """
    multi_constraint_groups_values = []
    for multi_constraint_group in multi_constraint:
        group_constraint_values = evaluate_linear_quadratic_constraints(x, multi_constraint_group['Linear'], multi_constraint_group['Quadratic'])
        group_value = max(group_constraint_values)
        multi_constraint_groups_values.append(group_value)
    multi_constraint_value = min(multi_constraint_groups_values)
    return multi_constraint_value


def evaluate_boundary_constraints(x, lower_bound, upper_bound):
    """ Evaluates boundary constraints. """
    x = np.asarray(x)
    values = np.empty(2 * x.size, dtype=float)
    values[0::2] = x - upper_bound
    values[1::2] = lower_bound - x
    return values


def check_linear_quadratic_constraints(x, linear_constraints, quadratic_constraints, bounds=None):
    """ Checks feasibility with respect to optional boundary constraints and the given linear and quadratic constraints. """
    if bounds is not None:
        lower_bound, upper_bound = bounds
        if not np.all((lower_bound <= x) & (x <= upper_bound)):
            return False
    constraints_values = np.array(evaluate_linear_quadratic_constraints(x, linear_constraints, quadratic_constraints))
    return np.all(constraints_values <= 0)


def create_boundary_linear_constraints(n_var, lower_bound, upper_bound):
    """ Creates the linear representation of boundary constraints used when projecting points with a solver. """
    constraints = []
    for i in range(n_var):
        P_upper = np.zeros(n_var)
        n_upper = np.zeros(n_var)
        n_upper[i] = 1
        P_upper[i] = upper_bound[i] if np.ndim(upper_bound) else upper_bound
        constraints.append({'P': P_upper, 'n': n_upper})

        P_lower = np.zeros(n_var)
        n_lower = np.zeros(n_var)
        n_lower[i] = -1
        P_lower[i] = lower_bound[i] if np.ndim(lower_bound) else lower_bound
        constraints.append({'P': P_lower, 'n': n_lower})
    return constraints


def compute_point(H1, H2, c1, c2, t):
    """ Samples a point from the unconstrained Pareto set between the peaks defined by (H1, c1) and (H2, c2). """
    H_combined_inv = np.linalg.inv(t * H1 + (1 - t) * H2)
    return H_combined_inv @ (t * (H1 @ c1) + (1 - t) * (H2 @ c2))


def get_unconstrained_pareto_set_linspace_weights(H1, H2, c1, c2, n_points):
    """ Samples n_points points from the unconstrained Pareto set between the peaks defined by (H1, c1) and (H2, c2) using weights uniformly spaced in [0, 1].
    Also returns the corresponding weights. """
    ts = np.linspace(0, 1, n_points)
    points = [compute_point(H1, H2, c1, c2, t) for t in ts]
    return np.array(points, dtype=float), ts


def get_current_point(current_t, compute_point_fun, tol_jump, t_min, t_max):
    """ Computes the point using compute_point_fun. If unsuccessful, increments current_t by tol_jump and retries until a point
    is computed. If no point is found within [t_min, t_max], returns None for the point. Also returns the final current_t. """
    current_point = compute_point_fun(current_t)
    while current_point is None:
        current_t += tol_jump
        if t_min <= current_t <= t_max:
            current_point = compute_point_fun(current_t)
        else:
            break
    return current_point, current_t


def get_next_t_bisection(current_point, current_t, direction, t_min, t_max, tol_distance, tol_jump,
                         max_iter, distance, compute_point_fun, distance_fun, force_equidistant):
    """
    Finds the next t within [t_min, t_max] such that the point at t is within the specified distance 
    from current_point (according to distance_fun), using bisection. Stops and returns None if the
    maximum number of iterations (max_iter) is exceeded. Also returns the corresponding point.

    - If direction is 1, searches for t > current_t.
    - If direction is -1, searches for t < current_t.
    - If force_equidistant is True, the point must be approximately exactly at the target distance 
    (within tol_distance); otherwise, any point closer than distance is accepted.
    """
    if direction == 1:
        t_low = current_t
        t_high = t_max

        if force_equidistant:
            x_high, _ = get_current_point(t_high, compute_point_fun, -tol_jump, t_low, t_high)
            if x_high is None:
                return None, None
            dist_high = distance_fun(x_high, current_point)
            if dist_high + tol_distance < distance:
                return None, None
    elif direction == -1:
        t_low = t_min
        t_high = current_t

        if force_equidistant:
            x_low, _ = get_current_point(t_low, compute_point_fun, tol_jump, t_low, t_high)
            if x_low is None:
                return None, None
            dist_low = distance_fun(x_low, current_point)
            if dist_low + tol_distance < distance:
                return None, None
    else:
        raise ValueError("Direction must be 1 (forward) or -1 (backward).")

    for _ in range(max_iter):
        t_mid = (t_low + t_high) / 2.0
        if t_mid == t_low or t_mid == t_high:
            return None, None
        x_mid, t_mid = get_current_point(t_mid, compute_point_fun, direction * tol_jump, t_low, t_high)
        if x_mid is None:
            return None, None
        dist = distance_fun(x_mid, current_point)

        if (force_equidistant and abs(dist - distance) < tol_distance) or ((not force_equidistant) and dist < distance):
            return x_mid, t_mid

        if dist < distance:
            if direction == 1:
                t_low = t_mid
            else:
                t_high = t_mid
        else:
            if direction == 1:
                t_high = t_mid
            else:
                t_low = t_mid

    return None, None


def get_pareto_set_bisection_weights(distance, compute_point_fun, distance_fun, t_min=0.0, t_max=1.0,
                                     tol_distance=1e-5, tol_jump=1e-3, max_iter=100, force_equidistant=False):
    """
    Samples points such that each point is approximately a specified distance away from the previously sampled point.
    Points are computed using compute_point_fun, and the distance between points is evaluated using distance_fun.
    Weights of points are within [t_min, t_max].

    If force_equidistant is True, points are placed approximately exactly at the target distance (within tol_distance);
    otherwise, points closer than the distance are accepted.

    The function returns all sampled points along with their corresponding weights.
    """
    points = []
    ts = []
    current_t = t_min
    current_point, current_t = get_current_point(current_t, compute_point_fun, tol_jump, t_min, t_max)
    if current_point is None:
        return np.array([]), np.array([])
    if t_min == t_max:
        return np.array([current_point], dtype=float), np.array([current_t])
    while current_t < (t_min + t_max) / 2.0:
        points.append(current_point)
        ts.append(current_t)
        current_point, current_t = get_next_t_bisection(current_point, current_t, 1, t_min, t_max, tol_distance, tol_jump, max_iter, distance,
                                                        compute_point_fun, distance_fun, force_equidistant)
        if current_point is None:
            break
    last_point_first_part = points[-1] if len(points) > 0 else None
    len_first_part = len(points)

    current_t = t_max
    current_point, current_t = get_current_point(current_t, compute_point_fun, -tol_jump, t_min, t_max)
    if current_point is None:
        return np.array(points, dtype=float), np.array(ts)
    while current_t > (t_min + t_max) / 2.0:
        points.append(current_point)
        ts.append(current_t)
        current_point, current_t = get_next_t_bisection(current_point, current_t, -1, t_min, t_max, tol_distance, tol_jump, max_iter, distance,
                                                        compute_point_fun, distance_fun, force_equidistant)
        if current_point is None:
            break
    last_point_second_part = points[-1] if len(points[len_first_part:]) > 0 else None

    if (last_point_first_part is not None) and (last_point_second_part is not None) and (current_point is not None) and (
            (force_equidistant and distance_fun(last_point_first_part, last_point_second_part) >= distance + tol_distance) or (
            not force_equidistant and distance_fun(last_point_first_part, last_point_second_part) >= distance)):
        points.append(current_point)
        ts.append(current_t)

    return np.array(points, dtype=float), np.array(ts)


def project_point(H1, H2, c1, c2, w, C, d, feasibility_tolerance=1e-8, lambda_tolerance=1e-8):
    """
    Solves the quadratic program:
        min_x w*(x - c1)^T H1 (x - c1) + (1 - w)*(x - c2)^T H2 (x - c2)
        subject to Cx <= d
    using the KKT conditions without iterations.

    Parameters:
    - H1, H2: (n x n) positive-definite matrices
    - c1, c2: (n,) vectors
    - w: scalar in [0, 1]
    - C: (m x n) constraint matrix
    - d: (m,) constraint vector
    - feasibility_tolerance: scalar specifying the tolerance for primal feasibility,
    - lambda_tolerance: scalar specifying the tolerance for dual feasibility

    Returns:
    - best_x: Optimal solution vector
    """
    n = H1.shape[0]
    m = C.shape[0]

    H = 2 * w * H1 + 2 * (1 - w) * H2
    h = 2 * w * H1 @ c1 + 2 * (1 - w) * H2 @ c2

    def objective(x):
        return w * (x - c1).T @ H1 @ (x - c1) + (1 - w) * (x - c2).T @ H2 @ (x - c2)

    best_x = None
    best_obj = np.inf

    # Iterate over all possible active sets (combinations of constraints)
    for k in range(m + 1):
        for active_indices in itertools.combinations(range(m), k):
            # The number of active constraints should not exceed n
            if len(active_indices) > n or len(active_indices) == 0:
                continue

            C_A = C[list(active_indices), :]  # Active constraint matrix
            d_A = d[list(active_indices)]
            try:
                KKT_matrix = np.block([
                    [H, C_A.T],
                    [C_A, np.zeros((len(active_indices), len(active_indices)))]
                ])
                KKT_rhs = np.hstack([h, d_A])

                # Solve the KKT system
                solution = np.linalg.solve(KKT_matrix, KKT_rhs)
                x_star = solution[:n]
                lambda_star = solution[n:]

                # Check primal feasibility: Cx <= d
                if np.all(C @ x_star - d <= feasibility_tolerance):
                    # Check dual feasibility: lambda >= 0 for active constraints
                    if np.all(lambda_star >= -lambda_tolerance):
                        # Compute objective
                        obj_val = objective(x_star)
                        if obj_val < best_obj:
                            best_obj = obj_val
                            best_x = x_star
                            #lambda_full = np.zeros(m)
                            #lambda_full[list(active_indices)] = lambda_star

            except np.linalg.LinAlgError:
                # Singular matrix, skip this active set
                continue

    return best_x


def squared_distance(x, y):
    """ Returns squared Euclidean distance. """
    d = x - y
    return np.dot(d, d)


def check_binding(constraint, evaluate_constraint, pareto_set, tol):
    """
    Check if a constraint is binding based on the Pareto set of the problem with this constraint removed.
    The constraint is evaluated at point x using evaluate_constraint(x, constraint), and the binding condition is
    checked with the tolerance tol.
    """
    for pt in pareto_set:
        if evaluate_constraint(pt, constraint) > tol:
            return True
    return False


def rectangles_axis_intersect(p1_a, p1_b, p2_a, p2_b, distance):
    """ Checks whether two axis-aligned rectangles (p1_a, p1_b) and (p2_a, p2_b) overlap along at least one axis by more than the specified distance. """
    r1_x_min, r1_x_max = min(p1_a[0], p1_b[0]), max(p1_a[0], p1_b[0])
    r1_y_min, r1_y_max = min(p1_a[1], p1_b[1]), max(p1_a[1], p1_b[1])

    r2_x_min, r2_x_max = min(p2_a[0], p2_b[0]), max(p2_a[0], p2_b[0])
    r2_y_min, r2_y_max = min(p2_a[1], p2_b[1]), max(p2_a[1], p2_b[1])

    x_overlap_len = min(r1_x_max, r2_x_max) - max(r1_x_min, r2_x_min)
    y_overlap_len = min(r1_y_max, r2_y_max) - max(r1_y_min, r2_y_min)

    return x_overlap_len > distance or y_overlap_len > distance


def count_curves_agglomerative(points, distance_threshold=0.05):
    """ Counts the number of connected curves using agglomerative clustering. """
    clustering = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=distance_threshold)
    labels = clustering.fit_predict(points)
    num_clusters = len(set(labels))
    return num_clusters


def load_problem(filename):
    """ Loads the saved CobiProblem with computed results from the specified file. """
    with open(filename, 'rb') as f:
        problem = pickle.load(f)


    # Ensure all objectives and constraints have transformation fields (backward compatibility)

    if hasattr(problem, "alpha"):
        alpha = problem.alpha
        for ix, obj in enumerate(problem.objectives):
            if obj.get("transformation") is None and alpha[ix] != 1:
                obj["transformation"] = {"name": "exponent", "params": {"exponent": alpha[ix]}}
        delattr(problem, "alpha")
    for obj in getattr(problem, "objectives", []):
        if "alphas" in obj:
            if obj.get("peak_transformations") is None:
                obj["peak_transformations"] = [{"name": "exponent", "params": {"exponent": a}} for a in obj["alphas"]]
            del obj["alphas"]
    for obj in getattr(problem, "objectives", []):
        obj.setdefault("transformation", None)
        obj.setdefault("peak_transformations", [None] * len(obj["H"]))

    constraints = getattr(problem, "constraints", {})
    old_boundary_constraints = constraints.pop("Boundary", [])
    if not hasattr(problem, "boundary_constraints"):
        problem.boundary_constraints = len(old_boundary_constraints) > 0
    for key in ["Linear", "Quadratic", "Multi"]:
        constraints.setdefault(key, [])
    for key in ["Linear", "Quadratic"]:
        for constr in constraints[key]:
            constr.setdefault("transformation", None)
    for multi in constraints["Multi"]:
        for group in multi:
            group.setdefault("Linear", [])
            group.setdefault("Quadratic", [])
            for constr in group["Linear"]:
                constr.setdefault("transformation", None)
            for constr in group["Quadratic"]:
                constr.setdefault("transformation", None)
    problem.constraints = constraints
    problem.boundary_linear_constraints = create_boundary_linear_constraints(
        problem.n_var, problem.xl, problem.xu
    ) if problem.boundary_constraints else []

    if hasattr(problem, "_initial_state") and isinstance(problem._initial_state, dict):
        initial_constraints = problem._initial_state.get("constraints", {})
        old_initial_boundary = initial_constraints.pop("Boundary", None)
        if "boundary_constraints" not in problem._initial_state:
            problem._initial_state["boundary_constraints"] = (
                len(old_initial_boundary) > 0 if old_initial_boundary is not None else problem.boundary_constraints
            )
        problem._initial_state["constraints"] = initial_constraints
        problem._initial_state["boundary_linear_constraints"] = create_boundary_linear_constraints(
            problem.n_var, problem.xl, problem.xu
        ) if problem._initial_state["boundary_constraints"] else []

    if getattr(problem, "objectives", None) is not None:
        f_min = np.array([
            np.min(problem.objectives[0]['b']),
            np.min(problem.objectives[1]['b'])
        ])
        for ix, obj in enumerate(problem.objectives):
            tr = obj.get("transformation")
            if tr is not None and tr.get("name") in ["exponent", "logarithm"]:
                tr["params"]["shift"] = f_min[ix]
            for j, pt in enumerate(obj.get("peak_transformations", [])):
                if pt is not None and pt.get("name") in ["exponent", "logarithm"]:
                    pt["params"]["shift"] = obj["b"][j]

    return problem


def validate_transformation(tr, context=""):
    """ Check that transformation is a dictionary with name and params. """
    if tr is None:
        return
    if not isinstance(tr, dict):
        raise ValueError(f"{context} transformation must be a dictionary.")
    if "name" not in tr or "params" not in tr:
        raise ValueError(f"{context} transformation must contain name and params.")
    if not isinstance(tr["params"], dict):
        raise ValueError(f"{context} transformation params must be a dictionary.")


class CobiProblem(ElementwiseProblem):
    """
    A COnstrained BI-objective optimization problem. Inherits from ElementwiseProblem.

    The problem is defined as:

        min_x (
            T1(min_i [T1_i(0.5 * (x - c1_i)^T H1_i (x - c1_i) + b1_i)]),
            T2(min_j [T2_j(0.5 * (x - c2_j)^T H2_j (x - c2_j) + b2_j)])
        )

    subject to linear, convex-quadratic, and multi-constraints, each optionally transformed using a sign-preserving transformation.
    T1 and T2 are strictly increasing transformations applied to the objectives. T1_i and T2_j are strictly increasing transformations applied to the individual peaks.

    Constraints are defined as:
        - Linear:               <x - P, n> <= 0
        - Convex-quadratic:     (x - c)^T H (x - c) <= b, where H is symmetric positive definite
        - Multi-constraints:    min_k [max_l [g_{k,l}]] <= 0, where g_{k,l} are linear or convex-quadratic constraints

    Parameters:
        - n_var (int): Number of decision variables (dimension of the search space).
        - objectives (tuple of dict): Pair of dictionaries describing the two objective functions.
            - Each dictionary contains: H, c, b, and optionally transformation (for the objective) and peak_transformations (for each individual peak) describing the objective.
            - H must be symmetric positive definite.
        - constraints (dict): Dictionary containing lists of constraints for each group: Linear, Quadratic, Multi.
            - Linear: list of dictionaries with keys P, n.
            - Quadratic: list of dictionaries with keys c, H, b.
            - Multi: list of multi-constraints, each represented as
                [{'Linear': linear_constraints_1, 'Quadratic': quadratic_constraints_1}, ..., {'Linear': linear_constraints_v, 'Quadratic': quadratic_constraints_v}],
            where linear_constraints_k and quadratic_constraints_k are lists representing corresponding linear or convex-quadratic constraints in [g_{k,1}, ..., g_{k,u}].
            - Each constraint can optionally contain a transformation.
        - domain (tuple of float): Tuple (min, max) specifying lower and upper bounds for all decision variables.
        - alpha (float or tuple of float): Tuple (alpha_1, alpha_2) representing transformation parameters for the objective functions. Can also be a single float for both objectives.
            This parameter is deprecated. Use transformation in each objective instead.
        - boundary_constraints (bool): If True, automatically adds boundary constraints for each decision variable, ensuring that constraint violations reflect the domain.

    Attributes:
        - objectives, constraints: Provided objectives and constraints.
        - normalization_constant, normalization_divisor: Used for solution normalization if set.
        - pareto_set, pareto_front: Computed Pareto set and front.
        - uncon_pareto_set, uncon_pareto_front: Computed unconstrained Pareto set and front.
        - num_solver_failed: Count of points the solver failed to project to feasible solutions during computation of Pareto set and front.
        - total_points_error: Count of points generated by the error sampling method (if used).
        - rectangles: Stores rectangle regions for visualization (if used).
        - active_constraints: Stores currently active constraints (constraints on which at least one Pareto point lies).
        - local_unconstrained_pareto_sets/fronts, local_pareto_sets/fronts: Local (unconstrained) Pareto sets and fronts.
        - sampling_options: Options for point sampling used to compute the Pareto set and front.
        - _hypervolume, _normalized_hypervolume: Cached hypervolume metrics for the problem.
        - _initial_state: Initial state of the problem.
    """
    def __init__(
        self,
        n_var: int,
        objectives: Tuple[dict, dict],
        constraints: dict[str, list],
        domain: Tuple[float, float] = (-5, 5),
        alpha: Union[float, Tuple[float, float]] = (1, 1),
        boundary_constraints: bool = True
    ) -> None:
        # Parameter checks

        # Check n_var
        if not isinstance(n_var, int) or n_var <= 0:
            raise ValueError("n_var must be a positive integer.")
        
        # Alpha
        if not (
            isinstance(alpha, (int, float)) or
            (isinstance(alpha, (tuple, list)) and len(alpha) == 2)
        ):
            raise ValueError("alpha must be a scalar or a tuple/list of length 2.")

        # Check objectives
        if not isinstance(objectives, (list, tuple)) or len(objectives) != 2:
            raise ValueError("objectives must be a list or tuple of length 2.")

        for ix, obj in enumerate(objectives):

            if not isinstance(obj, dict):
                raise ValueError("Each objective must be a dictionary.")

            for key in ["H", "c", "b"]:
                if key not in obj:
                    raise ValueError(f"Each objective dictionary must contain {key}.")

            # H
            H = obj["H"]

            if isinstance(H, list):
                if not all(isinstance(h, np.ndarray) and h.ndim == 2 for h in H):
                    raise ValueError("H must be a list of 2D numpy arrays.")
                H = np.stack(H)
                obj["H"] = H

            if not isinstance(H, np.ndarray) or H.ndim != 3:
                raise ValueError("H must be a 3D numpy array with shape (n_peaks, n_var, n_var).")

            n_H = H.shape[0]
            matrix_dim = H.shape[1]

            if H.shape[1] != H.shape[2]:
                raise ValueError("All matrices in H must be square.")

            if matrix_dim != n_var:
                raise ValueError("H matrices must have shape (n_var, n_var).")
            
            for i in range(n_H):
                if not check_spd(H[i]):
                    raise ValueError(f"Objective H[{i}] must be symmetric positive definite.")

            # c
            c = obj["c"]

            if isinstance(c, list):
                if not all(isinstance(ci, np.ndarray) and ci.ndim == 1 for ci in c):
                    raise ValueError("c must be a list of 1D numpy arrays.")
                c = np.stack(c)
                obj["c"] = c

            if not isinstance(c, np.ndarray) or c.ndim != 2:
                raise ValueError("c must be a 2D numpy array with shape (n_peaks, n_var).")

            if c.shape != (n_H, n_var):
                raise ValueError("c must have shape (n_peaks, n_var).")

            # b
            b = obj["b"]

            if isinstance(b, list):
                b = np.array(b)
                obj["b"] = b

            if not isinstance(b, np.ndarray) or b.ndim != 1:
                raise ValueError("b must be a 1D numpy array.")

            if b.shape[0] != n_H:
                raise ValueError("b must have length n_peaks.")

            # alphas
            if "alphas" in obj:
                alphas = obj["alphas"]

                if isinstance(alphas, list):
                    alphas = np.array(alphas)

                if not isinstance(alphas, np.ndarray) or alphas.ndim != 1:
                    raise ValueError("alphas must be a 1D numpy array.")

                if alphas.shape[0] != n_H:
                    raise ValueError("alphas must have length n_peaks.")
            
            # transformation
            transformation = obj.get("transformation", None)

            if transformation is not None:
                validate_transformation(transformation, "Objective")
                if alpha[ix] != 1:
                    raise ValueError("Cannot provide both alpha and transformation. alpha is deprecated.")
            elif alpha[ix] != 1:
                obj["transformation"] = {"name": "exponent", "params": {"exponent": alpha[ix]}}
                warnings.warn("The alpha parameter is deprecated. Use transformation in each objective instead.", DeprecationWarning)

            # peak_transformations
            peak_transformations = obj.get("peak_transformations", None)

            if peak_transformations is not None:
                if len(peak_transformations) != n_H:
                    raise ValueError("peak_transformations must have length n_peaks.")
                for pt in peak_transformations:
                    validate_transformation(pt, f"Objective peak")
                if "alphas" in obj:
                    raise ValueError("Cannot provide both alphas and peak_transformations. alphas is deprecated.")
            elif "alphas" in obj:
                alphas = obj.pop("alphas")
                obj["peak_transformations"] = [
                    {"name": "exponent", "params": {"exponent": transformation_alpha}}
                    for transformation_alpha in alphas
                ]
                warnings.warn("Using alphas in objectives is deprecated. Use peak_transformations in each objective instead.", DeprecationWarning)

        # Check constraints
        if not isinstance(constraints, dict):
            raise ValueError("constraints must be a dictionary.")

        # Linear constraints
        if "Linear" in constraints:

            if not isinstance(constraints["Linear"], list):
                raise ValueError("Linear constraints must be a list.")

            for lin in constraints["Linear"]:

                if not isinstance(lin, dict):
                    raise ValueError("Each linear constraint must be a dictionary.")

                for key in ["P", "n"]:
                    if key not in lin:
                        raise ValueError(f"Each linear constraint must contain {key}.")
                    if not isinstance(lin[key], np.ndarray) or lin[key].ndim != 1:
                        raise ValueError(f"Linear constraint {key} must be a 1D numpy array.")

                if lin["P"].shape != lin["n"].shape:
                    raise ValueError("Linear constraint arrays P and n must have the same shape.")

                if lin["P"].shape[0] != n_var:
                    raise ValueError("Linear constraint vectors P and n must have length n_var.")
            
                if "transformation" in lin:
                    validate_transformation(lin["transformation"], "Linear constraint")
                else:
                    lin["transformation"] = None

        # Quadratic constraints
        if "Quadratic" in constraints:

            if not isinstance(constraints["Quadratic"], list):
                raise ValueError("Quadratic constraints must be a list.")

            for quad in constraints["Quadratic"]:

                if not isinstance(quad, dict):
                    raise ValueError("Each quadratic constraint must be a dictionary.")

                for key in ["H", "c", "b"]:
                    if key not in quad:
                        raise ValueError(f"Each quadratic constraint must contain {key}.")

                if not isinstance(quad["H"], np.ndarray) or quad["H"].ndim != 2:
                    raise ValueError("Quadratic constraint H must be a 2D numpy array.")

                if quad["H"].shape[0] != quad["H"].shape[1]:
                    raise ValueError("Quadratic constraint H must be square.")

                if quad["H"].shape[0] != n_var:
                    raise ValueError("Quadratic constraint H must have shape (n_var, n_var).")
                
                if not check_spd(quad["H"]):
                    raise ValueError("Quadratic constraint H must be symmetric positive definite.")

                if not isinstance(quad["c"], np.ndarray) or quad["c"].ndim != 1:
                    raise ValueError("Quadratic constraint c must be a 1D numpy array.")

                if quad["c"].shape[0] != n_var:
                    raise ValueError("Quadratic constraint c must have length n_var.")

                if not isinstance(quad["b"], (int, float)):
                    raise ValueError("Quadratic constraint b must be a scalar (int or float).")
                
                if "transformation" in quad:
                    validate_transformation(quad["transformation"], "Quadratic constraint")
                else:
                    quad["transformation"] = None

        # Multi-constraints
        if "Multi" in constraints:

            if not isinstance(constraints["Multi"], list):
                raise ValueError("Multi constraints must be a list.")

            for multi in constraints["Multi"]:

                if not isinstance(multi, list):
                    raise ValueError("Each multi-constraint must be a list of constraint groups.")

                for group in multi:

                    if not isinstance(group, dict):
                        raise ValueError("Each constraint group in Multi must be a dictionary.")

                    group.setdefault("Linear", [])
                    group.setdefault("Quadratic", [])

                    if not isinstance(group["Linear"], list):
                        raise ValueError("Each Linear in multi-constraint must be a list.")

                    if not isinstance(group["Quadratic"], list):
                        raise ValueError("Each Quadratic in multi-constraint must be a list.")

                    # Linear
                    for lin in group["Linear"]:

                        if not isinstance(lin, dict):
                            raise ValueError("Each linear constraint in Multi must be a dictionary.")

                        for key in ["P", "n"]:
                            if key not in lin:
                                raise ValueError(f"Each linear constraint in Multi must contain {key}.")
                            if not isinstance(lin[key], np.ndarray) or lin[key].ndim != 1:
                                raise ValueError(f"Linear constraint {key} in Multi must be a 1D numpy array.")

                        if lin["P"].shape != lin["n"].shape:
                            raise ValueError("Linear constraint arrays P and n in Multi must have the same shape.")

                        if lin["P"].shape[0] != n_var:
                            raise ValueError("Linear constraint vectors P and n in Multi must have length n_var.")
                        
                        if "transformation" in lin:
                            validate_transformation(lin["transformation"], "Linear constraint in Multi")
                        else:
                            lin["transformation"] = None

                    # Quadratic
                    for quad in group["Quadratic"]:

                        if not isinstance(quad, dict):
                            raise ValueError("Each quadratic constraint in Multi must be a dictionary.")

                        for key in ["H", "c", "b"]:
                            if key not in quad:
                                raise ValueError(f"Each quadratic constraint in Multi must contain {key}.")

                        if not isinstance(quad["H"], np.ndarray) or quad["H"].ndim != 2:
                            raise ValueError("Quadratic constraint H in Multi must be a 2D numpy array.")

                        if quad["H"].shape[0] != quad["H"].shape[1]:
                            raise ValueError("Quadratic constraint H in Multi must be square.")

                        if quad["H"].shape[0] != n_var:
                            raise ValueError("Quadratic constraint H in Multi must have shape (n_var, n_var).")
                        
                        if not check_spd(quad["H"]):
                            raise ValueError("Quadratic constraint H in Multi must be symmetric positive definite.")

                        if not isinstance(quad["c"], np.ndarray) or quad["c"].ndim != 1:
                            raise ValueError("Quadratic constraint c in Multi must be a 1D numpy array.")

                        if quad["c"].shape[0] != n_var:
                            raise ValueError("Quadratic constraint c in Multi must have length n_var.")

                        if not isinstance(quad["b"], (int, float)):
                            raise ValueError("Quadratic constraint b in Multi must be a scalar (int or float).")
                        
                        if "transformation" in quad:
                            validate_transformation(quad["transformation"], "Quadratic constraint in Multi")
                        else:
                            quad["transformation"] = None
        
        if "Boundary" in constraints:
            raise ValueError("constraints should not contain Boundary constraints. Set boundary_constraints instead.")

        # Domain
        if not isinstance(domain, (tuple, list)) or len(domain) != 2:
            raise ValueError("domain must be a tuple or list of length 2.")

        if domain[0] >= domain[1]:
            raise ValueError("domain[0] must be less than domain[1].")

        # Boundary constraints
        if not isinstance(boundary_constraints, bool):
            raise ValueError("boundary_constraints must be a boolean.")


        # Initialization

        self.objectives = copy.deepcopy(objectives)
        for obj in self.objectives:
            if 'transformation' not in obj:
                obj['transformation'] = None
            if 'peak_transformations' not in obj:
                obj['peak_transformations'] = [None] * len(obj['H'])
    
        self.constraints = copy.deepcopy(constraints)
        for key in ['Linear', 'Quadratic', 'Multi']:
            if key not in self.constraints:
                self.constraints[key] = []
        self.boundary_constraints = boundary_constraints
        for key in ['Linear', 'Quadratic']:
            for constr in self.constraints[key]:
                if 'transformation' not in constr:
                    constr['transformation'] = None
        for multi_constraint in self.constraints['Multi']:
            for constraints_group in multi_constraint:
                if 'Linear' not in constraints_group:
                    constraints_group['Linear'] = []
                if 'Quadratic' not in constraints_group:
                    constraints_group['Quadratic'] = []
                for constr in constraints_group['Linear']:
                    if 'transformation' not in constr:
                        constr['transformation'] = None
                for constr in constraints_group['Quadratic']:
                    if 'transformation' not in constr:
                        constr['transformation'] = None
        n_boundary_constraints = 2 * n_var if boundary_constraints else 0
        n_constr = n_boundary_constraints + len(self.constraints['Linear']) + len(self.constraints['Quadratic']) + len(self.constraints['Multi'])
        super().__init__(n_var=n_var, n_obj=2, n_constr=n_constr, xl=domain[0], xu=domain[1])

        self.domain = domain
        self.boundary_linear_constraints = create_boundary_linear_constraints(
            n_var, self.xl, self.xu
        ) if boundary_constraints else []

        f_min = np.array([np.min(self.objectives[0]['b']), np.min(self.objectives[1]['b'])])
        for ix, obj in enumerate(self.objectives):
            if obj['transformation'] is not None and obj['transformation']['name'] in ['exponent', 'logarithm']:
                obj['transformation']['params']['shift'] = f_min[ix]
            for j, pt in enumerate(obj['peak_transformations']):
                if pt is not None and pt['name'] in ['exponent', 'logarithm']:
                    pt['params']['shift'] = obj['b'][j]

        self.normalization_constant = None  # If not None then self.normalization_constant and self.normalization_divisor are used to normalize solutions.
        self.normalization_divisor = None

        self.pareto_set = None
        self.pareto_front = None
        self.uncon_pareto_set = None
        self.uncon_pareto_front = None
        self.uncon_pareto_source = None
        self.local_unconstrained_pareto_sets = None
        self.local_unconstrained_pareto_fronts = None
        self.local_pareto_sets = None
        self.local_pareto_fronts = None

        self.last_projected_solution = None  # Last known projected solution (used for warm start).
        self.num_solver_failed = None  # The number of points not projected to a feasible solution by the solver.
        self.total_points_error = None  # The number of points generated by the error sampling method.
        self.rectangles = None
        self.sampling_options = None

        self._hypervolume = None
        self._normalized_hypervolume = None

        self._initial_state = copy.deepcopy(self.__dict__)

    def reset(self):
        """ Restores the problem to the exact state it had immediately after initialization. """
        initial_state = copy.deepcopy(self._initial_state)
        self.__dict__.clear()
        self.__dict__.update(initial_state)
        self._initial_state = initial_state

    def evaluate_objectives(self, x):
        """ Evaluates both objective functions at the point x. """
        f1 = np.min([transform_objective(peak_function(x, c, H, b), t) for c, b, H, t in zip(
            self.objectives[0]['c'], self.objectives[0]['b'], self.objectives[0]['H'],
            self.objectives[0]['peak_transformations'])], axis=0)
        f2 = np.min([transform_objective(peak_function(x, c, H, b), t) for c, b, H, t in zip(
            self.objectives[1]['c'], self.objectives[1]['b'], self.objectives[1]['H'],
            self.objectives[1]['peak_transformations'])], axis=0)
        f1 = transform_objective(f1, self.objectives[0]['transformation'])
        f2 = transform_objective(f2, self.objectives[1]['transformation'])
        transformed_f1_f2 = (f1, f2)
        if self.normalization_constant is not None and self.normalization_divisor is not None:
            transformed_f1_f2 = (transformed_f1_f2 - self.normalization_constant) / self.normalization_divisor
        return transformed_f1_f2
    
    def __call__(self, x):
        """ Evaluate x. """
        return self.evaluate_objectives(x)

    def evaluate_constraints(self, x):
        """ Evaluates all constraints at the point x. """
        all_constraints = []
        if self.boundary_constraints:
            all_constraints.extend(evaluate_boundary_constraints(x, self.xl, self.xu))
        all_constraints.extend(evaluate_linear_quadratic_constraints(x, self.constraints['Linear'], self.constraints['Quadratic']))
        for multi_constraint in self.constraints['Multi']:
            all_constraints.append(evaluate_multi_constraint(x, multi_constraint))
        return all_constraints

    def _evaluate(self, x, out, *args, **kwargs):
        """ Evaluates both objective functions and all constraints at the point x. """
        out["F"] = np.column_stack(self.evaluate_objectives(x))
        out["G"] = np.column_stack(self.evaluate_constraints(x)) if self.n_constr > 0 else np.array([])

    def choose_solver(self):
        """ Choose an appropriate solver. """
        if len(self.constraints['Quadratic']) == 0:
            for multi_constraint in self.constraints['Multi']:
                for constraints_group in multi_constraint:
                    if len(constraints_group['Quadratic']) > 0:
                        return 'scipy_COBYLA'
            return 'daqp'
        return 'scipy_COBYLA'

    def peak_pair_function(self, i, j, x):
        """ Evaluates a pair of single peak functions at the point x. """
        value = np.array([
            transform_objective(transform_objective(
                peak_function(x, self.objectives[0]['c'][i], self.objectives[0]['H'][i], self.objectives[0]['b'][i]),
                self.objectives[0]['peak_transformations'][i]), self.objectives[0]['transformation']),
            transform_objective(transform_objective(
                peak_function(x, self.objectives[1]['c'][j], self.objectives[1]['H'][j], self.objectives[1]['b'][j]),
                self.objectives[1]['peak_transformations'][j]), self.objectives[1]['transformation'])
        ])
        if self.normalization_constant is not None and self.normalization_divisor is not None:
            value = (value - self.normalization_constant) / self.normalization_divisor
        return value
    
    def violation_point(self, x):
        """ Computes the total constraint violation at the point x. """
        return np.sum(np.maximum(self.evaluate_constraints(x), 0))

    def join_multi_constraints(self):
        """ Joins self.constraints['Multi'] into one equivalent multi-constraint. """
        joint_multi_constraint = []
        for combination in itertools.product(*self.constraints['Multi']):
            joint_linear = []
            joint_quadratic = []
            for constraints_group in combination:
                if 'Linear' in constraints_group:
                    joint_linear.extend(constraints_group['Linear'])
                if 'Quadratic' in constraints_group:
                    joint_quadratic.extend(constraints_group['Quadratic'])

            joint_multi_constraint.append({
                'Linear': joint_linear,
                'Quadratic': joint_quadratic
            })
        return joint_multi_constraint

    def project_point_solver(self, H1, H2, c1, c2, linear_constraints, quadratic_constraints, w, tol_feasible, solver):
        """
        Solves the quadratic program:
            min_x w * (x - c1)^T H1 (x - c1) + (1 - w) * (x - c2)^T H2 (x - c2)
            subject to linear constraints Cx <= d and quadratic constraints (x - c_i)^T H_i (x - c_i) <= b_i for i=1,...,k
        using the solver.

        The kkt solver uses the KKT conditions without iterations.
        Only cvxpy solvers can handle quadratic constraints.

        Parameters:
        - H1, H2: (n x n) positive-definite matrices
        - c1, c2: (n,) vectors
        - linear_constraints: a list of linear constraints
        - quadratic_constraints: a list of quadratic constraints
        - w: scalar in [0, 1]
        - tol_feasible: the solution is considered feasible if its total violation does not exceed this tolerance
        - solver: a solver that will be used

        Returns:
            None: if no feasible optimal solution is found.
            Otherwise, returns:
                - x_opt: the solution vector returned by the solver
        """
        solver_linear_constraints = linear_constraints + self.boundary_linear_constraints
        Ps, ns = [c['P'] for c in solver_linear_constraints], [c['n'] for c in solver_linear_constraints]
        cs, Hs, bs = [c['c'] for c in quadratic_constraints], [c['H'] for c in quadratic_constraints], [c['b'] for c in quadratic_constraints]

        C = np.array(ns)
        if len(ns) > 0:
            d = np.sum(np.array(ns) * np.array(Ps), axis=1)
        else:
            d = np.array([])

        if solver == 'cvxpy_SCS':
            x = cp.Variable(H1.shape[0])
            objective = cp.Minimize(w * cp.quad_form(x - c1, H1) + (1 - w) * cp.quad_form(x - c2, H2))

            if len(C) > 0:
                constraints = [C @ x <= d]
            else:
                constraints = []
            constraints += [cp.quad_form(x - cs[i], Hs[i]) <= bs[i] for i in range(len(Hs))]
            problem = cp.Problem(objective, constraints)

            problem.solve(solver=cp.SCS, eps_abs=1e-12, eps_rel=1e-12, eps_infeas=1e-12, max_iters=1000000, warm_start=True)
            x_opt = x.value

        elif solver == 'scipy_COBYLA':
            def objective(x):
                return w * (x - c1).T @ H1 @ (x - c1) + (1 - w) * (x - c2).T @ H2 @ (x - c2)

            constraints = []
            for i in range(len(C)):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: d[i] - C[i] @ x
                })
            for i in range(len(Hs)):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: bs[i] - (x - cs[i]).T @ Hs[i] @ (x - cs[i])
                })

            x0 = self.last_projected_solution if self.last_projected_solution is not None else w * c1 + (1 - w) * c2

            res = minimize(
                objective,
                x0,
                method='COBYLA',
                constraints=constraints,
                options={
                    'tol': 1e-15,
                    'catol': tol_feasible / max(1, len(solver_linear_constraints) + len(quadratic_constraints)),
                    'maxiter': 1000000
                }
            )
            x_opt = res.x if res.success else None

        elif 0 < len(Hs):
            raise ValueError(
                f'The problem contains quadratic constraints, which are not supported by the solver {solver}. '
                f'The solvers cvxpy_SCS, scipy_COBYLA can handle quadratic constraints.')

        elif solver == 'kkt':
            x_opt = project_point(H1, H2, c1, c2, w, C, d)

        else:
            P = 2 * (w * H1 + (1 - w) * H2)
            q = -2 * (w * H1 @ c1 + (1 - w) * H2 @ c2)

            x_opt = solve_qp(P, q, C, d, solver=solver, initvals=self.last_projected_solution)

        self.last_projected_solution = x_opt
        if x_opt is None:
            return x_opt
        else:
            violation = self.violation_point(x_opt)
            if violation > tol_feasible:
                return None
            return x_opt

    def compute_and_project_point(self, H1, H2, c1, c2, linear_constraints, quadratic_constraints, t, tol_feasible, solver):
        """ Compute the point and, if necessary, its projection onto the feasible region. """
        pt = compute_point(H1, H2, c1, c2, t)
        bounds = (self.xl, self.xu) if self.boundary_constraints else None
        feas = check_linear_quadratic_constraints(pt, linear_constraints, quadratic_constraints, bounds=bounds)
        projected_pt = pt if feas else self.project_point_solver(H1, H2, c1, c2, linear_constraints,
                                                                 quadratic_constraints, t, tol_feasible, solver)
        if projected_pt is None and self.num_solver_failed is not None:
            self.num_solver_failed += 1
        return projected_pt

    def project_unconstrained_pareto_set(
            self, H1, H2, c1, c2,
            linear_constraints, quadratic_constraints,
            uncon_pareto_set, uncon_pareto_set_w, tol_feasible, solver):
        """
        Projects points from the local unconstrained Pareto set between peaks specified by (c1, H1) and (c2, H2) onto the feasible
        region defined by: linear constraints Cx <= d and quadratic constraints (x - c_i)^T H_i (x - c_i) <= b_i for i=1,...,k.


        Parameters:
        - H1, H2: (n x n) positive-definite matrices
        - c1, c2: (n,) vectors
        - linear_constraints
        - quadratic_constraints
        - uncon_pareto_set: the unconstrained Pareto set
        - uncon_pareto_set_w: the weights corresponding to the points in the unconstrained Pareto set
        - tol_feasible: the projected point is considered feasible if its total violation does not exceed this tolerance

        Returns:
        - ps: projected points
        - ws: weights of projected points
        """
        ps = []
        ws = []
        bounds = (self.xl, self.xu) if self.boundary_constraints else None
        for pt, w in zip(uncon_pareto_set, uncon_pareto_set_w):
            feas = check_linear_quadratic_constraints(pt, linear_constraints, quadratic_constraints, bounds=bounds)

            if feas:
                ps.append(pt)
                ws.append(w)

            else:
                projected_pt = self.project_point_solver(
                    H1, H2, c1, c2,
                    linear_constraints, quadratic_constraints,
                    w, tol_feasible, solver)
                if projected_pt is not None:
                    ps.append(projected_pt)
                    ws.append(w)
                else:
                    self.num_solver_failed += 1

        return ps, ws

    def calculate_pareto_set_and_front(self, sampling_options=None, tol_feasible=1e-8,
                                       skip_dominated=True, solver=None, print_output=False):
        """
        Sets the following attributes:
        - self.local_unconstrained_pareto_fronts:
            A dictionary storing the local unconstrained Pareto fronts for each pair of individual peaks.
        - self.local_unconstrained_pareto_sets:
            A dictionary storing the local unconstrained Pareto sets for each pair of individual peaks.
        - self.uncon_pareto_front:
            The global unconstrained Pareto front, aggregating all local unconstrained Pareto fronts.
        - self.uncon_pareto_set:
            The global unconstrained Pareto set, aggregating all local unconstrained Pareto sets.
        - self.uncon_pareto_source:
            Tracks the origin (peak indices and weights) of each point in the global unconstrained Pareto set.
        - self.local_pareto_fronts:
            A dictionary storing the local Pareto fronts after projection onto the feasible region for each pair of individual peaks and joint multi-constraint group.
        - self.local_pareto_sets:
            A dictionary storing the local Pareto sets after projection onto the feasible region for each pair of individual peaks and joint multi-constraint group.
        - self.pareto_front:
            The global Pareto front, aggregating all local Pareto fronts.
        - self.pareto_set:
            The global Pareto set, aggregating all local Pareto sets.
        - self.pareto_source:
            Tracks the origin (peak indices, weights, and joint multi-constraint group index) of each point in the global Pareto set.
        The attributes that are actually computed, and the accuracy of their values, depend on the chosen sampling and computation options.

        The variable self.num_solver_failed stores the number of points the solver failed to project to a feasible
        solution during execution of this function.
        The variable self.sampling_options stores the used sampling_options.
        The variable self.total_points_error stores the number of all added points during error minimization.
        The variable self.rectangles stores the final rectangles used by the rectangles sampling.

        Parameters:
        - sampling_options:
            A dictionary with the chosen sampling:
                - equi-w: sample n_points points along each unconstrained local Pareto set using equidistant weights, then compute their projections onto the feasible region
                - equi-uncon-x: sample points along each unconstrained local Pareto set such that the Euclidean distance between consecutive points approximately equals the chosen
                distance (uses bisection)
                - equi-x: the Euclidean distance between consecutive points in the Pareto set is approximately equal to the chosen distance (uses bisection)
                - equi-f: the Euclidean distance between consecutive points on the Pareto front is approximately equal to the chosen distance (uses bisection)
                - max-HV: sample points until the maximal possible theoretical hypervolume error is below max_error or the chosen number of points is reached. Each sampled point
                is chosen to minimize the difference between the current hypervolume and the maximal possible theoretical hypervolume
                - rectangles: identify the parts of the local Pareto sets that contribute to the global Pareto set, then sample points from each part using the method rectangles_sampling
                - edge: sample only edge points for each local Pareto set (useful for computing nadir and ideal points)
            and parameters:
                - n_points: The number of points to sample from each Pareto set between two individual peaks (used when
                sampling is equi-w, rectangles equi-w)
                - distance: The Euclidean distance between two consecutive sampled points (used when sampling is equi-uncon-x,
                equi-x, equi-f, rectangles equi-x, rectangles equi-f)
                - max_error: maximal allowed possible theoretical hypervolume error (used when sampling is max-HV)
                - rectangles_sampling: how to sample points on each local Pareto set (used when sampling is rectangles)
                    - equi-w: sample n_points points along each curve with equidistant weights
                    - equi-x: the Euclidean distance between consecutive projected points is equal to the chosen distance
                    - equi-f: the Euclidean distance between consecutive points on the Pareto front is equal to the chosen distance
                - max_points: if not None, sampling stops after this amount of points is reached in the Pareto set approximation, even if the error is not smaller than max_error
                (used when sampling is max-HV)
                - rectangles_min_distance: a rectangle whose diagonal is longer than this value may still be split. Two rectangles are considered intersecting if their projections to axes
                overlap by at least this distance
                - tol_distance: the tolerance used when computing the next point that should be distance away from the current
                point (used when sampling is equi-uncon-x, equi-x, equi-f, rectangles equi-x, rectangles equi-f)
                - tol_jump: the amount to move the parameter by when point computation fails, retried until a point is computed
                (used when sampling is equi-uncon-x, equi-x, equi-f, rectangles equi-x, rectangles equi-f)
                - max_iter: the maximum number of iterations for the bisection (used when sampling is equi-uncon-x, equi-x, equi-f, rectangles equi-x, rectangles equi-f)
                - force_equidistant: if true, the next point must be exactly the chosen distance from the previous point,
                otherwise, it is sufficient for the distance to be less than the chosen distance (used when sampling is
                equi-uncon-x, equi-x, equi-f, rectangles equi-x, rectangles equi-f)
                - always_compute_unconstrained: if true, compute unconstrained Pareto set and front, even when not required for computation of Pareto set and front
                (used when sampling is equi-x, equi-f, max-HV, rectangles)
        - tol_feasible: the projected point is considered feasible if its total violation does not exceed this tolerance
        - skip_dominated: if true, skips points that are already dominated by some point
        - solver (str or None): A solver that will be used for projection of the unconstrained Pareto set onto the feasible
        region. If None, an appropriate solver is automatically selected. Recommended solvers are daqp (when only linear constraints are
        present) and scipy_COBYLA or cvxpy_SCS (when quadratic constraints are also present). The kkt solver uses the KKT conditions without iterations.
        - print_output: if true, prints maximal possible theoretical hypervolume error during computation (used when sampling is max-HV, rectangles)
        or maximal squared length of diagonal of rectangle and number of completed and total parts of Pareto set
        (used when sampling is rectangles)

        Returns nothing.
        """
        # Unpack parameters

        default_values = {
            'sampling': 'max-HV',
            'n_points': 1000,
            'distance': 0.1,
            'max_error': 0.01,
            'max_points': 10000,
            'tol_distance': 1e-8,
            'tol_jump': 1e-3,
            'max_iter': 10000,
            'force_equidistant': False,
            'always_compute_unconstrained': False,
            'rectangles_sampling': 'equi-w',
            'rectangles_min_distance': 0.1
        }
        required_keys = {
            'equi-w':       ['n_points'],
            'equi-uncon-x': ['distance', 'tol_distance', 'tol_jump', 'max_iter', 'force_equidistant'],
            'equi-x':       ['distance', 'tol_distance', 'tol_jump', 'max_iter', 'force_equidistant', 'always_compute_unconstrained'],
            'equi-f':       ['distance', 'tol_distance', 'tol_jump', 'max_iter', 'force_equidistant', 'always_compute_unconstrained'],
            'max-HV':       ['max_error', 'max_points', 'always_compute_unconstrained'],
            'rectangles':   ['rectangles_sampling', 'rectangles_min_distance', 'n_points', 'distance', 'tol_distance', 'tol_jump', 'max_iter', 'force_equidistant', 'always_compute_unconstrained'],
            'edge':         []
        }
        if sampling_options is None:
            sampling_options = {}
        merged_options = default_values.copy()
        merged_options.update(sampling_options)
        sampling = sampling_options.get('sampling', default_values['sampling'])
        if sampling not in required_keys:
            raise ValueError(
                f"Invalid sampling option '{sampling}'."
                f"Expected one of: {list(required_keys.keys())}."
            )
        needed = required_keys[sampling]

        params = {}
        for key in needed:
            params[key] = sampling_options.get(key, default_values[key])
        
        n_points = params.get('n_points', None)
        distance = params.get('distance', None)
        max_error = params.get('max_error', None)
        max_points = params.get('max_points', None)
        tol_distance = params.get('tol_distance', None)
        tol_jump = params.get('tol_jump', None)
        max_iter = params.get('max_iter', None)
        force_equidistant = params.get('force_equidistant', None)
        always_compute_unconstrained = params.get('always_compute_unconstrained', None)
        rectangles_sampling = params.get('rectangles_sampling', None)
        rectangles_min_distance = params.get('rectangles_min_distance', None)

        if print_output:
            print(f"Computing Pareto set and front using {sampling} method.")


        # Initialization

        self.local_unconstrained_pareto_sets = {}
        self.local_unconstrained_pareto_fronts = {}
        uncon_pareto_set_and_front = get_mo_archive()
        self.local_pareto_sets = {}
        self.local_pareto_fronts = {}
        pareto_set_and_front = get_mo_archive()
        self.num_solver_failed = 0
        self.total_points_error = 0
        self.rectangles = []
        self.sampling_options = merged_options
        distance_squared = distance ** 2 if distance is not None else None
        
        if solver is None:
            solver = self.choose_solver()

        joint_multi_constraint = self.join_multi_constraints()

        if sampling == 'max-HV':
            queue = PriorityQueue()
            counter = 0
            total_error = 0
            secondary_pareto_set_and_front = get_mo_archive()  # Used to store good points and stop when the desired number of points is reached
        if sampling == 'rectangles':
            queue = PriorityQueue()
            counter = 0
            secondary_pareto_set_and_front = get_mo_archive()  # Used to store edge points
            global_pareto_set_parametrization = {}
            rectangles_min_distance_squared = rectangles_min_distance ** 2

        n_peaks1 = len(self.objectives[0]['c'])
        n_peaks2 = len(self.objectives[1]['c'])
        n_peaks = n_peaks1 * n_peaks2


        # Loop over pairs of peaks
        for i, (center_f1, Hessian_f1) in enumerate(zip(self.objectives[0]['c'], self.objectives[0]['H'])):
            for j, (center_f2, Hessian_f2) in enumerate(zip(self.objectives[1]['c'], self.objectives[1]['H'])):
                if print_output and sampling in ['equi-w', 'equi-uncon-x', 'equi-x', 'equi-f', 'edge']:
                    print(f'completed peak pairs: {i * n_peaks2 + j} / {n_peaks}')

                # Skip dominated local fronts
                if skip_dominated:
                    ideal_point = np.array([self.peak_pair_function(i, j, self.objectives[0]['c'][i])[0], self.peak_pair_function(i, j, self.objectives[1]['c'][j])[1]])
                    if pareto_set_and_front.dominates(ideal_point):
                        continue
                
                # Compute the unconstrained Pareto set and front approximation
                if sampling == 'equi-w':
                    unconstrained_pareto_points, unconstrained_pareto_points_w = get_unconstrained_pareto_set_linspace_weights(
                        Hessian_f1, Hessian_f2, center_f1, center_f2, n_points)
                elif sampling == 'equi-uncon-x' or (sampling in ['equi-x', 'equi-f'] and always_compute_unconstrained):
                    compute_point_fun = lambda l: compute_point(Hessian_f1, Hessian_f2, center_f1, center_f2, l)
                    distance_fun = lambda x, y: squared_distance(x, y)
                    unconstrained_pareto_points, unconstrained_pareto_points_w = get_pareto_set_bisection_weights(
                        distance_squared, compute_point_fun, distance_fun, 0, 1, tol_distance, tol_jump, max_iter, force_equidistant)
                elif sampling in ['equi-x', 'equi-f']:
                    unconstrained_pareto_points, unconstrained_pareto_points_w = [], []
                elif sampling in ['max-HV', 'edge', 'rectangles']:
                    unconstrained_pareto_points, unconstrained_pareto_points_w = np.array([center_f1, center_f2]), np.array([1, 0])
                self.local_unconstrained_pareto_sets.update({f'{i}-{j}': np.array(unconstrained_pareto_points)})
                unconstrained_pareto_front = np.array([self.evaluate_objectives(x) for x in unconstrained_pareto_points])
                self.local_unconstrained_pareto_fronts.update({f'{i}-{j}': np.array(unconstrained_pareto_front)})
                for x, f, w in zip(unconstrained_pareto_points, unconstrained_pareto_front, unconstrained_pareto_points_w):
                    uncon_pareto_set_and_front.add(f, info={'x': x, 'source': (i, j, w)})

                # Filter dominated points
                if skip_dominated and sampling in ['equi-w', 'equi-uncon-x']:
                    filtered_unconstrained_pareto_points = []
                    filtered_unconstrained_pareto_points_w = []
                    for x, w, f in zip(unconstrained_pareto_points, unconstrained_pareto_points_w, unconstrained_pareto_front):
                        if not pareto_set_and_front.dominates(f):
                            filtered_unconstrained_pareto_points.append(x)
                            filtered_unconstrained_pareto_points_w.append(w)
                else:
                    filtered_unconstrained_pareto_points = unconstrained_pareto_points
                    filtered_unconstrained_pareto_points_w = unconstrained_pareto_points_w

                # Loop over joint multi-constraint groups
                for k, constraints in enumerate(joint_multi_constraint):

                    # Skip dominated local fronts
                    if skip_dominated:
                        pareto_points, _ = self.project_unconstrained_pareto_set(
                            Hessian_f1, Hessian_f2, center_f1, center_f2,
                            self.constraints['Linear'] + constraints['Linear'],
                            self.constraints['Quadratic'] + constraints['Quadratic'],
                            [center_f1, center_f2], [1, 0],
                            tol_feasible, solver)
                        if len(pareto_points) == 2:
                            ideal_point = np.array([self.peak_pair_function(i, j, pareto_points[0])[0], self.peak_pair_function(i, j, pareto_points[1])[1]])
                            if pareto_set_and_front.dominates(ideal_point):
                                continue
                    
                    if sampling in ['equi-x', 'equi-f']:
                        # Compute Pareto set approximation using bisection
                        if sampling == 'equi-x':
                            distance_fun = lambda x, y: squared_distance(x, y)
                        elif sampling == 'equi-f':
                            distance_fun = lambda x, y: squared_distance(self.evaluate_objectives(x), self.evaluate_objectives(y))
                        compute_and_project_point = lambda l: self.compute_and_project_point(
                            Hessian_f1, Hessian_f2, center_f1, center_f2,
                            self.constraints['Linear'] + constraints['Linear'],
                            self.constraints['Quadratic'] + constraints['Quadratic'],
                            l, tol_feasible, solver)
                        pareto_points, pareto_points_w = get_pareto_set_bisection_weights(
                            distance_squared, compute_and_project_point, distance_fun, 0, 1, tol_distance, tol_jump, max_iter, force_equidistant)
                        if self.n_constr > 0 and not always_compute_unconstrained:
                            unconstrained_pareto_points = [compute_point(Hessian_f1, Hessian_f2, center_f1, center_f2, w) for w in pareto_points_w]
                            unconstrained_pareto_front = np.array([self.evaluate_objectives(x) for x in unconstrained_pareto_points])
                            for x, f, w in zip(unconstrained_pareto_points, unconstrained_pareto_front, pareto_points_w):
                                uncon_pareto_set_and_front.add(f, info={'x': x, 'source': (i, j, w)})
                    else:
                        # Compute Pareto set approximation using weights
                        pareto_points, pareto_points_w = self.project_unconstrained_pareto_set(
                            Hessian_f1, Hessian_f2, center_f1, center_f2,
                            self.constraints['Linear'] + constraints['Linear'],
                            self.constraints['Quadratic'] + constraints['Quadratic'],
                            filtered_unconstrained_pareto_points, filtered_unconstrained_pareto_points_w,
                            tol_feasible, solver)
                        if sampling in ['max-HV', 'rectangles']:
                            if len(pareto_points) == 2:
                                # Add pair to queue and to Pareto set and front
                                y1 = self.peak_pair_function(i, j, pareto_points[0])
                                y1_true = self.evaluate_objectives(pareto_points[0])
                                y2 = self.peak_pair_function(i, j, pareto_points[1])
                                y2_true = self.evaluate_objectives(pareto_points[1])
                                ideal_rect = np.minimum(y1, y2)
                                if pareto_set_and_front.dominates(ideal_rect):
                                    continue
                                if sampling == 'max-HV':
                                    error = abs(y2[0] - y1[0]) * abs(y1[1] - y2[1])
                                    queue.put((-error, counter, (i, j, k, 1, 0, y1, y2)))
                                    self.total_points_error += 2
                                    total_error += error
                                elif sampling == 'rectangles':
                                    error = squared_distance(y1, y2)
                                    intersects = []
                                    rectangle = (i, j, k, 1, 0, y1, y2, intersects)
                                    for _, _, other in queue.queue:
                                        _, _, _, _, _, oy1, oy2, other_intersects = other
                                        if rectangles_axis_intersect(y1, y2, oy1, oy2, rectangles_min_distance):
                                            intersects.append(other)
                                            other_intersects.append(rectangle)
                                    queue.put((-error, counter, rectangle))
                                counter += 1
                                good1 = np.all(y1 == y1_true)
                                pareto_set_and_front.add(y1, info={'x': pareto_points[0], 'good': good1, 'source': (i, j, k, 1)})
                                if good1:
                                    secondary_pareto_set_and_front.add(y1, info={'x': pareto_points[0], 'source': (i, j, k, 1)})
                                good2 = np.all(y2 == y2_true)
                                pareto_set_and_front.add(y2, info={'x': pareto_points[1], 'good': good2, 'source': (i, j, k, 0)})
                                if good2:
                                    secondary_pareto_set_and_front.add(y2, info={'x': pareto_points[1], 'source': (i, j, k, 0)})
                            elif len(pareto_points) == 1:
                                # Add to Pareto set and front
                                y = self.peak_pair_function(i, j, pareto_points[0])
                                y_true = self.evaluate_objectives(pareto_points[0])
                                if sampling == 'max-HV':
                                    self.total_points_error += 1
                                good = np.all(y == y_true)
                                pareto_set_and_front.add(y, info={'x': pareto_points[0], 'good': good, 'source': (i, j, k, 0)})
                                if good:
                                    secondary_pareto_set_and_front.add(y, info={'x': pareto_points[0], 'source': (i, j, k, 0)})
                    
                    # Add to Pareto set and front
                    self.local_pareto_sets.update({f'{k}:{i}-{j}': np.array(pareto_points)})
                    pareto_front = np.array([self.evaluate_objectives(x) for x in pareto_points])
                    self.local_pareto_fronts.update({f'{k}:{i}-{j}': np.array(pareto_front)})
                    if sampling in ['equi-w', 'equi-uncon-x', 'equi-x', 'equi-f', 'edge']:
                        for x, f, w in zip(pareto_points, pareto_front, pareto_points_w):
                            pareto_set_and_front.add(f, info={'x': x, 'source': (i, j, k, w)})

        # max-HV method
        if sampling == 'max-HV':
            while total_error > max_error and (max_points is None or len(secondary_pareto_set_and_front) < max_points) and queue.qsize() > 0:
                if print_output:
                    print(f"maximal possible hypervolume error: {total_error:12.6g}, size of Pareto set: {len(secondary_pareto_set_and_front):12d}")
                
                neg_error, _, (i, j, k, weight1, weight2, y1, y2) = queue.get()
                Hessian_f1, Hessian_f2, center_f1, center_f2 = self.objectives[0]['H'][i], self.objectives[1]['H'][j], self.objectives[0]['c'][i], self.objectives[1]['c'][j]

                # Compute the middle point and add it to Pareto set and front
                weight_middle = (weight1 + weight2) / 2
                if weight_middle == weight1 or weight_middle == weight2:
                    continue  # skip, do not decrease error
                uncon_point_middle = compute_point(Hessian_f1, Hessian_f2, center_f1, center_f2, weight_middle)
                feas = check_linear_quadratic_constraints(
                    uncon_point_middle,
                    self.constraints['Linear'] + joint_multi_constraint[k]['Linear'],
                    self.constraints['Quadratic'] + joint_multi_constraint[k]['Quadratic'],
                    bounds=(self.xl, self.xu) if self.boundary_constraints else None
                )
                if feas:
                    point_middle = uncon_point_middle
                else:
                    pareto_points, _ = self.project_unconstrained_pareto_set(
                        Hessian_f1, Hessian_f2, center_f1, center_f2,
                        self.constraints['Linear'] + joint_multi_constraint[k]['Linear'],
                        self.constraints['Quadratic'] + joint_multi_constraint[k]['Quadratic'],
                        [uncon_point_middle], [weight_middle],
                        tol_feasible, solver)
                    if len(pareto_points) == 0:
                        continue
                    point_middle = pareto_points[0]
                self.total_points_error += 1
                y_middle = self.peak_pair_function(i, j, point_middle)
                y_middle_true = self.evaluate_objectives(point_middle)
                good = np.all(y_middle_true == y_middle)
                pareto_set_and_front.add(y_middle, info={'x': point_middle, 'good': good, 'source': (i, j, k, weight_middle)})
                if good:
                    secondary_pareto_set_and_front.add(y_middle)

                error = -neg_error
                total_error -= error

                # Split the rectangle with the middle point
                ideal_point1 = np.min(np.array([y1, y_middle]), axis=0)
                if not pareto_set_and_front.dominates(ideal_point1):
                    error1 = abs(y_middle[0] - y1[0]) * abs(y1[1] - y_middle[1])
                    total_error += error1
                    queue.put((-error1, counter, (i, j, k, weight1, weight_middle, y1, y_middle)))
                    counter += 1
                
                ideal_point2 = np.min(np.array([y_middle, y2]), axis=0)
                if not pareto_set_and_front.dominates(ideal_point2):
                    error2 = abs(y2[0] - y_middle[0]) * abs(y_middle[1] - y2[1])
                    total_error += error2
                    queue.put((-error2, counter, (i, j, k, weight_middle, weight2, y_middle, y2)))
                    counter += 1
            
            if print_output:
                print(f"maximal possible hypervolume error: {total_error:12.6g}, size of Pareto set: {len(secondary_pareto_set_and_front):12d}")

        # rectangles method
        elif sampling == 'rectangles':
            # Split rectangles until they are small or no longer intersect
            while queue.qsize() > 0:
                neg_squared_distance, _, rectangle = queue.get()
                i, j, k, weight1, weight2, y1, y2, intersects = rectangle
                Hessian_f1, Hessian_f2, center_f1, center_f2 = self.objectives[0]['H'][i], self.objectives[1]['H'][j], self.objectives[0]['c'][i], self.objectives[1]['c'][j]

                for other in intersects:
                    other_intersects = other[-1]
                    if rectangle in other_intersects:
                        other_intersects.remove(rectangle)

                ideal_rect = np.minimum(y1, y2)
                if pareto_set_and_front.dominates(ideal_rect):
                    continue

                rect_squared_distance = -neg_squared_distance

                if print_output:
                    print(f'squared length of the longest rectangle diagonal: {rect_squared_distance:12.6g}')
                
                if rect_squared_distance < rectangles_min_distance_squared or len(intersects) == 0:
                    # Add information from the rectangle to the parametrization of the global Pareto set
                    new1 = weight1  # new1 > new2
                    new2 = weight2

                    key = (i, j, k)
                    if key not in global_pareto_set_parametrization:
                        global_pareto_set_parametrization[key] = []
                    
                    intervals = []
                    for a, b in global_pareto_set_parametrization[key]:
                        if new2 <= a and b <= new1:
                            new1 = max(new1, a)
                            new2 = min(new2, b)
                        else:
                            intervals.append((a, b))
                    intervals.append((new1, new2))
                    global_pareto_set_parametrization[key] = intervals
                    
                    self.rectangles.append((i, j, k, y1, y2))
                    
                    continue

                # Compute the middle point and add it to Pareto set and front
                weight_middle = (weight1 + weight2) / 2
                if weight_middle == weight1 or weight_middle == weight2:
                    continue
                uncon_point_middle = compute_point(Hessian_f1, Hessian_f2, center_f1, center_f2, weight_middle)
                feas = check_linear_quadratic_constraints(
                    uncon_point_middle,
                    self.constraints['Linear'] + joint_multi_constraint[k]['Linear'],
                    self.constraints['Quadratic'] + joint_multi_constraint[k]['Quadratic'],
                    bounds=(self.xl, self.xu) if self.boundary_constraints else None
                )
                if feas:
                    point_middle = uncon_point_middle
                else:
                    pareto_points, _ = self.project_unconstrained_pareto_set(
                        Hessian_f1, Hessian_f2, center_f1, center_f2,
                        self.constraints['Linear'] + joint_multi_constraint[k]['Linear'],
                        self.constraints['Quadratic'] + joint_multi_constraint[k]['Quadratic'],
                        [uncon_point_middle], [weight_middle],
                        tol_feasible, solver)
                    if len(pareto_points) > 0:
                        point_middle = pareto_points[0]
                    else:
                        continue
                y_middle = self.peak_pair_function(i, j, point_middle)
                y_middle_true = self.evaluate_objectives(point_middle)
                good = np.all(y_middle_true == y_middle)
                pareto_set_and_front.add(y_middle, info={'x': point_middle, 'good': good, 'source': (i, j, k, weight_middle)})

                # Split the rectangle with the middle point
                ideal_point1 = np.min(np.array([y1, y_middle]), axis=0)
                if not pareto_set_and_front.dominates(ideal_point1):
                    error1 = squared_distance(y_middle, y1)
                    child_intersects1 = []
                    rectangle1 = (i, j, k, weight1, weight_middle, y1, y_middle, child_intersects1)
                    for other in intersects:
                        _, _, _, _, _, oy1, oy2, other_intersects = other
                        if rectangles_axis_intersect(y1, y_middle, oy1, oy2, rectangles_min_distance):
                            child_intersects1.append(other)
                            other_intersects.append(rectangle1)
                    queue.put((-error1, counter, rectangle1))
                    counter += 1
                
                ideal_point2 = np.min(np.array([y_middle, y2]), axis=0)
                if not pareto_set_and_front.dominates(ideal_point2):
                    error2 = squared_distance(y_middle, y2)
                    child_intersects2 = []
                    rectangle2 = (i, j, k, weight_middle, weight2, y_middle, y2, child_intersects2)
                    for other in intersects:
                        _, _, _, _, _, oy1, oy2, other_intersects = other
                        if rectangles_axis_intersect(y_middle, y2, oy1, oy2, rectangles_min_distance):
                            child_intersects2.append(other)
                            other_intersects.append(rectangle2)
                    queue.put((-error2, counter, rectangle2))
                    counter += 1
            
            # Compute the Pareto set and front from the parametrization
            total_parts = sum(len(p) for p in global_pareto_set_parametrization.values())
            completed_parts = 0
            pareto_set_and_front = secondary_pareto_set_and_front
            for (i, j, k) in global_pareto_set_parametrization:
                for (weight1, weight2) in global_pareto_set_parametrization[(i, j, k)]:  # weight1 > weight2
                    Hessian_f1, Hessian_f2, center_f1, center_f2 = self.objectives[0]['H'][i], self.objectives[1]['H'][j], self.objectives[0]['c'][i], self.objectives[1]['c'][j]

                    if print_output:
                        print(f'completed parts of the global Pareto set: {completed_parts} / {total_parts}')
                    completed_parts += 1
                    
                    if rectangles_sampling == 'equi-w':
                        # Compute the Pareto set and front using weights
                        weights = np.linspace(weight2, weight1, n_points)
                        unconstrained_points = [compute_point(Hessian_f1, Hessian_f2, center_f1, center_f2, w) for w in weights]
                        pareto_points, pareto_points_w = self.project_unconstrained_pareto_set(
                            Hessian_f1, Hessian_f2, center_f1, center_f2,
                            self.constraints['Linear'] + joint_multi_constraint[k]['Linear'],
                            self.constraints['Quadratic']
                            + joint_multi_constraint[k]['Quadratic'],
                            unconstrained_points,
                            weights,
                            tol_feasible,
                            solver
                        )
                    elif rectangles_sampling in ['equi-x', 'equi-f']:
                        # Compute the Pareto set and front using bisection
                        if rectangles_sampling == 'equi-x':
                            distance_fun = lambda x, y: squared_distance(x, y)
                        else:
                            distance_fun = lambda x, y: squared_distance(self.evaluate_objectives(x), self.evaluate_objectives(y))
                        compute_and_project_point = lambda l: self.compute_and_project_point(
                            Hessian_f1, Hessian_f2, center_f1, center_f2,
                            self.constraints['Linear'] + joint_multi_constraint[k]['Linear'],
                            self.constraints['Quadratic'] + joint_multi_constraint[k]['Quadratic'],
                            l, tol_feasible, solver)
                        pareto_points, pareto_points_w = get_pareto_set_bisection_weights(
                            distance_squared, compute_and_project_point, distance_fun, weight2, weight1, tol_distance, tol_jump, max_iter,
                            force_equidistant)

                    pareto_front = np.array([self.evaluate_objectives(x) for x in pareto_points])
                    for x, f, w in zip(pareto_points, pareto_front, pareto_points_w):
                        pareto_set_and_front.add(f, info={'x': x, 'source': (i, j, k, w)})


        # Set attributes

        self.pareto_set = np.array([list(d['x']) for d in pareto_set_and_front.infos if d.get('good', True)]) if len(pareto_set_and_front) > 0 else np.empty((0, 2))
        self.pareto_front = np.array([f for f, d in zip(pareto_set_and_front, pareto_set_and_front.infos) if d.get('good', True)]) if len(pareto_set_and_front) > 0 else np.empty((0, 2))
        self.pareto_source = np.array([list(d['source']) for d in pareto_set_and_front.infos if d.get('good', True)]) if len(pareto_set_and_front) > 0 else np.empty((0, 2))

        print_output_final = True
        if sampling in ['equi-w', 'equi-uncon-x', 'edge'] or (
                sampling in ['equi-x', 'equi-f'] and self.n_constr > 0 and not always_compute_unconstrained):
            self.uncon_pareto_front = np.array(uncon_pareto_set_and_front) if len(uncon_pareto_set_and_front) > 0 else np.empty((0, 2))
            self.uncon_pareto_set = np.array([list(d['x']) for d in uncon_pareto_set_and_front.infos]) if len(uncon_pareto_set_and_front) > 0 else np.empty((0, 2))
            self.uncon_pareto_source = np.array([list(d['source']) for d in uncon_pareto_set_and_front.infos]) if len(uncon_pareto_set_and_front) > 0 else np.empty((0, 2))
        elif self.n_constr == 0:
            self.uncon_pareto_set = self.pareto_set
            self.uncon_pareto_front = self.pareto_front
            self.uncon_pareto_source = self.pareto_source
        elif always_compute_unconstrained:
            # Compute the unconstrained Pareto set and front
            if print_output:
                print("Calculating unconstrained Pareto set and front")
            unconstrained_problem = CobiProblem(self.n_var, self.objectives, {'Linear': [], 'Quadratic': [], 'Multi': []}, self.domain, boundary_constraints=False)
            if self.normalization_constant is not None and self.normalization_divisor is not None:
                unconstrained_problem.normalization_constant = self.normalization_constant
                unconstrained_problem.normalization_divisor = self.normalization_divisor
            unconstrained_problem.calculate_pareto_set_and_front(sampling_options=sampling_options, tol_feasible=tol_feasible, skip_dominated=skip_dominated, print_output=print_output)
            self.uncon_pareto_set = unconstrained_problem.pareto_set
            self.uncon_pareto_front = unconstrained_problem.pareto_front
            self.uncon_pareto_source = unconstrained_problem.pareto_source
            print_output_final = False

        if print_output and print_output_final:
            print("Pareto set and front computed successfully.\n")

    def split_active_constraints(self, active_constraints):
        """ Splits active_constraints indices into sets of boundary, linear, quadratic, and multi constraint indices. Returns a dictionary containing them. """
        boundary_num = 2 * self.n_var if self.boundary_constraints else 0
        linear_num = len(self.constraints['Linear'])
        quadratic_num = len(self.constraints['Quadratic'])

        boundary_set = set()
        linear_set = set()
        quadratic_set = set()
        multi_set = set()

        for idx in active_constraints:
            if idx < boundary_num:
                boundary_set.add(idx + 1)  # Add 1 to make the indices in the output start at 1
            elif idx < boundary_num + linear_num:
                linear_set.add(idx - boundary_num + 1)
            elif idx < boundary_num + linear_num + quadratic_num:
                quadratic_set.add(idx - boundary_num - linear_num + 1)
            else:
                multi_set.add(idx - boundary_num - linear_num - quadratic_num + 1)
        return {'Boundary': boundary_set, 'Linear': linear_set, 'Quadratic': quadratic_set, 'Multi': multi_set}

    def get_active_constraints(self, x, tol_active):
        """ Computes the dictionary of active constraints for point x. A constraint g is considered active at x if abs(g(x)) < tol_active. """
        active_constraints = {int(i) for i in np.where(np.abs(self.evaluate_constraints(x)) < tol_active)[0]}
        return self.split_active_constraints(active_constraints)
    
    def get_active_constraints_pareto_set(self, tol_active):
        """
        Returns the active constraints at each point of the Pareto set. Requires computed Pareto set.
        A constraint g is considered active at x if abs(g(x)) < tol_active.
        """
        if self.pareto_set is None:
            raise RuntimeError("Could not run calculate_active_constraints, because it requires a computed Pareto set. Use calculate_pareto_set_and_front first.")
        return [self.get_active_constraints(pt, tol_active) for pt in self.pareto_set]

    def calculate_active_constraints(self, tol_active=1e-8):
        """
        Calculates which constraints are active, meaning they have at least one Pareto point satisfying abs(g(x)) < tol_active. Requires computed Pareto set.
        Returns a dictionary containing sets of indices for active boundary, linear, quadratic, and multi constraints.
        """
        if self.pareto_set is None:
            raise RuntimeError("Could not run calculate_active_constraints, because it requires a computed Pareto set. Use calculate_pareto_set_and_front first.")
        else:
            active_constraints = self.get_active_constraints_pareto_set(tol_active)
            boundary_set = set()
            linear_set = set()
            quadratic_set = set()
            multi_set = set()
            for ac in active_constraints:
                boundary_set |= ac['Boundary']
                linear_set |= ac['Linear']
                quadratic_set |= ac['Quadratic']
                multi_set |= ac['Multi']
            return {"Boundary": boundary_set, "Linear": linear_set, "Quadratic": quadratic_set, "Multi": multi_set}
    
    def nadir_point(self, *args, **kwargs):
        """ Computes the nadir point of the problem. """
        if self.pareto_front is None:
            self.calculate_pareto_set_and_front(sampling_options={'sampling': 'edge'})
        if len(self.pareto_front) == 0:
            raise RuntimeError("Nadir point cannot be computed, because Pareto front is empty. Can happen when the problem is infeasible.")
        return np.max(self.pareto_front, axis=0)

    def ideal_point(self, *args, **kwargs):
        """ Computes the ideal point of the problem. """
        if self.pareto_front is None:
            self.calculate_pareto_set_and_front(sampling_options={'sampling': 'edge'})
        if len(self.pareto_front) == 0:
            raise RuntimeError("Ideal point cannot be computed, because Pareto front is empty. Can happen when the problem is infeasible.")
        return np.min(self.pareto_front, axis=0)

    def normalize_problem(self):
        """ Resets the problem and normalizes it using the ideal and nadir points. Sets the normalization constant and divisor. """
        self.reset()
        ideal = self.ideal_point()
        nadir = self.nadir_point()
        self.normalization_constant = ideal
        self.normalization_divisor = nadir - ideal

        if np.any(self.normalization_divisor == 0):
            raise RuntimeError("Normalization divisor contains zero(s). Cannot normalize problem.")

    def _compute_hypervolume(self):
        """ Computes the non-normalized and normalized hypervolumes. Requires computed Pareto front. """
        if self.pareto_set is None:
            raise RuntimeError("Could not run _compute_hypervolume, because it requires a computed Pareto front. Use calculate_pareto_set_and_front first.")
        if len(self.pareto_front) == 0:
            raise RuntimeError("Could not run _compute_hypervolume, because Pareto front is empty. Can happen when the problem is infeasible.")
        ideal = self.ideal_point()
        nadir = self.nadir_point()
        # Compute the regular hypervolume first
        hv_archive = get_mo_archive(reference_point=nadir)
        hv_archive.add_list(self.pareto_front)
        self._hypervolume = float(hv_archive.hypervolume)
        # Then compute the normalized hypervolume
        hypervolume_max = np.prod(nadir - ideal)
        if hypervolume_max == 0:
            self._normalized_hypervolume = None
        else:
            self._normalized_hypervolume = self._hypervolume / hypervolume_max

    @property
    def hypervolume(self):
        """ Returns the approximated hypervolume of the Pareto front. """
        if self._hypervolume is None:
            self._compute_hypervolume()
        return self._hypervolume

    @property
    def normalized_hypervolume(self):
        """ Returns the approximated normalized hypervolume of the Pareto front. """
        if self._normalized_hypervolume is None:
            self._compute_hypervolume()
        return self._normalized_hypervolume

    def reduce_pareto_set_size(self, size):
        """
        Iteratively removes the point with the least contribution to the overall hypervolume until the desired number of points is reached.
        Requires computed Pareto set.
        """
        if self.pareto_set is None:
            raise RuntimeError("Could not run reduce_pareto_set_size, because it requires a computed Pareto set. Use calculate_pareto_set_and_front first.")
        elif size < len(self.pareto_set):
            nadir = self.nadir_point()
            pareto_set_and_front = get_mo_archive(reference_point=nadir)
            for x, f, s in zip(self.pareto_set, self.pareto_front, self.pareto_source):
                pareto_set_and_front.add(f, info={'x': x, 'source': s})
            while len(pareto_set_and_front) > size:
                hv_contributions = pareto_set_and_front.contributing_hypervolumes
                min_contributing_point = pareto_set_and_front[np.argmin(hv_contributions)]
                pareto_set_and_front.remove(min_contributing_point)
            self.reset()  # Reset the problem to avoid inconsistencies
            self.pareto_front = np.array(pareto_set_and_front)
            self.pareto_set = np.array([list(d['x']) for d in pareto_set_and_front.infos])
            self.pareto_source = np.array([list(d['source']) for d in pareto_set_and_front.infos])

    def is_feasible(self):
        """ Returns true if the problem is feasible and false otherwise. Requires computed Pareto set. """
        if self.pareto_set is None:
            raise RuntimeError("Could not run is_feasible, because it requires a computed Pareto set. Use calculate_pareto_set_and_front first.")
        else:
            return len(self.pareto_set) > 0

    def calculate_binding_constraints(self, tol=1e-6, **params):
        """
        Calculates which constraints are binding, meaning that the Pareto set changes when they are removed.
        Computes the Pareto set of each subproblem with a single constraint removed and checks if the Pareto set has changed.
        
        Parameters:
        - tol: the solution to the subproblem is considered new if it violates the original problem's constraint by
        more than this tolerance
        - params: parameters used by the calculate_pareto_set_and_front function to compute the Pareto set of the subproblem

        Returns a dictionary containing sets of indices for binding linear, quadratic, and multi-constraints.
        """
        binding_constraints = {
            "Linear": set(),
            "Quadratic": set(),
            "Multi": set()
        }

        constraint_info = [
            ("Linear", evaluate_linear_constraint),
            ("Quadratic", evaluate_quadratic_constraint),
            ("Multi", evaluate_multi_constraint),
        ]

        for name, evaluator in constraint_info:
            constr_list = self.constraints[name]
            for i in range(len(constr_list)):
                subproblem_constraints = copy.deepcopy(self.constraints)
                sub_constr_i = subproblem_constraints[name][i]
                del subproblem_constraints[name][i]
                subproblem_boundary_constraints = self.boundary_constraints
                subproblem = CobiProblem(self.n_var, self.objectives, subproblem_constraints, self.domain, (1, 1), subproblem_boundary_constraints)
                if self.normalization_constant is not None and self.normalization_divisor is not None:
                    subproblem.normalization_constant = self.normalization_constant
                    subproblem.normalization_divisor = self.normalization_divisor
                subproblem.calculate_pareto_set_and_front(**params)
                if check_binding(sub_constr_i, evaluator, subproblem.pareto_set, tol):
                    binding_constraints[name].add(i + 1)  # Add 1 to make the indices start at 1

        return binding_constraints

    def calculate_ps_pf_parts(self, dist_thresh_set=None, dist_thresh_front=None):
        """
        Calculates how many separate parts the Pareto set and front consist of, using agglomerative clustering with the distance thresholds dist_thresh_set
        and dist_thresh_front. The distance thresholds control how close points must be to be considered part of the same cluster.
        Requires computed Pareto set and front. If a threshold is not provided, the corresponding number of parts will be returned as None.
        """
        if self.pareto_set is None or self.pareto_front is None:
            raise RuntimeError("Could not run calculate_ps_pf_parts, because it requires computed Pareto set and front. Use calculate_pareto_set_and_front first.")
        elif len(self.pareto_set) == 0:
            return 0, 0
        else:
            if dist_thresh_set is None:
                ps_parts = None
            else:
                ps_parts = count_curves_agglomerative(self.pareto_set, distance_threshold=dist_thresh_set)

            if dist_thresh_front is None:
                pf_parts = None
            else:
                pf_parts = count_curves_agglomerative(self.pareto_front, distance_threshold=dist_thresh_front)

            return ps_parts, pf_parts

    def characterize_problem(self, tol_active=1e-8, dist_thresh_set=None, dist_thresh_front=None):
        """
        Returns a characterization of the problem. Requires computed Pareto set and front.

        Parameters:
        - tol_active: A constraint g is considered active if it has at least one Pareto point x satisfying abs(g(x)) < tol_active.
        - dist_thresh_set, dist_thresh_front: Distance thresholds for agglomerative clustering of the Pareto set and front.

        Returns a dictionary containing properties of the problem.
        """
        if self.pareto_set is None or self.pareto_front is None:
            raise RuntimeError ("Could not run characterize_problem, because it requires computed Pareto set and front. Use calculate_pareto_set_and_front first.")

        feasible = self.is_feasible()
        nadir = self.nadir_point() if feasible else None
        ideal = self.ideal_point() if feasible else None
        hypervolume = self.hypervolume if feasible else None
        normalized_hypervolume = self.normalized_hypervolume if feasible else None
        active_constraints = self.calculate_active_constraints(tol_active)
        ps_parts, pf_parts = self.calculate_ps_pf_parts(dist_thresh_set, dist_thresh_front)

        return {
            "feasible": feasible,
            "nadir": nadir,
            "ideal": ideal,
            "hypervolume": hypervolume,
            "normalized_hypervolume": normalized_hypervolume,
            "active_constraints": active_constraints,
            "pareto_set_parts": ps_parts,
            "pareto_front_parts": pf_parts
        }

    def save_problem(self, filename):
        """ Saves the problem with computed results to the specified file. """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def get_figure_1d(self, algorithm_X=None, algorithm_F=None, algorithm_name='Algorithm', algorithm_color='green', algorithm_point_size=6,
                      plot_objective_space=True, plot_search_space=True, plot_unconstrained_pareto=True, unconstrained_pareto_size=6,
                      plot_constrained_pareto=True, plot_normalized_front=False, normalize_algorithm=False,
                      color_peaks=False, plot_large_peak_centers=True, rasterized=True, fig_width=3.5, show_dimension_objective=True,
                      show_legend=True, show_title=True, center_constrained_front=True):
        """ Figure for problems with a one-dimensional search space. """

        # Determine number of plots
        num_plots = 2 if plot_objective_space and plot_search_space else 1
        _, axes = plt.subplots(1, num_plots, figsize=(fig_width * num_plots, fig_width))
        ax = axes if num_plots == 1 else axes[0]

        # Peak colors
        peak_color1 = peak_color2 = 'black'
        if color_peaks:
            peak_color1 = plt.get_cmap('Set1')(0)
            peak_color2 = plt.get_cmap('Accent')(4)

        # Prepare x-axis
        x_range = np.linspace(self.xl[0], self.xu[0], 500)

        # Plot search space
        if plot_search_space:
            # Plot function curves
            Z1 = [multi_peak_function([x], self.objectives[0]['c'], self.objectives[0]['H'], self.objectives[0]['b'])
                  for x in x_range]
            Z2 = [multi_peak_function([x], self.objectives[1]['c'], self.objectives[1]['H'], self.objectives[1]['b'])
                  for x in x_range]
            ax.plot(x_range, Z1, color='blue', alpha=0.3, linewidth=1, label='$f_1$')
            ax.plot(x_range, Z2, color='red', alpha=0.3, linewidth=1, label='$f_2$')

            # Plot peak centers
            if plot_large_peak_centers:
                ax.scatter(self.objectives[0]['c'][:, 0], np.zeros(len(self.objectives[0]['c'])), 
                        color=peak_color1, marker='x', s=100, rasterized=rasterized)
                ax.scatter(self.objectives[1]['c'][:, 0], np.zeros(len(self.objectives[1]['c'])), 
                        color=peak_color2, marker='+', s=140, rasterized=rasterized)
            else:
                ax.scatter(self.objectives[0]['c'][:, 0], np.zeros(len(self.objectives[0]['c'])), 
                        color=peak_color1, marker='x', s=40, rasterized=rasterized)
                ax.scatter(self.objectives[1]['c'][:, 0], np.zeros(len(self.objectives[1]['c'])), 
                        color=peak_color2, marker='+', s=56, rasterized=rasterized)

            # Plot unconstrained Pareto set
            if self.uncon_pareto_set is not None and plot_unconstrained_pareto:
                ax.scatter(self.uncon_pareto_set[:, 0], np.zeros(len(self.uncon_pareto_set)),
                        color='grey', s=unconstrained_pareto_size, label='Uncon. Pareto set',
                        zorder=2, rasterized=rasterized)
                
            # Plot constrained Pareto set
            if self.pareto_set is not None and plot_constrained_pareto:
                ax.scatter(self.pareto_set[:, 0], np.zeros(len(self.pareto_set)), 
                        color='black', s=10, label='Pareto set', zorder=2, rasterized=rasterized)
            
            # Plot algorithm solutions
            if algorithm_X is not None and len(algorithm_X) > 0:
                xs = [x[0] for x in algorithm_X]
                ax.scatter(xs, np.zeros(len(xs)), label=algorithm_name,
                           facecolors=algorithm_color, edgecolors='black', linewidths=0.03, marker='o',
                           s=algorithm_point_size, zorder=5, rasterized=rasterized)

            ax.set_xlabel('$x_1$', size='larger')
            ax.set_ylabel('f(x)', size='larger')
            if show_legend:
                ax.legend(fontsize='small')
            ax.set_xlim(self.xl[0], self.xu[0])
            ax.set_ylim(-0.5, 1)
            ax.grid(True, which='both', linestyle='--')

            if show_title:
                ax.set_title('Search space ($n=1$)')

        # Plot objective space
        if plot_objective_space:
            ax2 = axes[1] if plot_search_space else axes

            # Plot unconstrained Pareto front
            if self.uncon_pareto_front is not None and plot_unconstrained_pareto:
                upf = self.uncon_pareto_front
                if plot_normalized_front:
                    ideal = self.ideal_point()
                    nadir = self.nadir_point()
                    upf = (upf - ideal) / (nadir - ideal)
                ax2.scatter(upf[:, 0], upf[:, 1], s=unconstrained_pareto_size, color='grey', label='Uncon. Pareto front', rasterized=rasterized)

            # Plot constrained Pareto front
            if self.pareto_front is not None and plot_constrained_pareto:
                pf = self.pareto_front
                if plot_normalized_front:
                    ideal = self.ideal_point()
                    nadir = self.nadir_point()
                    pf = (pf - ideal) / (nadir - ideal)
                ax2.scatter(pf[:, 0], pf[:, 1], s=10, color='black', label='Pareto front', rasterized=rasterized)

            # Plot algorithm results
            if algorithm_F is not None and len(algorithm_F) > 0:
                af = algorithm_F
                if normalize_algorithm:
                    ideal = self.ideal_point()
                    nadir = self.nadir_point()
                    af = (af - ideal) / (nadir - ideal)
                ax2.scatter(af[:, 0], af[:, 1], label=algorithm_name,
                            facecolors=algorithm_color, edgecolors='black', linewidths=0.03, marker='o',
                            s=algorithm_point_size, zorder=5, rasterized=rasterized)

            if show_title:
                title = "Search space ($n=1$)"
                ax.set_title(title)
            ax2.set_xlabel(f'$f_1$', size='larger')
            ax2.set_ylabel(f'$f_2$', size='larger', rotation=0, labelpad=7)
            ax2.grid(True, which='both', linestyle='--')
            if show_legend:
                ax2.legend(fontsize='small')

            if plot_normalized_front:
                ax2.set_xlim(-0.1, 1.1)
                ax2.set_ylim(-0.1, 1.1)
                if show_title:
                    title = "Norm. objective space ($m=2$)" if show_dimension_objective else "Objective space"
                    ax2.set_title(title)
            elif center_constrained_front:
                pf = self.pareto_front
                pad = 0.1
                x_min, x_max = pf[:, 0].min(), pf[:, 0].max()
                y_min, y_max = pf[:, 1].min(), pf[:, 1].max()
                dx = (x_max - x_min) * pad
                dy = (y_max - y_min) * pad
                ax2.set_xlim(x_min - dx, x_max + dx)
                ax2.set_ylim(y_min - dy, y_max + dy)

            if show_title:
                title = "Objective space ($m=2$)" if show_dimension_objective else "Objective space"
                ax2.set_title(title)

        return axes

    def _filter_local_points_to_nondominated(self, local_sets, local_fronts):
        """ Keep only nondominated points within each local Pareto front. """
        if local_fronts is None:
            return local_sets, local_fronts

        filtered_fronts = {}
        filtered_sets = {} if local_sets is not None else None

        for key, local_front in local_fronts.items():
            if len(local_front) == 0:
                filtered_fronts[key] = local_front.copy()
                if local_sets is not None:
                    filtered_sets[key] = local_sets[key].copy()
                continue

            archive = get_mo_archive()

            if local_sets is not None:
                local_set = local_sets[key]
                if len(local_set) != len(local_front):
                    raise RuntimeError(
                        f"Local Pareto set/front size mismatch for {key}: "
                        f"{len(local_set)} != {len(local_front)}."
                    )
                for x, f in zip(local_set, local_front):
                    archive.add(f, info={'x': x})
                filtered_sets[key] = np.array([info['x'] for info in archive.infos])
            else:
                archive.add_list(local_front)

            filtered_fronts[key] = np.array(list(archive))

        return filtered_sets, filtered_fronts

    def get_figure(self, algorithm_X=None, algorithm_F=None, algorithm_name='Algorithm', algorithm_color='green', algorithm_point_size=6,
                   plot_objective_space=True, plot_search_space=True, plot_unconstrained_pareto=True, unconstrained_pareto_size=6,
                   plot_constrained_pareto=True, plot_normalized_front=False, normalize_algorithm=False, shade_infeasible_lin_quad=True, plot_rectangles=False,
                   color_peaks=False, plot_large_peak_centers=True, shade_infeasible_multi_constraints=True, shade_grid_res=400, contour_res=100, multi_constraint_single_label=False,
                   plot_local_constrained_pareto_sets=False, plot_local_unconstrained_pareto_sets=False, plot_only_nondominated_local_points=False,
                   rasterized=True, fig_width=3.5, cmap=CMAP, show_dimension_objective=True, show_legend=True, show_title=True,
                   center_constrained_front=True):
        """ Figure for problems with a multi-dimensional search space. """
        ax0 = 0
        ax1 = 1

        # Determine number of plots
        num_plots = 2 if plot_objective_space and plot_search_space else 1
        _, axes = plt.subplots(1, num_plots, figsize=(fig_width * num_plots, fig_width))
        ax = axes if num_plots == 1 else axes[0]

        # Peak colors
        peak_color1 = peak_color2 = 'black'
        levels_color1 = levels_color2 = 'gray'
        linewidths = 0.5
        if color_peaks:
            peak_color1 = plt.get_cmap('Set1')(0)
            peak_color2 = plt.get_cmap('Accent')(4)
            levels_color1 = [peak_color1]
            levels_color2 = [peak_color2]
            linewidths = 1

        local_cmap = plt.get_cmap('Set2')
        n_local_colors = max(1, local_cmap.N - 1)

        local_unconstrained_sets_to_plot = self.local_unconstrained_pareto_sets
        local_unconstrained_fronts_to_plot = self.local_unconstrained_pareto_fronts
        local_constrained_sets_to_plot = self.local_pareto_sets
        local_constrained_fronts_to_plot = self.local_pareto_fronts

        if plot_only_nondominated_local_points:
            local_unconstrained_sets_to_plot, local_unconstrained_fronts_to_plot = self._filter_local_points_to_nondominated(
                self.local_unconstrained_pareto_sets, self.local_unconstrained_pareto_fronts)
            local_constrained_sets_to_plot, local_constrained_fronts_to_plot = self._filter_local_points_to_nondominated(
                self.local_pareto_sets, self.local_pareto_fronts)

        # Plot search space
        if plot_search_space:
            if self.n_var == 2:
                # Define the grid for contour plotting
                x_range = np.linspace(self.xl[ax0], self.xu[ax0], contour_res)
                y_range = np.linspace(self.xl[ax1], self.xu[ax1], contour_res)
                X, Y = np.meshgrid(x_range, y_range)
                grid = np.stack([X, Y], axis=-1)

                # Calculate function values for contour plots
                Z1 = np.apply_along_axis(lambda x: multi_peak_function(
                    x, self.objectives[0]['c'], self.objectives[0]['H'], self.objectives[0]['b']
                ), -1, grid)
                Z2 = np.apply_along_axis(lambda x: multi_peak_function(
                    x, self.objectives[1]['c'], self.objectives[1]['H'], self.objectives[1]['b']
                ), -1, grid)

                ax.contour(X, Y, Z1, levels=25, colors=levels_color1, alpha=0.3, linewidths=linewidths)
                ax.contour(X, Y, Z2, levels=25, colors=levels_color2, alpha=0.3, linewidths=linewidths)

            # Plot the peak centers for f1 and f2
            if plot_large_peak_centers:
                ax.scatter(self.objectives[0]['c'][:, ax0], self.objectives[0]['c'][:, ax1],
                           color=peak_color1, marker='x', s=100, rasterized=rasterized)
                ax.scatter(self.objectives[1]['c'][:, ax0], self.objectives[1]['c'][:, ax1],
                           color=peak_color2, marker='+', s=100 * 1.4, rasterized=rasterized)
                for i in range(len(self.objectives[0]['c'])):
                    point_name = f'$c_{{1,{i + 1}}}$' if len(self.objectives[0]['c']) > 1 else '$c_1$'
                    ax.text(self.objectives[0]['c'][i, ax0] + 0.3, self.objectives[0]['c'][i, ax1] - 0.5, point_name,
                            fontsize=16, color=peak_color1, zorder=4)
                for i in range(len(self.objectives[1]['c'])):
                    point_name = f'$c_{{2,{i + 1}}}$' if len(self.objectives[1]['c']) > 1 else '$c_2$'
                    ax.text(self.objectives[1]['c'][i, ax0] + 0.3, self.objectives[1]['c'][i, ax1] - 0.5, point_name,
                            fontsize=16, color=peak_color2, zorder=4)
            else:
                ax.scatter(self.objectives[0]['c'][:, ax0], self.objectives[0]['c'][:, ax1],
                           color=peak_color1, marker='x', s=40, rasterized=rasterized)
                ax.scatter(self.objectives[1]['c'][:, ax0], self.objectives[1]['c'][:, ax1],
                           color=peak_color2, marker='+', s=40 * 1.4, rasterized=rasterized)

            # Plot all constraints
            lo, hi, res = -20, 20, shade_grid_res
            grid = np.meshgrid(np.linspace(lo, hi, res), np.linspace(lo, hi, res))
            plot_linear_constraints(ax, self.constraints['Linear'], ax0, ax1, cmap, shade=shade_infeasible_lin_quad, grid=grid)
            start_index = len(self.constraints['Linear'])
            plot_quadratic_constraints(ax, self.constraints['Quadratic'], ax0, ax1, cmap, shade=shade_infeasible_lin_quad,
                                       grid=grid, base_index=start_index)
            start_index = len(self.constraints['Linear']) + len(self.constraints['Quadratic'])
            plot_multi_constraints(ax, self.constraints['Multi'], ax0, ax1, cmap,
                                   shade=shade_infeasible_multi_constraints, grid=grid,
                                   single_label=multi_constraint_single_label,
                                   start_index=start_index)

            # Plot local unconstrained Pareto sets
            if local_unconstrained_sets_to_plot is not None and plot_local_unconstrained_pareto_sets:
                for i, (_, local_pareto_set) in enumerate(local_unconstrained_sets_to_plot.items()):
                    if len(local_pareto_set) > 0:
                        color = local_cmap((2 * i + 1) % n_local_colors)
                        ax.scatter(local_pareto_set[:, ax0], local_pareto_set[:, ax1],
                                   label='Local Uncon. Pareto set',
                                   color=color, zorder=2, s=10, rasterized=rasterized)

            # Plot local constrained Pareto sets
            if local_constrained_sets_to_plot is not None and plot_local_constrained_pareto_sets:
                for i, (_, local_pareto_set) in enumerate(local_constrained_sets_to_plot.items()):
                    if len(local_pareto_set) > 0:
                        color = local_cmap((2 * i) % n_local_colors)
                        ax.scatter(local_pareto_set[:, ax0], local_pareto_set[:, ax1], label='Local Pareto set',
                                   color=color, zorder=2, s=10, rasterized=rasterized)

            # Plot the unconstrained Pareto set
            if self.uncon_pareto_set is not None and plot_unconstrained_pareto:
                ax.scatter(self.uncon_pareto_set[:, ax0], self.uncon_pareto_set[:, ax1], label='Uncon. Pareto set',
                           color='grey', s=unconstrained_pareto_size, alpha=1, rasterized=rasterized)
                
            # Plot constrained Pareto set
            if self.pareto_set is not None and len(self.pareto_set) > 0 and plot_constrained_pareto:
                ax.scatter(self.pareto_set[:, ax0], self.pareto_set[:, ax1], label='Pareto set',
                           color='black', zorder=2, s=10, rasterized=rasterized)

            # Plot the algorithm results
            if algorithm_X is not None:
                ax.scatter([x[ax0] for x in algorithm_X], [x[ax1] for x in algorithm_X], label=algorithm_name,
                           facecolors=algorithm_color, edgecolors='black', linewidths=0.3, marker='o',
                           s=algorithm_point_size, zorder=5, rasterized=rasterized)

            if show_title:
                ax.set_title(f'Search space ($n={self.n_var}$)')
            ax.set_xlabel(f'$x_{ax0 + 1}$', size='larger')
            ax.set_ylabel(f'$x_{ax1 + 1}$', size='larger', rotation=0, labelpad=5)
            if show_legend:
                ax.legend(fontsize='small')
            ax.set_xlim(self.xl[ax0], self.xu[ax0])
            ax.set_ylim(self.xl[ax1], self.xu[ax1])
            ax.set_aspect('equal', adjustable='box')

        # Plot objective space
        if plot_objective_space:
            ax = axes[1] if plot_search_space else axes

            # Plot local unconstrained Pareto fronts
            if local_unconstrained_fronts_to_plot is not None and plot_local_unconstrained_pareto_sets:
                ideal = self.ideal_point() if plot_normalized_front else None
                nadir = self.nadir_point() if plot_normalized_front else None

                for i, (_, local_pareto_front) in enumerate(local_unconstrained_fronts_to_plot.items()):
                    if len(local_pareto_front) > 0:
                        color = local_cmap((2 * i + 1) % n_local_colors)
                        front_to_plot = (local_pareto_front - ideal) / (nadir - ideal) if plot_normalized_front else local_pareto_front
                        ax.scatter(front_to_plot[:, ax0], front_to_plot[:, ax1],
                                   label='Local Uncon. Pareto front',
                                   color=color, zorder=2, s=10, rasterized=rasterized)

            # Plot local constrained Pareto fronts
            if local_constrained_fronts_to_plot is not None and plot_local_constrained_pareto_sets:
                ideal = self.ideal_point() if plot_normalized_front else None
                nadir = self.nadir_point() if plot_normalized_front else None

                for i, (_, local_pareto_front) in enumerate(local_constrained_fronts_to_plot.items()):
                    if len(local_pareto_front) > 0:
                        color = local_cmap((2 * i) % n_local_colors)
                        front_to_plot = (local_pareto_front - ideal) / (nadir - ideal) if plot_normalized_front else local_pareto_front
                        ax.scatter(front_to_plot[:, ax0], front_to_plot[:, ax1],
                                   label='Local Pareto front',
                                   color=color, zorder=2, s=10, rasterized=rasterized)

            # Plot unconstrained Pareto front
            if self.uncon_pareto_front is not None and plot_unconstrained_pareto:
                upf_to_plot = self.uncon_pareto_front
                if plot_normalized_front:
                    ideal = self.ideal_point()
                    nadir = self.nadir_point()
                    upf_to_plot = (self.uncon_pareto_front - ideal) / (nadir - ideal)

                ax.scatter(upf_to_plot[:, 0], upf_to_plot[:, 1],
                        c='grey', s=unconstrained_pareto_size, label='Uncon. Pareto front', rasterized=rasterized)

            # Plot constrained Pareto front
            if self.pareto_front is not None and len(self.pareto_front) > 0 and plot_constrained_pareto:
                pf_to_plot = self.pareto_front
                if plot_normalized_front:
                    ideal = self.ideal_point()
                    nadir = self.nadir_point()
                    pf_to_plot = (self.pareto_front - ideal) / (nadir - ideal)

                ax.scatter(pf_to_plot[:, 0], pf_to_plot[:, 1],
                        color='black', s=10, zorder=2, label='Pareto front', rasterized=rasterized)

            # Plot algorithm results
            if algorithm_F is not None and len(algorithm_F) > 0:
                if normalize_algorithm:
                    ideal = self.ideal_point()
                    nadir = self.nadir_point()
                    algorithm_F_plot = (algorithm_F - ideal) / (nadir - ideal)
                else:
                    algorithm_F_plot = algorithm_F

                ax.scatter(algorithm_F_plot[:, 0], algorithm_F_plot[:, 1], label=algorithm_name,
                           facecolors=algorithm_color, edgecolors='black', linewidths=0.03, marker='o',
                           zorder=5, rasterized=rasterized, s=algorithm_point_size
                )

            if show_title:
                title = "Objective space ($m=2$)" if show_dimension_objective else "Objective space"
                ax.set_title(title)
            ax.set_xlabel(f'$f_1$', size='larger')
            ax.set_ylabel(f'$f_2$', size='larger', rotation=0, labelpad=7)
            ax.grid(True, which='both', linestyle='--')
            if show_legend:
                ax.legend(fontsize='small')

            if plot_normalized_front:
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                if show_title:
                    title = "Norm. objective space ($m=2$)" if show_dimension_objective else "Objective space"
                    ax.set_title(title)
            elif center_constrained_front and self.pareto_front is not None and len(self.pareto_front) > 0:
                pf = self.pareto_front
                pad = 0.1
                x_min, x_max = pf[:, 0].min(), pf[:, 0].max()
                y_min, y_max = pf[:, 1].min(), pf[:, 1].max()
                dx = (x_max - x_min) * pad
                dy = (y_max - y_min) * pad
                ax.set_xlim(x_min - dx, x_max + dx)
                ax.set_ylim(y_min - dy, y_max + dy)

            # Plot rectangles
            if self.rectangles is not None and plot_rectangles:
                if plot_normalized_front:
                    ideal = self.ideal_point()
                    nadir = self.nadir_point()

                for rect in self.rectangles:
                    (i, j, k, p1, p2) = rect

                    if plot_normalized_front:
                        p1 = (p1 - ideal) / (nadir - ideal)
                        p2 = (p2 - ideal) / (nadir - ideal)

                    xmin = min(p1[0], p2[0])
                    ymin = min(p1[1], p2[1])
                    width = abs(p2[0] - p1[0])
                    height = abs(p2[1] - p1[1])

                    h = hash((i, j, k)) % cmap.N
                    color = cmap(h / cmap.N)

                    ax.add_patch(
                        Rectangle(
                            (xmin, ymin),
                            width,
                            height,
                            edgecolor=color,
                            facecolor='none',
                            linewidth=1.5,
                            linestyle='--',
                            zorder=3
                        )
                    )

        return axes

    def save_figure(self, algorithm_X=None, algorithm_name='Algorithm', show=False, save=False,
                    folder='plots', extension='png', dpi=300, plot_name=None, **kwargs):
        """ Saves the generated figure to file or displays it. """
        plt.tight_layout()
        if save:
            if plot_name is None:
                plot_name = f'problem_{self.name}'
            if algorithm_X is not None:
                plt.savefig(f'{folder}/{plot_name}-{algorithm_name}.{extension}', dpi=dpi)
            else:
                plt.savefig(f'{folder}/{plot_name}.{extension}', dpi=dpi)
        if show:
            plt.show()
        plt.close()

    def visualize(self, algorithm_X=None, algorithm_F=None, algorithm_name='Algorithm',
                  algorithm_color='green', algorithm_point_size=6,
                  show=False, save=False, folder='plots', extension='png', dpi=300, plot_name=None,
                  plot_objective_space=True, plot_search_space=True,
                  plot_unconstrained_pareto=True, unconstrained_pareto_size=6,
                  plot_constrained_pareto=True, plot_normalized_front=False, normalize_algorithm=False,
                  shade_infeasible_lin_quad=True, shade_infeasible_multi_constraints=True, shade_grid_res=400, contour_res=100,
                  plot_rectangles=False, color_peaks=False, plot_large_peak_centers=True, multi_constraint_single_label=False,
                  plot_local_constrained_pareto_sets=False, plot_local_unconstrained_pareto_sets=False,
                  plot_only_nondominated_local_points=False,
                  rasterized=True, fig_width=3.5, cmap=CMAP, show_dimension_objective=True, show_legend=True,
                  show_title=True, center_constrained_front=True):
        """
        Visualizes the optimization problem and computed results (search space and objective space).

        Depending on the number of decision variables self.n_var, it calls either get_figure_1d for 1D problems or get_figure for multi-dimensional problems.

        Parameters:
            - algorithm_X: Algorithm solutions in the decision/search space
            - algorithm_F: Corresponding objective values for algorithm_X
            - algorithm_name: Name for algorithm results in the legend
            - algorithm_color: Color for algorithm points
            - algorithm_point_size: Size of algorithm points
            - show: If True, displays the figure
            - save: If True, saves the figure
            - folder: Directory for saving the figure
            - extension: File extension for saved figure
            - dpi: Resolution for saved figure
            - plot_name: Custom name for saved figure file
            - plot_objective_space: If True, plots objective space (f1 vs f2)
            - plot_search_space: If True, plots search space (decision variable space)
            - plot_unconstrained_pareto: If True, plots unconstrained Pareto set/front
            - unconstrained_pareto_size: Marker size for unconstrained Pareto points
            - plot_constrained_pareto: If True, plots constrained Pareto set/front
            - plot_normalized_front: If True, normalizes Pareto fronts between ideal and nadir points
            - normalize_algorithm: If True, normalizes algorithm results in objective space with nadir and ideal from Pareto front
            - shade_infeasible_lin_quad: If True, shades infeasible regions for linear and quadratic constraints (multi-dimensional only)
            - shade_infeasible_multi_constraints: If True, shades infeasible regions for multipeak constraints (multi-dimensional only)
            - shade_grid_res: Resolution used for shading infeasible regions.
            - contour_res: Resolution used for contour plots of the objective functions.
            - plot_rectangles: If True, plots rectangles in objective space (multi-dimensional only)
            - color_peaks: If True, colors peaks differently in the search space
            - plot_large_peak_centers: If True, makes peak center markers larger
            - multi_constraint_single_label: If True, multi-constraint infeasible regions share a single legend label (multi-dimensional only)
            - plot_local_constrained_pareto_sets: If True, plots local constrained Pareto sets (multi-dimensional only)
            - plot_local_unconstrained_pareto_sets: If True, plots local unconstrained Pareto sets (multi-dimensional only)
            - plot_only_nondominated_local_points: If True, each local Pareto set/front is filtered independently so that only nondominated points of that local front are shown
            - rasterized: If True, rasterizes the plot
            - fig_width: Width of individual subplots
            - cmap: Colormap for constraints, rectangles, and local fronts (multi-dimensional only)
            - show_dimension_objective: If True, includes dimensional information in objective space title
            - show_legend: If True, displays legend
            - show_title: If True, displays plot titles
            - center_constrained_front: If True, centers axes around constrained Pareto front (multi-dimensional only)

        Returns nothing.
        """

        if self.n_var == 1:
            self.get_figure_1d(algorithm_X=algorithm_X, algorithm_F=algorithm_F, algorithm_name=algorithm_name, algorithm_color=algorithm_color,
                               algorithm_point_size=algorithm_point_size, plot_objective_space=plot_objective_space,
                               plot_search_space=plot_search_space,
                               plot_unconstrained_pareto=plot_unconstrained_pareto, unconstrained_pareto_size=unconstrained_pareto_size,
                               plot_constrained_pareto=plot_constrained_pareto,
                               plot_normalized_front=plot_normalized_front, normalize_algorithm=normalize_algorithm,
                               color_peaks=color_peaks, plot_large_peak_centers=plot_large_peak_centers,
                               rasterized=rasterized, fig_width=fig_width, show_dimension_objective=show_dimension_objective,
                               show_legend=show_legend, show_title=show_title, center_constrained_front=center_constrained_front)
        else:
            self.get_figure(algorithm_X=algorithm_X, algorithm_F=algorithm_F, algorithm_name=algorithm_name, algorithm_color=algorithm_color,
                            algorithm_point_size=algorithm_point_size, plot_objective_space=plot_objective_space,
                            plot_search_space=plot_search_space,
                            plot_unconstrained_pareto=plot_unconstrained_pareto, unconstrained_pareto_size=unconstrained_pareto_size,
                            plot_constrained_pareto=plot_constrained_pareto,
                            plot_normalized_front=plot_normalized_front, normalize_algorithm=normalize_algorithm,
                            shade_infeasible_lin_quad=shade_infeasible_lin_quad, shade_infeasible_multi_constraints=shade_infeasible_multi_constraints,
                            shade_grid_res=shade_grid_res, contour_res=contour_res, plot_rectangles=plot_rectangles, color_peaks=color_peaks, plot_large_peak_centers=plot_large_peak_centers,
                            multi_constraint_single_label=multi_constraint_single_label,
                            plot_local_constrained_pareto_sets=plot_local_constrained_pareto_sets,
                            plot_local_unconstrained_pareto_sets=plot_local_unconstrained_pareto_sets,
                            plot_only_nondominated_local_points=plot_only_nondominated_local_points,
                            rasterized=rasterized, fig_width=fig_width, cmap=cmap, show_dimension_objective=show_dimension_objective,
                            show_legend=show_legend, show_title=show_title, center_constrained_front=center_constrained_front)

        self.save_figure(algorithm_X=algorithm_X, algorithm_name=algorithm_name, show=show, save=save, folder=folder,
                         extension=extension, dpi=dpi, plot_name=plot_name)
    
    def properties_to_string(self, n_digits=4):
        """ Returns a string representation of the problem's properties, with numbers formatted to n_digits digits. """
        out = []
        out.append(f"Number of decision variables: {self.n_var}")
        out.append(f"Peaks f1 / f2: {len(self.objectives[0]['c'])} / {len(self.objectives[1]['c'])}")
        out.append(f"Number of constraints: {self.n_constr}")
        out.append(f"Domain: {self.domain}")
        out.append(f"Boundary constraints included: {self.boundary_constraints}")

        for i, obj in enumerate(self.objectives):
            out.append(f"-- Objective f{i+1} --")
            out.append("H:")
            for H in obj['H']:
                out.append(np.array2string(H, separator=', ', precision=n_digits))
            out.append("c:")
            out.append(np.array2string(np.array(obj['c']), separator=', ', precision=n_digits))
            out.append("b:")
            out.append(np.array2string(np.array(obj['b']), separator=', ', precision=n_digits))
            if obj["transformation"] is not None:
                trans = obj["transformation"]
                out.append(
                    f'Transformation: {trans["name"]} with parameters '
                    f'{ {k: v.item() if isinstance(v, np.generic) else v for k, v in trans["params"].items()} }'
                )
            if obj["peak_transformations"] is not None:
                peak_trans = obj["peak_transformations"]
                out.append("Peak transformations:")
                for trans in peak_trans:
                    if trans is not None:
                        out.append(
                            f'  Peak transformation: {trans["name"]} with parameters '
                            f'{ {k: v.item() if isinstance(v, np.generic) else v for k, v in trans["params"].items()} }'
                        )
        out.append("-- Constraints --")
        for key in ['Linear', 'Quadratic', 'Multi']:
            out.append(f"{key}:")
            data = self.constraints.get(key, [])
            if key == 'Multi':
                for i, item in enumerate(data):
                    out.append(f" Multi constraint {i+1}:")
                    for j, group in enumerate(item):
                        out.append(f"  Group {j+1}:")
                        for groupkey in ['Linear', 'Quadratic']:
                            if groupkey in group:
                                out.append(f"   {groupkey}:")
                                for item in group[groupkey]:
                                    for k, v in item.items():
                                        if v is not None:
                                            if isinstance(v, np.ndarray):
                                                out.append(f"    {k}:")
                                                s = np.array2string(np.array(v), separator=', ', precision=n_digits)
                                                indent_str = ' ' * 6
                                                out.append('\n'.join(indent_str + line for line in s.splitlines()))
                                            else:
                                                out.append(f"    {k}: {v}")
            else:
                for item in data:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if v is not None:
                                if isinstance(v, np.ndarray):
                                    s = np.array2string(v, separator=', ', precision=n_digits)
                                    indent_str = ' ' * 2
                                    out.append(f" {k}:")
                                    out.append('\n'.join(indent_str + line for line in s.splitlines()))
                                else:
                                    out.append(f" {k}: {v}")
                    else:
                        out.append(f" {item}")
        return "\n".join(out)
    
    @property
    def name(self):
        """ Identifier for the problem. """
        n_var = self.n_var
        
        n_peaks_1 = len(self.objectives[0]['c'])
        n_peaks_2 = len(self.objectives[1]['c'])
        
        n_linear = len(self.constraints.get('Linear', []))
        n_quadratic = len(self.constraints.get('Quadratic', []))
        n_multi = len(self.constraints.get('Multi', []))

        hash_value = int(hashlib.md5(self.properties_to_string(n_digits=None).encode('utf-8')).hexdigest(), 16)
        
        return f"COBI-{n_var}-{n_peaks_1}-{n_peaks_2}-{n_linear}-{n_quadratic}-{n_multi}-{hash_value}"

    def to_string(self, n_digits=4):
        """ Returns a string representation of the problem, with numbers formatted to n_digits digits. """
        out = []
        out.append("=== CobiProblem Data ===")
        out.append(f"Name: {self.name}")
        out.append(self.properties_to_string(n_digits=n_digits))
        return "\n".join(out) + "\n"

    def __str__(self):
        """ Returns a string representation of the problem. """
        return self.to_string(n_digits=4)