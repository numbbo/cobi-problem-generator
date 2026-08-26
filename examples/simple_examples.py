from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # Switch backend to TkAgg
import matplotlib.pyplot as plt
import numpy as np
from cobi import CobiProblem


# Options

VISUALIZE_SEARCH = False
SHOW_FIGURES = True
SAVE_FIGURES = True

FIGURE_FOLDER = Path("results/simple_examples")
FIGURE_EXTENSION = "pdf"
FIGURE_DPI = 300
FIGURE_WIDTH = 4.0

N_POINTS_SIMPLE = 10000
N_POINTS_LOCAL = 5000


def _no_constraints():
    return {"Linear": [], "Quadratic": [], "Multi": []}


def _single_peak_objective(center, *, exponent=None, transformation=None):
    """ Create one strictly convex-quadratic objective peak. """
    peak_transformation = None
    if exponent is not None:
        peak_transformation = {
            "name": "exponent",
            "params": {"exponent": exponent},
        }

    return {
        "H": np.array([np.eye(2)]),
        "c": np.array([center], dtype=float),
        "b": np.array([0.0]),
        "transformation": transformation,
        "peak_transformations": [peak_transformation],
    }


def _two_center_problem(*, peak_exponent=None, transformation_f1=None, transformation_f2=None):
    """ Problem used as the base geometry of Examples 1--4a. """
    objectives = (
        _single_peak_objective(
            [-2.0, 0.0],
            exponent=peak_exponent,
            transformation=transformation_f1,
        ),
        _single_peak_objective(
            [2.0, 0.0],
            exponent=peak_exponent,
            transformation=transformation_f2,
        ),
    )

    return CobiProblem(
        n_var=2,
        objectives=objectives,
        constraints=_no_constraints(),
        domain=(-3.0, 3.0),
        boundary_constraints=False,
    )


# -----------------------------------------------------------------------------
# 1. Linear Pareto front
# -----------------------------------------------------------------------------

def create_linear_front_problem():
    """ The straight line Pareto front. """
    return _two_center_problem(peak_exponent=0.5)


# -----------------------------------------------------------------------------
# 2. Convex Pareto front
# -----------------------------------------------------------------------------

def create_convex_front_problem():
    """ The convex front. """
    return _two_center_problem()


# -----------------------------------------------------------------------------
# 3. Concave Pareto front
# -----------------------------------------------------------------------------

def create_concave_front_problem():
    """ The concave Pareto front. """
    return _two_center_problem(peak_exponent=0.25)


# -----------------------------------------------------------------------------
# 4a. Disconnected Pareto front using a transformation
# -----------------------------------------------------------------------------

def create_disconnected_front_problem():
    """ Two separated Pareto-front pieces created by a step transformation. """
    step = {
        "name": "step",
        "params": {
            "value": np.sqrt(2.0),
            "shift": 0.8,
        },
    }
    return _two_center_problem(
        peak_exponent=0.5,
        transformation_f1=step,
    )


# -----------------------------------------------------------------------------
# 4b. Disconnected Pareto front from multiple quadratic peaks
# -----------------------------------------------------------------------------

def create_disconnected_convex_parts_problem():
    """
    Two disconnected convex Pareto-front parts, using two quadratic peaks.
    """
    centers_f1 = np.array([
        [-1.0,  3.0],
        [-1.0, -3.0],
    ])
    centers_f2 = np.array([
        [1.0,  3.0],
        [1.0, -3.0],
    ])

    hessians = np.repeat(np.eye(2)[None, :, :], 2, axis=0)

    objectives = (
        {
            "H": hessians.copy(),
            "c": centers_f1,
            "b": np.array([0.0, 3.0]),
            "transformation": None,
            "peak_transformations": [None, None],
        },
        {
            "H": hessians.copy(),
            "c": centers_f2,
            "b": np.array([3.0, 0.0]),
            "transformation": None,
            "peak_transformations": [None, None],
        },
    )

    return CobiProblem(
        n_var=2,
        objectives=objectives,
        constraints=_no_constraints(),
        domain=(-4.0, 4.0),
        boundary_constraints=False,
    )


# -----------------------------------------------------------------------------
# 5. Many local Pareto fronts
# -----------------------------------------------------------------------------

def create_many_local_fronts_problem():
    """ A simple 4-by-4 multipeak problem with 16 local peak-pair fronts. """
    y = np.array([-3.0, -1.0, 1.0, 3.0])

    centers_f1 = np.column_stack((-3.0 * np.ones(4), y))
    centers_f2 = np.column_stack((3.0 * np.ones(4), y))

    # Keeping every Hessian equal to I makes the construction easy to explain;
    # the different center separations and value shifts create distinct local
    # fronts in objective space.
    hessians = np.repeat(np.eye(2)[None, :, :], 4, axis=0)

    objectives = (
        {
            "H": hessians.copy(),
            "c": centers_f1,
            "b": np.array([0.0, 1.0, 2.0, 3.0]),
            "transformation": None,
            "peak_transformations": [None] * 4,
        },
        {
            "H": hessians.copy(),
            "c": centers_f2,
            "b": np.array([3.0, 2.0, 1.0, 0.0]),
            "transformation": None,
            "peak_transformations": [None] * 4,
        },
    )

    return CobiProblem(
        n_var=2,
        objectives=objectives,
        constraints=_no_constraints(),
        domain=(-4.0, 4.0),
        boundary_constraints=False,
    )


def compute_problem(problem, *, show_all_local_fronts=False):
    """ Compute a dense Pareto approximation suitable for visualization. """
    n_points = N_POINTS_LOCAL if show_all_local_fronts else N_POINTS_SIMPLE
    problem.calculate_pareto_set_and_front(
        sampling_options={"sampling": "equi-w", "n_points": n_points},
        skip_dominated=not show_all_local_fronts,
        print_output=False,
    )
    return problem


def plot_problem(problem, title, file_name, *, show_all_local_fronts=False):
    """Create a clean objective-space figure using CobiProblem's plotting code."""
    axes = problem.get_figure(
        plot_objective_space=True,
        plot_search_space=VISUALIZE_SEARCH,
        plot_unconstrained_pareto=False,
        plot_constrained_pareto=True,
        plot_local_unconstrained_pareto_sets=show_all_local_fronts,
        plot_local_constrained_pareto_sets=False,
        plot_only_nondominated_local_points=show_all_local_fronts,
        shade_infeasible_lin_quad=False,
        shade_infeasible_multi_constraints=False,
        color_peaks=not show_all_local_fronts,
        plot_large_peak_centers=True,
        rasterized=False,
        fig_width=FIGURE_WIDTH,
        show_dimension_objective=False,
        show_legend=False,
        show_title=False,
        center_constrained_front=not show_all_local_fronts,
    )

    if VISUALIZE_SEARCH:
        search_ax, objective_ax = axes
        search_ax.set_title("Search space")
    else:
        objective_ax = axes

    objective_ax.set_title(title)
    plt.tight_layout()

    if SAVE_FIGURES:
        FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            FIGURE_FOLDER / f"{file_name}.{FIGURE_EXTENSION}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

    if SHOW_FIGURES:
        plt.show()

    plt.close()


def main():
    examples = [
        (
            create_linear_front_problem,
            "Linear Pareto front",
            "01_linear_pareto_front",
            False,
        ),
        (
            create_convex_front_problem,
            "Convex Pareto front",
            "02_convex_pareto_front",
            False,
        ),
        (
            create_concave_front_problem,
            "Concave Pareto front",
            "03_concave_pareto_front",
            False,
        ),
        (
            create_disconnected_front_problem,
            "Disconnected Pareto front (transformation)",
            "04a_disconnected_pareto_front_transformation",
            False,
        ),
        (
            create_disconnected_convex_parts_problem,
            "Two convex Pareto-front components",
            "04b_disconnected_convex_pareto_front",
            False,
        ),
        (
            create_many_local_fronts_problem,
            "Global and local Pareto fronts",
            "05_many_local_pareto_fronts",
            True,
        ),
    ]

    for create_problem, title, file_name, show_all_local_fronts in examples:
        problem = create_problem()
        compute_problem(problem, show_all_local_fronts=show_all_local_fronts)
        plot_problem(
            problem,
            title,
            file_name,
            show_all_local_fronts=show_all_local_fronts,
        )


if __name__ == "__main__":
    main()
