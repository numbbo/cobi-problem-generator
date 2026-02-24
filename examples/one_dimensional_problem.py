import numpy as np
from cobi import CobiProblem, NSGA2, run_algorithm_track_diff_to_opt
import platform
import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')


print('Creating a 1D user-defined COBI problem...')

# Objective function 1
centers_f1 = np.array([[2.0], [-2.0]])
Hessians_f1 = np.array([
    [[1.0]],
    [[2.0]]
])
v_shifts_f1 = np.array([0.0, 4.0])
alphas_f1 = np.array([1.1, 1.1])

# Objective function 2
centers_f2 = np.array([[3.0], [-3.0], [0.0]])
Hessians_f2 = np.array([
    [[1.5]],
    [[1.0]],
    [[0.5]]
])
v_shifts_f2 = np.array([0.0, 6.0, 10.0])
alphas_f2 = np.array([0.9, 0.9, 0.9])

# Constraints
linear_constraints = [
    {'P': np.array([0.0]), 'n': np.array([-1.0])}
]
quadratic_constraints = [
    {
        'H': np.array([[1.5]]),
        'c': np.array([1.0]),
        'b': 4.0
    }
]
multi_constraints = [
    [
        {
            'Quadratic': [
                {
                    'H': np.array([[3.5]]),
                    'c': np.array([10.0]),
                    'b': 1.0,
                }
            ],
            'Linear': []
        },
        {
            'Quadratic': [],
            'Linear': [{'P': np.array([2.5]), 'n':  np.array([-1.0])}]
        }
    ]
]

# Create problem
objectives = (
    {'H': Hessians_f1, 'c': centers_f1, 'b': v_shifts_f1, 'alphas': alphas_f1},
    {'H': Hessians_f2, 'c': centers_f2, 'b': v_shifts_f2, 'alphas': alphas_f2}
)
constraints = {
    'Linear': linear_constraints,
    'Quadratic': quadratic_constraints,
    'Multi': multi_constraints
}
problem = CobiProblem(
    n_var=1,
    objectives=objectives,
    constraints=constraints,
    domain=(-5, 5),
    alpha=(2, 0.5)
)
problem.normalize_problem()

# Calculate Pareto set and front
print("Computing the Pareto set and front...")
sampling_options = {'sampling': 'max-HV', 'max_error': 1e-3, 'always_compute_unconstrained': True}
problem.calculate_pareto_set_and_front(sampling_options=sampling_options)

# Run NSGA-II for comparison
print("Running the NSGA-II algorithm on the problem...")
algorithm_n_evals = 100000
nda, diff = run_algorithm_track_diff_to_opt(
    algorithm=NSGA2(),
    problem=problem,
    n_evals=algorithm_n_evals,
    seed=1,
    indicator='HV+'
)
algorithm_X = np.array([list(d['x']) for d in nda.infos])
algorithm_F = np.array(list(nda))

# Visualize results
problem.visualize(
    algorithm_name='NSGA-II',
    algorithm_X=algorithm_X,
    algorithm_F=algorithm_F,
    show=True
)
