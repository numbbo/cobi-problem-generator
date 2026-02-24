import numpy as np
from cobi import CobiProblem, rotation_matrix, NSGA2, run_algorithm_track_diff_to_opt, load_problem
import platform
import matplotlib
import os
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')


print('Creating a user-defined COBI problem...\n')

os.makedirs('results', exist_ok=True)

R15 = rotation_matrix(15)

# Objective function 1
centers_f1 = np.array([[4, 4], [-4, -4]])
Hessians_f1 = np.array([[[1, 0], [0, 1]], R15 @ [[2, 0], [0, 1]] @ R15.T])
value_shifts_f1 = np.array([0, 5])
alphas_f1 = np.array([1.1, 1.1])

# Objective function 2
centers_f2 = np.array([[4, -4], [-4, 4], [0, 0]])
Hessians_f2 = np.array([[[3, 0], [0, 1]], R15 @ [[2, 0], [0, 3]] @ R15.T, R15.T @ [[2, 0], [0, 3]] @ R15])
value_shifts_f2 = np.array([0, 10, 15])
alphas_f2 = np.array([0.9, 1.0, 1.1])

# Constraints
linear_constraints = [
    {'P': np.array([0, 0]), 'n':  np.array([1, 1])},
    {'P': np.array([-1, -1]), 'n':  np.array([-1, 1])}
]
quadratic_constraints = [
    {
        'H': np.array([[2.5, 1.5], [1.5, 2.5]]),
        'c': np.array([0.0, 0.0]),
        'b': 20.0
    },
    {
        'H': R15 @ np.diag([1.0, 2.0]) @ R15.T,
        'c': np.array([1.5, 1.0]),
        'b': 30.0
    }
]
multi_constraints = [
    [
        {
            'Quadratic': [
                {
                    'H': np.array([[2.5, 1.5], [1.5, 2.5]]),
                    'c': np.array([4.0, 4.0]),
                    'b': 5.0,
                },
                {
                    'H': np.diag([2.0, 1.0]),
                    'c': np.array([4.0, 4.0]),
                    'b': 1,
                }
            ],
            'Linear': []
        },
        {
            'Quadratic': [
                {
                    'H': np.array(R15 @ np.diag([1.0, 2.0]) @ R15.T),
                    'c': np.array([-1.5, -1.0]),
                    'b': 2,
                }
            ],
            'Linear': [{'P': np.array([2, 2]), 'n':  np.array([0, 1])}]
        }
    ],
    [
        {
            'Quadratic': [],
            'Linear': [{'P': np.array([-2, -2]), 'n':  np.array([-1, 0])}]
        },
        {
            'Quadratic': [
                {
                    'H': np.array(np.diag([1.0, 5.0])),
                    'c': np.array([-2.5, -2.5]),
                    'b': 2,
                }
            ],
            'Linear': []
        }
    ]
]

# Create problem
objectives = ({'H': Hessians_f1, 'c': centers_f1, 'b': value_shifts_f1, 'alphas': alphas_f1}, {'H': Hessians_f2, 'c': centers_f2, 'b': value_shifts_f2, 'alphas': alphas_f2})
constraints = {'Linear': linear_constraints, 'Quadratic': quadratic_constraints, 'Multi': multi_constraints}
problem = CobiProblem(
    n_var=2,
    objectives=objectives,
    constraints=constraints,
    domain=(-5, 5),
    alpha=(2, 0.5),
    boundary_constraints=True
)
problem.normalize_problem()  # Make the front normalized

# Print information about the problem
print(problem)

# Sampling options
#sampling_options = {'sampling': 'equi-w', 'n_points': 1000}
#sampling_options = {'sampling': 'equi-uncon-x', 'distance': 0.05, 'tol_distance': 1e-8, 'tol_jump': 1e-3, 'max_iter': 10000, 'force_equidistant': True}
#sampling_options = {'sampling': 'equi-x', 'distance': 0.15, 'tol_distance': 1e-8, 'tol_jump': 1e-3, 'max_iter': 10000, 'force_equidistant': True, 'always_compute_unconstrained': True}
#sampling_options = {'sampling': 'equi-f', 'distance': 0.15, 'tol_distance': 1e-8, 'tol_jump': 1e-3, 'max_iter': 10000, 'force_equidistant': True, 'always_compute_unconstrained': True}
sampling_options = {'sampling': 'max-HV', 'max_error': 1e-4, 'max_points': 250, 'always_compute_unconstrained': True}
#sampling_options = {'sampling': 'rectangles', 'rectangles_sampling': 'equi-w', 'rectangles_min_distance': 1e-3, 'n_points': 1000, 'always_compute_unconstrained': True}
#sampling_options = {'sampling': 'rectangles', 'rectangles_sampling': 'equi-x', 'rectangles_min_distance': 1e-3, 'distance': 0.1, 'tol_distance': 1e-8, 'tol_jump': 1e-3, 'max_iter': 10000, 'force_equidistant': True, 'always_compute_unconstrained': True}
#sampling_options = {'sampling': 'rectangles', 'rectangles_sampling': 'equi-f', 'rectangles_min_distance': 1e-3, 'distance': 0.1, 'tol_distance': 1e-8, 'tol_jump': 1e-3, 'max_iter': 10000, 'force_equidistant': True, 'always_compute_unconstrained': True}
#sampling_options = {'sampling': 'edge'}

# Calculate Pareto set and front
problem.calculate_pareto_set_and_front(sampling_options=sampling_options, tol_feasible=1e-8, skip_dominated=True, solver=None, print_output=True)

# Problem characterization
props = problem.characterize_problem(dist_thresh_set=0.25, dist_thresh_front=0.2)
print("Properties of problem:")
for key, value in props.items():
    print(f"  {key:25}: {value}")

# Binding constraints
binding_sampling_options = {'sampling': 'max-HV', 'max_error': 0.1, 'max_points': 50, 'always_compute_unconstrained': False}
print("\nComputing binding constraints...")
props = problem.calculate_binding_constraints(tol=1e-6, sampling_options=binding_sampling_options, tol_feasible=1e-8, skip_dominated=True, solver=None, print_output=False)
print("Binding constraints:")
for key, value in props.items():
    print(f"  {key:25}: {value}")

# Run NSGA-II for comparison
print("\nRunning the NSGA-II algorithm on the problem...")
algorithm_n_evals = 100000
nda, diff = run_algorithm_track_diff_to_opt(algorithm=NSGA2(), problem=problem, n_evals=algorithm_n_evals, seed=1, indicator='HV+')
algorithm_X = np.array([list(d['x']) for d in nda.infos])
algorithm_F = np.array(list(nda))

# Visualize results
print("Visualizing the results...\n")
problem.visualize(algorithm_name='NSGA-II', algorithm_X=algorithm_X, algorithm_F=algorithm_F, show=True)

# Reduce size of the Pareto set approximation and compare hypervolumes
print("Reducing the size of the computed Pareto set and front...")
new_size = len(problem.pareto_set) // 2  # Reduce Pareto set size by half
hypervolume_before = problem.hypervolume
problem.reduce_pareto_set_size(new_size)
hypervolume_after = problem.hypervolume
print(f"Hypervolume before reduction: {hypervolume_before}")
print(f"Hypervolume after reduction:  {hypervolume_after}")

# Save results, load the saved problem, and print the first point of its Pareto set and front
print("\nSaving and loading solutions...")
problem.save_problem('results/test_user_problem_results.pkl')
problem_saved = load_problem('results/test_user_problem_results.pkl')
if len(problem_saved.pareto_set) > 0 and len(problem_saved.pareto_front) > 0:
    print("First Pareto set point:", problem_saved.pareto_set[0])
    print("First Pareto front point:", problem_saved.pareto_front[0])
else:
    print("Pareto set and Pareto front are empty.")
