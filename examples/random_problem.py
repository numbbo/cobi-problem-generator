from cobi import create_random_problem, NSGA2, run_algorithm_track_diff_to_opt, load_problem
import numpy as np
import os
import time
import platform
import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')


print('Creating a random COBI problem...\n')

SEED = 2

# Random problem
problem = create_random_problem(n_var=2,
                                domain=(-5, 5),
                                seed=SEED,
                                n_peaks=((2, 4), (2, 4)),
                                peaks_value_shift=15,
                                peaks_condition_number=None,
                                peaks_alphas=(0.5, 2),
                                alpha=(2, 0.5),
                                n_constraints={'Linear': 1, 'Quadratic': 1, 'Multi': 1},
                                quadratic_constraints_size=(5, 15),
                                quadratic_constraints_condition_number=None,
                                n_multi_constraints_groups=(2, 3),
                                n_multi_constraints_group_linear=(0, 1),
                                n_multi_constraints_group_quadratic=(1, 2),
                                constraints_feasible=True,
                                perpendicular_linear_constraints=False,
                                n_digits=8,
                                print_seed=False)

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

# Make the front normalized
problem.normalize_problem()

# Print information about the problem
print(problem)

# Calculate Pareto set and front
start = time.time()
problem.calculate_pareto_set_and_front(sampling_options=sampling_options, tol_feasible=1e-8, skip_dominated=True, print_output=True)
end = time.time()

# Properties of the computed Pareto set and front approximations
print(f'Time needed to approximate Pareto set and front: {end - start:.4f} s')
print(f'Number of times the numerical solver failed: {problem.num_solver_failed}')
print(f'Pareto set size: {len(problem.pareto_set)}')
props = problem.characterize_problem(dist_thresh_set=0.25, dist_thresh_front=0.2)
print("Properties of problem:")
for key, value in props.items():
    print(f"  {key:25}: {value}")

# Binding constraints
binding_sampling_options = {'sampling': 'max-HV', 'max_error': 0.1, 'max_points': 50, 'always_compute_unconstrained': False}
print("\nComputing binding constraints...")
props = problem.calculate_binding_constraints(tol=1e-6, sampling_options=sampling_options, tol_feasible=1e-8)
print("Binding constraints:")
for key, value in props.items():
    print(f"  {key:25}: {value}")
print()

FOLDER = 'results'
os.makedirs(FOLDER, exist_ok=True)

# Run NSGA-II for comparison
algorithm_n_evals = 100000
print("Running the NSGA-II algorithm on the problem...")
algorithm_nda, diff = run_algorithm_track_diff_to_opt(problem=problem, algorithm=NSGA2(), n_evals=algorithm_n_evals, seed=SEED, indicator='HV+')
algorithm_X = np.array([list(d['x']) for d in algorithm_nda.infos])
algorithm_F = np.array(list(algorithm_nda))

# Visualize results
print("Visualizing the results...\n")
if problem.n_var > 1:
    ax = problem.get_figure(algorithm_name='NSGA-II', algorithm_X=algorithm_X, algorithm_F=algorithm_F)
else:
    ax = problem.get_figure_1d(algorithm_name='NSGA-II', algorithm_X=algorithm_X, algorithm_F=algorithm_F)
ax[0].legend().remove()  # Remove legend from the search space
#ax[1].legend().remove()  # Remove legend from the objective space
problem.save_figure(show=True, save=True, folder=FOLDER)

# Save and load results
print("Saving and loading solutions...")
problem.save_problem(f'{FOLDER}/{problem.name}.pkl')
problem_saved = load_problem(f'{FOLDER}/{problem.name}.pkl')
print(f"The size of the loaded Pareto set is {len(problem_saved.pareto_set)}.")
