import numpy as np
from cobi import CobiProblem, NSGA2, run_algorithm_track_diff_to_opt
from cobi.utils import rotation_matrix as rm
import os


print("Creating a user-defined COBI multimodal problem...")

# Objective function
objectives = [{
    'H': np.array([
        [[ 2.5253, -0.7085], [-0.7085,  2.9692]],
        [[ 3.4871, -0.1211], [-0.1211,  7.0987]],
        [[ 6.1024,  4.2827], [ 4.2827,  6.5278]],
        [[ 2.7183, -1.3104], [-1.3104,  7.5628]],
        [[ 3.9242, -1.5418], [-1.5418,  7.1961]],
        [[ 2.7107,  0.7220], [ 0.7220,  3.5682]],
        [[ 3.8157,  0.1686], [ 0.1686,  2.6604]],
        [[ 3.5365,  1.0651], [ 1.0651,  2.7384]],
        [[ 6.0075, -0.0366], [-0.0366,  2.0027]],
        [[ 2.6386,  1.2332], [ 1.2332,  4.4279]]
    ]),
    'c': np.array([
        [ 4.6702,  0.4723],
        [ 4.7268,  2.1481],
        [ 1.9772, -2.8391],
        [ 4.7627, -4.9376],
        [-2.4701, -0.6520],
        [ 2.7938, -3.0231],
        [ 3.6299,  4.8340],
        [-3.3615,  0.9733],
        [-4.9101, -1.1342],
        [-4.5583,  4.5665]
    ]),
    'b': np.array([-1.2770,  8.9795,  5.7261,  7.3257, -6.5366,
          -8.5010,  2.0148, -6.6405,  4.6676, -1.8311])
}, {
    'H': np.array([
        [[ 6.3110, -3.6048], [-3.6048,  5.0206]],
        [[ 4.7799, -2.3436], [-2.3436,  4.0929]],
        [[ 2.4205, -0.1137], [-0.1137,  5.1677]],
        [[ 3.6427, -0.8879], [-0.8879,  2.7827]],
        [[ 3.5076, -2.4769], [-2.4769,  6.3291]],
        [[ 3.2504, -0.7793], [-0.7793,  4.8227]],
        [[ 3.3059,  0.4955], [ 0.4955,  2.8149]],
        [[ 3.7240,  1.8201], [ 1.8201,  3.9476]],
        [[ 2.8785,  0.4171], [ 0.4171,  2.2327]],
        [[ 2.4286,  0.8989], [ 0.8989,  6.3845]],
        [[ 2.0583,  0.1552], [ 0.1552,  3.2704]],
        [[ 4.6987,  1.3660], [ 1.3660,  2.6924]],
        [[ 2.0632,  0.2336], [ 0.2336,  2.9621]],
        [[ 4.7211,  0.9051], [ 0.9051,  3.1059]],
        [[ 4.4249,  2.2017], [ 2.2017,  4.0774]],
        [[ 2.2792,  0.2274], [ 0.2274,  8.0169]],
        [[ 4.6200,  3.4052], [ 3.4052,  6.5345]],
        [[ 3.7073, -1.4068], [-1.4068,  3.1594]],
        [[ 2.5391,  0.4166], [ 0.4166,  2.3247]],
        [[ 4.1711,  1.8466], [ 1.8466,  7.7948]]
    ]),
    'c': np.array([
        [ 2.3259,  3.9465],
        [ 0.1473,  1.0356],
        [-4.3493,  0.4007],
        [-3.7081,  1.1456],
        [-1.3634,  2.6775],
        [-4.5146, -3.9018],
        [ 1.8402,  0.1465],
        [ 0.7164,  3.4370],
        [-0.1226,  3.1014],
        [ 0.1024,  4.2672],
        [ 1.6692, -3.5127],
        [-1.3544,  3.6577],
        [-1.4971, -3.1097],
        [-0.2737, -1.0721],
        [ 1.1892, -0.6323],
        [-2.3907, -0.8752],
        [-0.8096,  4.0242],
        [ 4.7961,  1.2356],
        [-4.1681,  2.3299],
        [ 1.7868,  3.2602]
    ]),
    'b': np.array([-3.0509, -8.8223,  2.2364, -7.5197,  5.1905,
           5.8884, -1.8274,  8.8759, -6.5244,  8.8517,
          -0.6902,  5.2598,  5.0564,  9.6241,  9.4655,
          -9.1981, -9.6805,  0.8658, -9.3224,  3.1496])
}]

# Constraints
linear_constraints = [
    {
        'P': np.array([-3, 3]),
        'n': np.array([-1, 4])
    }
]
multi_constraints = [
    {
        'H': rm(40) @ np.array([[1, 0], [0, 1.5]]) @ rm(40).T,
        'c': np.array([-4.0, -3.0]),
        'b': 2.0,
    },
    {
        'H': rm(-45) @ np.array([[1.5, 0], [0, 2]]) @ rm(-45).T,
        'c': np.array([-3.0, 3.0]),
        'b': 0.75,
    },
    {
        'H': rm(45) @ np.array([[1.5, 0], [0, 1]]) @ rm(45).T,
        'c': np.array([-2.0, 0.0]),
        'b': 0.2,
    },
    {
        'H': rm(10) @ np.array([[1.5, 0], [0, 1]]) @ rm(10).T,
        'c': np.array([-1.0, -3.0]),
        'b': 1.0,
    },
    {
        'H': rm(30) @ np.array([[1, 0], [0, 40]]) @ rm(30).T,
        'c': np.array([-1.0, 2.0]),
        'b': 3.5,
    },
    {
        'H': rm(40) @ np.array([[1, 0], [0, 1.5]]) @ rm(40).T,
        'c': np.array([-0.5, 3.5]),
        'b': 0.15,
    },
    {
        'H': rm(70) @ np.array([[1.0, 0], [0, 2.5]]) @ rm(70).T,
        'c': np.array([3.0, 1.0]),
        'b': 12.0,
    },
    {
        'H': rm(60) @ np.array([[1.5, 0], [0, 1]]) @ rm(60).T,
        'c': np.array([3.0, 4.0]),
        'b': 1.0,
    },
    {
        'H': rm(-30) @ np.array([[1.5, 0], [0, 1]]) @ rm(-30).T,
        'c': np.array([1.5, -3.5]),
        'b': 0.5,
    },
]
constraints = {
    'Linear': linear_constraints,
    'Quadratic': [],
    'Multi': [[{'Quadratic': [c]} for c in multi_constraints]]
}

# Create problem
problem = CobiProblem(
    n_var=2,
    objectives=objectives,
    constraints=constraints,
    domain=(-5.0, 5.0)
)

# Calculate Pareto set and front
print("Computing the Pareto set and front (might take a while)...")
sampling_options = {'sampling': 'equi-w', 'n_points': 100}
problem.calculate_pareto_set_and_front(sampling_options=sampling_options)

# Run NSGA-II for comparison
print("Running the NSGA-II algorithm on the problem...")
algorithm_n_evals = 100000
nda, diff = run_algorithm_track_diff_to_opt(algorithm=NSGA2(), problem=problem, n_evals=algorithm_n_evals, seed=1, indicator='HV+')

# Visualize results
print("Visualizing the results...")
axes = problem.get_figure(multi_constraint_single_label=True,
                          plot_large_peak_centers=False,
                          algorithm_name='NSGA-II',
                          algorithm_X=np.array([list(d['x']) for d in nda.infos]),
                          algorithm_F=np.array(list(nda)),
                          plot_normalized_front=True)
axes[0].legend().remove()
os.makedirs('results', exist_ok=True)
problem.save_figure(plot_name=f'multimodal', show=False, save=True, folder='results', extension='pdf')
print("Done. Plot saved to 'results/multimodal.pdf'.")
