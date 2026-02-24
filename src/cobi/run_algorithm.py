import numpy as np
from .problem_generator import CobiProblem
from moarchiving import get_mo_archive, get_cmo_archive
from matplotlib import pyplot as plt


def run_algorithm_track_diff_to_opt(problem: CobiProblem, algorithm, indicator, n_evals, seed):
    """ Run the given algorithm on the provided problem while tracking the (normalized) indicator difference to the
    optimum. Possible indicators:
    - 'HV': Hypervolume
    - 'HV+': Hypervolume plus (equals HV when solutions dominate the reference point and distance to the reference
    point otherwise)
    - 'ICMOP': ICMOP indicator (equals HV+ when solutions are feasible and tau + distance to reference point otherwise)
    """
    hv_ref = problem.normalized_hypervolume
    ideal = problem.ideal_point()
    nadir = problem.nadir_point()
    if indicator in ['HV', 'HV+']:
        nda = get_mo_archive(reference_point=np.ones_like(nadir))
    else:
        nda = get_cmo_archive(reference_point=np.ones_like(nadir))
    algorithm.setup(problem, termination=('n_evals', n_evals), verbose=False, seed=seed)
    diff = []

    while algorithm.has_next():
        algorithm.next()
        pop = algorithm.unsorted_pop
        for x, f, cv in zip(pop.get('X'), pop.get('F'), pop.get('CV')):
            if indicator == 'HV' and cv <= 0:
                nda.add((f - ideal) / (nadir - ideal), info={'x': x})
                diff.append(hv_ref - float(nda.hypervolume))
            elif indicator == 'HV+' and cv <= 0:
                nda.add((f - ideal) / (nadir - ideal), info={'x': x})
                diff.append(hv_ref - float(nda.hypervolume_plus))
            elif indicator == 'ICMOP':
                nda.add((f - ideal) / (nadir - ideal), cv, info={'x': x})
                diff.append(hv_ref - float(nda.hypervolume_plus_constr))
    return nda, diff[:n_evals]


def plot_algorithm_performance(file_name, performances, names, labels, colors,
                               title=None, x_label='Solution evaluations',
                               y_label='Average difference to the\napproximated optimal hypervolume',
                               y_lim = (1e-5, 1),  fig_size=(3.5, 3.5), **file_options):
    fig, ax = plt.subplots(figsize=fig_size)
    for performance, name, label, color in zip(performances, names, labels, colors):
        ax.plot(1 + np.arange(len(performance)), performance, label=label, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yscale('log')
    ax.set_ylim(y_lim)
    ax.set_title(title)
    ax.grid(True, which='major', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{file_options['folder']}/{file_name}.{file_options['extension']}", dpi=file_options['dpi'])
    plt.close()
