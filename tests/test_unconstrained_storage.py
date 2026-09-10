import unittest

import numpy as np

from cobi import CobiProblem


class TestUnconstrainedStorage(unittest.TestCase):
    def test_equi_x_keeps_unconstrained_points_when_not_recomputed(self):
        objectives = (
            {
                'H': np.array([[[1.0]]]),
                'c': np.array([[-1.0]]),
                'b': np.array([0.0]),
                'transformation': None,
                'peak_transformations': [None]
            },
            {
                'H': np.array([[[1.0]]]),
                'c': np.array([[1.0]]),
                'b': np.array([0.0]),
                'transformation': None,
                'peak_transformations': [None]
            }
        )
        constraints = {
            'Linear': [{'P': np.array([0.0]), 'n': np.array([-1.0])}],
            'Quadratic': [],
            'Multi': []
        }
        problem = CobiProblem(
            n_var=1,
            objectives=objectives,
            constraints=constraints,
            domain=(-2, 2),
            boundary_constraints=False
        )

        sampling_options = {
            'sampling': 'equi-x',
            'distance': 0.1,
            'tol_distance': 1e-8,
            'tol_jump': 1e-3,
            'max_iter': 1000,
            'force_equidistant': False,
            'always_compute_unconstrained': False
        }
        problem.calculate_pareto_set_and_front(
            sampling_options=sampling_options,
            solver='scipy_COBYLA',
            print_output=False
        )

        self.assertGreater(len(problem.pareto_set), 0)
        self.assertGreater(len(problem.uncon_pareto_set), 0)
        self.assertGreater(len(problem.uncon_pareto_front), 0)
        self.assertGreater(len(problem.uncon_pareto_source), 0)


if __name__ == '__main__':
    unittest.main()
