# COBI Problem Generator

This repository contains a problem generator of COBI (COnstrained Bi-objective optimization) problems 
that can be used for benchmarking optimization algorithms. The problems can have:
- Any number of real-valued variables
- Two multipeak objective functions to be minimized 
- Any number of inequality constraints that can be linear, convex-quadratic, multipeak or a combination of these types.

The Pareto set and Pareto front of a COBI problem can be approximated to arbitrary accuracy.


## Installation

Download the repository as a zip file and extract it to `cobi-problem-generator`. Then navigate to the repository directory and install the required dependencies and the package:

```bash
cd cobi-problem-generator
pip install -r requirements.txt
pip install .
```

If you plan to modify the source code, install in editable mode:

```bash
pip install -e .
```

In editable mode, changes to the source files are immediately available without reinstalling the package.

## Test installation

You can test the installation by running the example problems in the `examples` folder.
For example, you can run the `user_problem.py` with:

```bash
python examples/user_problem.py
```
If the installation is correct, you should see the message:

```
Creating a user-defined COBI problem...
```

This will be followed by information about the problem in the console. After a short moment, a figure will appear showing a visualization of the search space and objective space, including the constraints, the calculated Pareto set and front, and the points found by NSGA-II for comparison. The results will be saved to `results/test_user_problem_results.pkl`.

## Defining COBI problems

You can create your own COBI problems with

```python
from cobi import CobiProblem

my_problem = CobiProblem(
    n_var=n_var,
    objectives=objectives,
    constraints=constraints,
    domain=(-5, 5),
    alpha=(2, 0.5),
    boundary_constraints=True
)
```

Here `n_var` is the number of decision variables (dimension of the search space), `objectives` is a tuple with two elements defining the objective functions (described below), `constraints` is the dictionary describing the constraints 
(described below), `domain` is the domain in the decision space (lower and upper bound for decision variables), `alpha` is 
used for transformation of the objective function and if `boundary_constraints` is `True`, then boundary constraints for each decision variable enforcing the domain are also added to constraints.

See `user_problem.py`, `multimodal_problem.py`, and `one_dimensional_problem.py` in the `examples` folder.

### Objectives

A **transformed strictly convex-quadratic function** is defined as:

```
f(x) = (0.5 * (x - c)^T H (x - c)) ^ alpha + b
```

where:

- `H` is a symmetric positive definite matrix of shape `(n_var, n_var)`  
- `c` is a vector of length `n_var`  
- `alpha` is a positive scalar  
- `b` is an arbitrary scalar  

This function has a **unique local minimum** at `c`, where its value is `b`.

---

More complex **multipeak functions** can be formed by taking the minimum of multiple transformed strictly convex-quadratic functions `f_i(x)`:

```
F(x) = min_i f_i(x)
```

Such functions can have multiple local extrema and more complex shapes.

---

Our **bi-objective functions** are of the form:

```
(F_1(x), F_2(x))
```

where `F_1(x)` and `F_2(x)` are multipeak functions.  

The objectives passed to `CobiProblem` are represented as a tuple `(F_1(x), F_2(x))`.
Each multipeak function is represented by a **dictionary** containing the keys:

- `H` – list of matrices `H_i` for all transformed strictly convex-quadratic components in the minimum of the multipeak function 
- `c` – list of vectors `c_i`  
- `b` – list of offsets `b_i`  
- `alphas` – list of exponents `alpha_i`  

These lists correspond to all the individual transformed strictly convex-quadratic functions that form the minimum defining the multipeak function.

---

We can additionally transform the bi-objective functions by first defining `m_1` and `m_2` as the minimal values of `F_1(x)` and `F_2(x)`, respectively, and then applying:

```
F*_k(x) = (F_k(x) - m_k) ^ alpha_k + m_k
```

Here `alpha_k` are positive scalars. Then our bi-objective function is:
```
 (F*_1(x), F*_2(x))
```

This transformation can significantly affect the shape of the Pareto front and allows the construction of problems with concave or partially convex and partially concave Pareto fronts.  
The **`alpha` parameter** passed to `CobiProblem` corresponds to the components `alpha_1` and `alpha_2` for the two objectives.

### Constraints

COBI problems can include three types of constraints:

1. **Linear constraints**  
   Defined with scalar product as:
   ```
   <x - P, n> <= 0
   ```

   Such a constraint is encoded as a dictionary with the keys:
   ```python
   {'P': P_vector, 'n': n_vector}
   ```

2. **Strictly convex-quadratic constraints**  
   Defined as:
   ```
   (x - c)^T H (x - c) <= b
   ```
   Here `H` is a symmetric positive definite matrix.  

   Such a constraint is encoded as a dictionary with the keys:
   ```python
   {'c': c_vector, 'H': H_matrix, 'b': b_scalar}
   ```

3. **Multipeak constraints**  
   These are of the form:

   ```
   min_k [ max_l [ g_{k,l}(x) ] ] <= 0
   ```
   Here `g_{k,l}` are either linear or convex-quadratic constraints.  

   Such a constraint is encoded as a list of groups that appear in the minimum, with each group encoded as a dictionary containing lists of linear and convex-quadratic constraints that appear in it:
   ```python
   [
       {'Linear': linear_constraints_1, 'Quadratic': quadratic_constraints_1},
       ...,
       {'Linear': linear_constraints_v, 'Quadratic': quadratic_constraints_v}
   ]
   ```

   Here, each `linear_constraints_k` and `quadratic_constraints_k` is a list of linear or convex-quadratic constraints that appear in `{g_{k,1}, ..., g_{k,u_k}}`.

---

All constraints are passed to `CobiProblem` as a **single dictionary** with the keys `Linear`, `Quadratic`, and `Multi`, each containing a list of the corresponding constraints:

```python
constraints = {
    'Linear': [...],
    'Quadratic': [...],
    'Multi': [...]
}
```

## Creating Random Problems

You can generate random COBI problems using the `create_random_problem` function.  
This is useful for testing algorithms or creating benchmark problems.

### Example

```python
from cobi import create_random_problem

# Generate a random problem with 2 variables
problem = create_random_problem(n_var=2, seed=1)

# Print basic information about the problem
print(problem)
```

This function allows you to control many aspects of the generated problem, such as:

- `n_var` – number of decision variables
- `seed` – random seed for reproducibility
- `domain` – lower and upper bounds for each decision variable 
- `n_peaks` – number of peaks for each objective function
- `alpha` – transformation exponents for the objectives
- `n_constraints` – number of constraints of each type (`Linear`, `Quadratic`, `Multi`)
- `boundary_constraints` – whether to automatically add boundary constraints
- `constraints_feasible` – ensure some feasible points exist
- And others controlling the size, shape, and condition numbers of objectives and constraints  

For a complete example, see `random_problem.py` in the `examples` folder, which demonstrates how to generate a random problem, compute its Pareto set and front, and visualize its objectives and constraints.

## Calculating Pareto Set and Front

Once you have a `CobiProblem` instance, you can compute its Pareto set and Pareto front using the method:

```python
problem.calculate_pareto_set_and_front(
    sampling_options=None,
    tol_feasible=1e-8,
    skip_dominated=True,
    solver=None,
    print_output=False
)
```

### What this method computes

This method sets several attributes of your `CobiProblem` instance:

- `local_unconstrained_pareto_fronts`: local unconstrained Pareto fronts for each pair of individual peaks  
- `local_unconstrained_pareto_sets`: local unconstrained Pareto sets for each pair of individual peaks  
- `uncon_pareto_front`: global unconstrained Pareto front  
- `uncon_pareto_set`: global unconstrained Pareto set  
- `uncon_pareto_source`: origin (peak indices, and weight) of each point in the global unconstrained Pareto set  
- `local_pareto_fronts`: local Pareto fronts after projection onto the feasible region  
- `local_pareto_sets`: local Pareto sets after projection onto the feasible region  
- `pareto_front`: global Pareto front after feasible projection  
- `pareto_set`: global Pareto set after feasible projection  
- `pareto_source`: origin (peak indices, multi-constraint group index, and weight) of each point in the global Pareto set  
The attributes that are actually computed, and the accuracy of their values, depend on the chosen sampling and computation options.

---

### Key Parameters

- `sampling_options` – dictionary specifying how points are sampled along the Pareto sets/fronts:  
  - `equi-w`: sample `n_points` using equidistant weights  
  - `equi-uncon-x`: sample points with approximately equal Euclidean distances along local unconstrained Pareto sets  
  - `equi-x`: distance between consecutive points on projected Pareto set  
  - `equi-f`: distance between consecutive points on Pareto front  
  - `max-HV`: sample until maximal theoretical hypervolume error is below `max_error` or Pareto set contains `max_points`
  - `rectangles`: sample points from local Pareto sets using rectangles decomposition  
  - `edge`: sample only edge points from each local Pareto set  
Each of these methods uses additional parameters, which can be included in the dictionary to control the accuracy of the approximations.

- `tol_feasible` – tolerance for considering a projected point feasible  
- `skip_dominated` – whether to skip points dominated by others  
- `solver` – solver used for projecting points onto the feasible region (e.g., `"daqp"`, `"cvxpy_SCS"`, `"kkt"`). If `None`, the solver is chosen automatically based on the type of constraints
- `print_output` – whether to print progress

---

### Example

```python
# Compute Pareto set and front
problem.calculate_pareto_set_and_front(
    sampling_options={'sampling': 'equi-w', 'n_points': 50},
    tol_feasible=1e-8
)

# Access the global Pareto set and front
pareto_set = problem.pareto_set
pareto_front = problem.pareto_front
```

This will generate both **unconstrained** and **feasible** Pareto sets/fronts according to the chosen sampling options.

## Normalizing the problem

You can normalize the problem with:

```python
problem.normalize_problem()
```

This sets normalization constants internally for further calculations. The Pareto set and front need to be recomputed if they have already been computed.

## Characterizing, Saving, and Visualizing the Problem

After computing the Pareto set and front, `CobiProblem` provides several methods to analyze, save, and visualize the problem.

---

### **1. Get Ideal and Nadir Points**

- **Nadir point** – the worst objective values on the Pareto front:
```python
nadir = problem.nadir_point()
```

- **Ideal point** – the best objective values on the Pareto front:
```python
ideal = problem.ideal_point()
```

> If the Pareto front is not yet computed, these methods will automatically compute it (using `edge` sampling).

---

### **2. Characterize the Problem**

Get a full characterization including feasibility, hypervolume, active constraints, and Pareto set/front structure:

```python
characterization = problem.characterize_problem(dist_thresh_set=0.25, dist_thresh_front=0.1)
print(characterization)
```

Returned dictionary includes:

- `feasible` – whether a feasible Pareto set exists  
- `nadir` – nadir point  
- `ideal` – ideal point  
- `hypervolume` – approximated hypervolume of the Pareto front  
- `normalized_hypervolume` – approximated normalized hypervolume  
- `active_constraints` – sets of constraints active at Pareto points  
- `pareto_set_parts` – number of disconnected parts in the Pareto set (if `dist_thresh_set` is set)  
- `pareto_front_parts` – number of disconnected parts in the Pareto front (if `dist_thresh_front` is set) 

You can also calculate **binding constraints** (constraints whose removal changes the Pareto set) with `calculate_binding_constraints`.

---

### **3. Save and Load the Problem**

Save the problem with all computed results:

```python
problem.save_problem('my_problem.pkl')
```

Later you can load it using `load_problem` function.

---

### **4. Visualize the Problem**

`CobiProblem` provides visualization method:

```python
problem.visualize(
    algorithm_X=alg_X,
    algorithm_F=alg_F,
    algorithm_name='My Algorithm',
    show=True,
    save=True
)
```

Options include:

- `plot_objective_space`, `plot_search_space` – show objective and search space  
- `plot_unconstrained_pareto`, `plot_constrained_pareto` – show unconstrained and constrained Pareto sets/fronts  
- `plot_normalized_front`, `normalize_algorithm` – normalize values between ideal and nadir points  
- `algorithm_X`, `algorithm_F`, `algorithm_name` – overlay algorithm solutions  
- `show`, `save` – display or save the figure  
- ...

---

### **5. Inspect Problem Details**

Print a summary of the problem:

```python
print(problem)
```

This outputs number of decision variables, domain, alpha, objectives, constraints, and other problem properties.


## Choice of solvers

For problems with only linear constraints, we use the [DAQP](https://pypi.org/project/daqp/) [1] solver from the [qpsolvers](https://github.com/qpsolvers/qpsolvers) [2] module. Our choice of DAQP is based on preliminary experiments with a variety of Python-based solvers (namely [CVXOPT](https://cvxopt.org/) [3], [DAQP](https://pypi.org/project/daqp/) [1], [PIQP](https://pypi.org/project/piqp/) [4], [ProxQP](https://pypi.org/project/proxsuite/) [5], and [quadprog](https://pypi.org/project/quadprog/) (using the Goldfarb/Idnani dual algorithm [6]), all accessed via the [qpsolvers](https://github.com/qpsolvers/qpsolvers) module), in which DAQP showed up as the most precise and most CPU-efficient alternative.

When a problem contains also nonlinear constraints, we use the [SCS](https://pypi.org/project/scs/) [7] solver  from the [CVXPY](https://www.cvxpy.org/) [8, 9] module. The choice of SCS is based on preliminary experiments with a variety of Python-based solvers (namely [ECOS](https://pypi.org/project/ecos/) [10], [SCS](https://pypi.org/project/scs/) [7], [MOSEK](https://www.mosek.com/) [11], [GUROBI](https://www.gurobi.com/) [12], all accessed via the [CVXPY](https://www.cvxpy.org/) module), in which no solver performed notably faster or more precisely than SCS.


## References

[1] D. Arnström, A. Bemporad, and D. Axehill,  
    “A Dual Active-Set Solver for Embedded Quadratic Programming Using Recursive LDLᵀ Updates,”  
    *IEEE Transactions on Automatic Control*, vol. 67, no. 8, pp. 4362–4369, 2022.  
    https://doi.org/10.1109/TAC.2022.3176430

[2] S. Caron, D. Arnström, S. Bonagiri, A. Dechaume, N. Flowers, A. Heins, et al.,  
    *qpsolvers: Quadratic Programming Solvers in Python*, version 4.8.0, 2025.  
    https://github.com/qpsolvers/qpsolvers

[3] M. S. Andersen, J. Dahl, and L. Vandenberghe,  
    *CVXOPT: A Python package for convex optimization*, version 1.3.2, 2025.  
    https://cvxopt.org/

[4] R. Schwan, Y. Jiang, D. Kuhn, and C. N. Jones,  
    “PIQP: A Proximal Interior-Point Quadratic Programming Solver,”  
    *IEEE Conference on Decision and Control (CDC)*, pp. 1088–1093, 2023.  
    https://doi.org/10.1109/CDC49753.2023.10383915

[5] A. Bambade, S. El-Kazdadi, A. Taylor, and J. Carpentier,  
    “PROX-QP: Yet another Quadratic Programming Solver for Robotics and beyond,”  
    *Robotics: Science and Systems (RSS)*, 2022.  
    https://inria.hal.science/hal-03683733

[6] D. Goldfarb and A. U. Idnani,  
    “A numerically stable dual method for solving strictly convex quadratic programs,”  
    *Mathematical Programming*, vol. 27, pp. 1–33, 1983.  
    https://doi.org/10.1007/BF02591962

[7] B. O’Donoghue,  
    “Operator Splitting for a Homogeneous Embedding of the Linear Complementarity Problem,”  
    *SIAM Journal on Optimization*, vol. 31, no. 3, pp. 1999–2023, 2021.

[8] S. Diamond and S. Boyd,  
    “CVXPY: A Python-embedded modeling language for convex optimization,”  
    *Journal of Machine Learning Research*, vol. 17, no. 83, pp. 1–5, 2016.

[9] A. Agrawal, R. Verschueren, S. Diamond, and S. Boyd,  
    “A rewriting system for convex optimization problems,”  
    *Journal of Control and Decision*, vol. 5, no. 1, pp. 42–60, 2018.

[10] A. Domahidi, E. Chu, and S. Boyd,  
     “ECOS: An SOCP solver for embedded systems,”  
     *European Control Conference (ECC)*, pp. 3071–3076, 2013.

[11] MOSEK ApS,  
     *MOSEK Optimizer API for Python*, version 11.0.22, 2025.  
     https://docs.mosek.com/11.0/pythonapi/index.html

[12] Gurobi Optimization, LLC,  
     *Gurobi Optimizer Reference Manual*, 2025.  
     https://www.gurobi.com
