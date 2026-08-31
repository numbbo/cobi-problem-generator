# COBI Problem Generator

This repository contains a problem generator of COBI (COnstrained BI-objective optimization) problems 
that can be used for benchmarking optimization algorithms. The problems can have:
- Any number of real-valued variables
- Two multipeak objective functions to be minimized 
- Any number of inequality constraints that can be linear, convex-quadratic, multipeak or a combination of these types.

The Pareto set and Pareto front of a COBI problem can be approximated to arbitrary accuracy.

## Installation

To avoid issues we suggest a clean environment. You can make it with Conda:

```bash
conda create -n cobi -c conda-forge python=3.12 numpy=1.26.4 scipy scikit-learn -y
conda activate cobi
```

Download the repository as a ZIP file and extract it. Then navigate to the extracted repository folder containing `requirements.txt` and install the dependencies and the package:

```bash
pip install -r requirements.txt
pip install .
```

If you plan to modify the source code, install in editable mode:

```bash
pip install -e .
```

In editable mode, changes to the source files are immediately available without reinstalling the package.

## Test Installation

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

## Defining COBI Problems

You can create your own COBI problems with

```python
from cobi import CobiProblem

my_problem = CobiProblem(
    n_var=n_var,
    objectives=objectives,
    constraints=constraints,
    domain=(-5, 5),
    boundary_constraints=True
)
```

Here `n_var` is the number of decision variables (dimension of the search space), `objectives` is a tuple with two elements defining the objective functions (described below), `constraints` is the dictionary describing the constraints 
(described below), `domain` is the domain in the decision space (lower and upper bound for decision variables), and if `boundary_constraints` is `True`, then the domain bounds are also enforced as constraints.

The old `alpha` argument is still accepted for backward compatibility. Objective transformations should now be specified in the objective dictionaries as described below.

See `user_problem.py`, `multimodal_problem.py`, `one_dimensional_problem.py`, and `simple_examples.py` in the `examples` folder.

### Objectives

A **strictly convex-quadratic function** is defined as:

```
f(x) = 0.5 * (x - c)^T H (x - c) + b
```

where:

- `H` is a symmetric positive definite matrix of shape `(n_var, n_var)`  
- `c` is a vector of length `n_var`  
- `b` is an arbitrary scalar  

This function has a **unique local minimum** at `c`, where its value is `b`. An optional strictly increasing transformation `T` can be applied to such individual function.

---

More complex **multipeak functions** can be formed by taking the minimum of multiple optionally transformed strictly convex-quadratic functions `q_i(x)`:

```
F(x) = min_i q_i(x)
```

Such functions can have multiple local extrema and more complex shapes.

---

Our **bi-objective functions** are of the form:

```
(T_1(F_1(x)), T_2(F_2(x)))
```

where `F_1(x)` and `F_2(x)` are multipeak functions and `T_1`, `T_2` are optional strictly increasing transformations.  

The objectives passed to `CobiProblem` are represented as a tuple of two dictionaries. Each dictionary contains:

- `H` – list of matrices `H_i` for all optionally transformed strictly convex-quadratic components in the minimum of the multipeak function 
- `c` – list of vectors `c_i`  
- `b` – list of offsets `b_i`  
- `peak_transformations` – list containing an optional transformation for each peak  
- `transformation` – optional transformation applied to the complete multipeak objective  

A transformation is either `None` or a dictionary with the keys `name` and `params`. Currently supported objective transformations are `exponent`, `step`, and `logarithm`. For `exponent` and `logarithm`, the shift is set automatically.

The old `alphas` entries in objective dictionaries and the `alpha` argument of `CobiProblem` are deprecated and are kept only for backward compatibility.

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

Linear and strictly convex-quadratic constraints, including those inside multipeak constraints, can optionally be transformed by adding a `transformation` entry to their dictionary. A transformation is either `None` or a dictionary with the keys `name` and `params`. Currently, the supported constraint transformation is `mask`, which maps every feasible constraint value (`<= 0`) to `0` and every violated value (`> 0`) to `1`.

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
- `n_constraints` – number of constraints of each type (`Linear`, `Quadratic`, `Multi`)
- `boundary_constraints` – whether to enforce the domain bounds as constraints
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

### Computed Attributes

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

## Normalizing the Problem

You can normalize the problem with:

```python
problem.normalize_problem()
```

This sets normalization constants internally for further calculations. The Pareto set and front need to be recomputed if they have already been computed.

## Working with the Problem

`CobiProblem` provides several methods to analyze, save, and visualize the problem.

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

### **5. Print the Problem**

Print a summary of the problem:

```python
print(problem)
```

This outputs number of decision variables, domain, objectives, constraints, transformations, and other problem properties.

## Choice of Solvers

For problems with only linear constraints, we use the [DAQP](https://pypi.org/project/daqp/) solver from the [qpsolvers](https://github.com/qpsolvers/qpsolvers) module. When a problem contains also nonlinear constraints, we use the [COBYLA](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html) solver from the [SciPy](https://docs.scipy.org/doc/scipy/) module. The [SCS](https://pypi.org/project/scs/) solver through [CVXPY](https://www.cvxpy.org/) is also available.
