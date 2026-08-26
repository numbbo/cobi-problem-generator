import math


EPS = 1e-100


# Objectives

def transform_exponent(x, exponent, shift):
    """ Returns (x - shift)^exponent + shift. """
    return (x - shift) ** exponent + shift


def transform_step(x, value, shift):
    """ Adds shift to x if it is greater than value. """
    return x + shift if x >= value else x


def transform_logarithm(x, base, shift):
    """ Returns log_base(x - shift + EPS) + shift. """
    if base <= 1:
        raise ValueError(f"Base for logarithmic transformation must be greater than 1, otherwise it is not a strictly increasing transformation. Got base={base}.")
    return math.log(x - shift + EPS, base) + shift


def transform_objective(x, transformation):
    """ Transforms the objective vector. """
    if transformation is None:
        return x
    transformations = {
        "exponent": transform_exponent,
        "step": transform_step,
        "logarithm": transform_logarithm
    }
    return transformations[transformation["name"]](x, **transformation["params"])


# Constraints

def transform_mask(x):
    """ Returns 0 if feasible, 1 otherwise. """
    return 0 if x <= 0 else 1


def transform_constraint(x, transformation):
    """ Transforms the scalar returned by constraint. """
    if transformation is None:
        return x
    transformations = {
        "mask": transform_mask
    }
    return transformations[transformation["name"]](x, **transformation["params"])