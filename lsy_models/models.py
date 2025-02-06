"""This file returns a selected model with the selected config."""

from functools import partial

import casadi as cs

import lsy_models.numeric as num
import lsy_models.symbolic as sym
from lsy_models.utils.constants import Constants

# available methods, used in testing
available_configs = ["cf2x-", "cf2x+"]
available_models = ["first_principles"]


def dynamics(model: str, config: str, symbolic: bool = False) -> callable:
    """This methods lets you select the dynamics function f(x,u).

    TODO.
    """
    constants = Constants.from_config(config)

    match model:
        case "first_principles":
            if not symbolic:
                return partial(num.f_first_principles, constants=constants)
            else:
                X_dot, X, U, _ = sym.first_principles(constants)
                return cs.Function("first_principles", [X, U], [X_dot])
        case "fit_SI":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case "fit_DI":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case _:
            raise ValueError(f"Model '{model}' is not supported")


# TODO method for a casadi optimizer object
