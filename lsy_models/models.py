"""This file returns a selected model with the selected config."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import casadi as cs

from lsy_models import (
    numeric,  # TODO would be nice to directly import all the functions -> currently they are called the same though
    symbolic,
)
from lsy_models.utils.constants import Constants

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

    from lsy_models.dataclasses import QuadrotorState

# used in testing
available_models = ["first_principles"]  # , "fitted_SI", "fitted_DI"
# available_models = ({"name": "first_principles", "continuous": True},
#                     ...) # TODO


def dynamics_numeric(
    model: str, config: str
) -> Callable[[QuadrotorState, NDArray[np.floating]], QuadrotorState]:  # TODO how precise?
    """Creates a numerical dynamics function f(x,u).

    Args:
        model: The chosen dynamical model. See available_models
        config: The chosen config/constants for the model. See Constants.available_configs

    Returns:
        A callable which takes the states and input and returns the derivative of the state.
    """
    constants = Constants.from_config(config)

    match model:
        case "first_principles":
            return partial(numeric.f_first_principles, constants=constants)
        case "fitted_SI":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case "fitted_DI":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case _:
            raise ValueError(f"Model '{model}' is not supported")


def dynamic_numeric_from_symbolic(model: str, config: str) -> Callable[[NDArray, NDArray], NDArray]:
    """Creates a numerical dynamics function f(x,u) from a CasADi model.

    Args:
        model: The chosen dynamical model. See available_models
        config: The chosen config/constants for the model. See Constants.available_configs

    Returns:
        A callable which takes the states and input and returns the derivative of the state.
    """
    constants = Constants.from_config(config)

    match model:
        case "first_principles":
            X_dot, X, U, _ = symbolic.first_principles(constants)
            return cs.Function("first_principles", [X, U], [X_dot])
        case "fitted_SI":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case "fitted_DI":
            raise ValueError(f"Model '{model}' is not supported")  # TODO
        case _:
            raise ValueError(f"Model '{model}' is not supported")


# TODO method for a casadi optimizer object
