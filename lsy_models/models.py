"""This file returns a selected model with the selected config."""

from functools import partial

import casadi as cs

import lsy_models.numeric as num
import lsy_models.symbolic as sym
import lsy_models.utils.const as const

# available methods, used in testing
available_configs = ["cf2x-", "cf2x+"]
available_models = ["first_principles"] 

def dynamics(model: str, config: str, symbolic: bool = False) -> callable:
    """This methods lets you select the dynamics function f(x,u).
    
    TODO.
    """
    match config: # TODO make constants in jp or np
        case "cf2x-":
            C = const.Constants.create("data/cf2x_-B250.xml")
        case "cf2x+":
            C = const.Constants.create("data/cf2x_+B250.xml")
        case _:
            raise ValueError(f"Drone config '{config}' is not supported")
        
    match model:
        case "first_principles":
            if not symbolic:
                return partial(num.f_first_principles, C=C)
            else:
                X_dot, X, U, Y = sym.first_principles(C)
                return cs.Function('first_principles', [X, U], [X_dot])
        case "fit_SI":
            raise ValueError(f"Model '{model}' is not supported") # TODO
        case "fit_DI":
            raise ValueError(f"Model '{model}' is not supported") # TODO
        case _:
            raise ValueError(f"Model '{model}' is not supported")
        

# TODO method for a casadi optimizer object