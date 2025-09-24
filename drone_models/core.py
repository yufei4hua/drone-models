"""Core tools for registering and capability checking for the drone models."""

from __future__ import annotations

import tomllib
import warnings
from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, Protocol, TypeVar, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from array_api_typing import Array


F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


def supports(rotor_dynamics: bool = True) -> Callable[[F], F]:
    """Decorator to indicate which features are supported."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(
            pos: Array,
            quat: Array,
            vel: Array,
            ang_vel: Array,
            cmd: Array,
            rotor_vel: Array | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Array, Array, Array, Array, Array | None]:
            if not rotor_dynamics and rotor_vel is not None:
                raise ValueError("Rotor dynamics not supported, but rotor_vel is provided.")
            if rotor_dynamics and rotor_vel is None:
                warnings.warn("Rotor velocity not provided, using commanded rotor velocity.")
            return fn(pos, quat, vel, ang_vel, cmd, rotor_vel, *args, **kwargs)

        wrapper.__drone_model_features__ = {"rotor_dynamics": rotor_dynamics}

        return wrapper  # type: ignore

    return decorator


model_parameter_registry: dict[str, type[ModelParams]] = {}


def named_tuple2xp(params: ModelParams, xp: ModuleType, device: str | None = None) -> ModelParams:
    """Convert a named tuple to an array API framework."""
    return params.__class__(
        **{k: xp.asarray(v, device=device) for k, v in params._asdict().items()}
    )


def parametrize(
    fn: Callable[P, R], drone_model: str, xp: ModuleType | None = None, device: str | None = None
) -> Callable[P, R]:
    """Parametrize a controller function with the default controller parameters for a drone model.

    Args:
        fn: The controller function to parametrize.
        drone_model: The drone model to use.
        xp: The array API module to use. If not provided, numpy is used.
        device: The device to use. If none, the device is inferred from the xp module.

    Example:
        ```python
        controller_fn = parametrize(state2attitude, drone_model="cf2x_L250")
        command_rpyt, int_pos_err = controller_fn(
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            cmd=cmd,
            ctrl_errors=(int_pos_err,),
            ctrl_freq=100,
        )
        ```

    Returns:
        The parametrized controller function with all keyword argument only parameters filled in.
    """
    model_id = fn.__module__ + "." + fn.__name__
    try:
        params = model_parameter_registry[model_id].load(drone_model)
        if xp is not None:  # Convert to any array API framework
            params = named_tuple2xp(params, xp=xp, device=device)
    except KeyError as e:
        raise KeyError(
            f"Model `{model_id}` does not exist in the parameter registry for drone `{drone_model}`"
        ) from e
    except ValueError as e:
        raise ValueError(f"Drone model `{drone_model}` not supported for `{model_id}`") from e
    return partial(fn, **params._asdict())


@runtime_checkable
class ModelParams(Protocol):
    """Protocol for model parameters."""

    @staticmethod
    def load(drone_model: str) -> ModelParams:
        """Load the parameters from the config file."""

    def _asdict(self) -> dict[str, Any]:
        """Convert the parameters to a dictionary."""


def register_model_parameters(
    params: ModelParams | type[ModelParams],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Register the default model parameters for this model.

    Warning:
        The model parameters **must** be a named tuple with a function `load` that takes in the
        drone model name and returns an instance of itself, or a class that implements the
        ModelParams protocol.

    Args:
        params: The model parameter type.

    Returns:
        A decorator function that registers the parameters and returns the function unchanged.
    """
    if not isinstance(params, ModelParams):
        raise ValueError(f"{params} does not implement the ModelParams protocol")

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        controller_id = fn.__module__ + "." + fn.__name__
        if controller_id in model_parameter_registry:
            raise ValueError(f"Model `{controller_id}` already registered")
        model_parameter_registry[controller_id] = params
        return fn

    return decorator


def load_params(physics: str, drone_model: str, xp: ModuleType | None = None) -> dict:
    """TODO.

    Args:
        physics: _description_
        drone_model: _description_
        xp: The array API module to use. If not provided, numpy is used.

    Returns:
        dict[str, Array]: _description_
    """
    xp = np if xp is None else xp
    with open(Path(__file__).parent / "data/params.toml", "rb") as f:
        physical_params = tomllib.load(f)
    if drone_model not in physical_params:
        raise KeyError(f"Drone model `{drone_model}` not found in data/params.toml")
    with open(Path(__file__).parent / f"{physics}/params.toml", "rb") as f:
        model_params = tomllib.load(f)
    if drone_model not in model_params:
        raise KeyError(f"Drone model `{drone_model}` not found in model params.toml")
    params = physical_params[drone_model] | model_params[drone_model]
    params["J_inv"] = np.linalg.inv(params["J"])
    params = {k: xp.asarray(v) for k, v in params.items()}  # if k in fields
    return params
