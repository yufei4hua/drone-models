"""TODO."""

from functools import wraps
from typing import Any, Callable, TypeVar

from array_api_typing import Array

from drone_models.utils.constants import Constants

F = TypeVar("F", bound=Callable[..., Any])


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
            constants: Constants,
            rotor_vel: Array | None = None,
            dist_f: Array | None = None,
            dist_t: Array | None = None,
        ) -> tuple[Array, Array, Array, Array, Array | None]:
            if not rotor_dynamics and rotor_vel is not None:
                raise ValueError("Rotor dynamics not supported, but rotor_vel is provided.")
            return fn(pos, quat, vel, ang_vel, cmd, constants, rotor_vel, dist_f, dist_t)

        wrapper.__drone_model_features__ = {"rotor_dynamics": rotor_dynamics}

        return wrapper  # type: ignore

    return decorator
