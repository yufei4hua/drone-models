"""Models of quadrotor drones for estimation and control."""

from typing import Callable

from drone_models.models.first_principles import dynamics as _first_principles_dynamics

# from drone_models.models.first_principles import (
#     dynamics_sybolic as _first_principles_dynamics_symbolic,
# )
from drone_models.models.identified.so_rpy import dynamics as _so_rpy_dynamics

# from drone_models.models.identified.so_rpy import dynamics_symbolic as _so_rpy_dynamics_symbolic
from drone_models.models.identified.so_rpy_rotor import dynamics as _so_rpy_rotor_dynamics

# from drone_models.models.identified.so_rpy_rotor import (
#     dynamics_symbolic as _so_rpy_rotor_dynamics_symbolic,
# )
from drone_models.models.identified.so_rpy_rotor_drag import dynamics as _so_rpy_rotor_drag_dynamics

# from drone_models.models.identified.so_rpy_rotor_drag import (
#     dynamics_symbolic as _so_rpy_rotor_drag_dynamics_symbolic,
# )

available_models: dict[str, Callable] = {
    "first_principles": _first_principles_dynamics,
    "so_rpy": _so_rpy_dynamics,
    "so_rpy_rotor": _so_rpy_rotor_dynamics,
    "so_rpy_rotor_drag": _so_rpy_rotor_drag_dynamics,
    # "first_principles_symbolic": _first_principles_dynamics_symbolic,
    # "so_rpy_symbolic": _so_rpy_dynamics_symbolic,
    # "so_rpy_rotor_symbolic": _so_rpy_rotor_dynamics_symbolic,
    # "so_rpy_rotor_drag_symbolic": _so_rpy_rotor_drag_dynamics_symbolic,
}


def model_features(model: Callable) -> dict[str, bool]:
    """Get the features of a model."""
    return getattr(model, "__drone_model_features__")
