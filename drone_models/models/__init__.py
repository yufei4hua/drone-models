"""Models of quadrotor drones for estimation and control."""

from typing import Callable

from drone_models.models.first_principles import dynamics as _first_principles_dynamics
from drone_models.models.identified.so_rpy import dynamics as _so_rpy_dynamics
from drone_models.models.identified.so_rpy_rotor import dynamics as _so_rpy_rotor_dynamics
from drone_models.models.identified.so_rpy_rotor_drag import dynamics as _so_rpy_rotor_drag_dynamics

available_models: dict[str, Callable] = {
    "first_principles": _first_principles_dynamics,
    "so_rpy": _so_rpy_dynamics,
    "so_rpy_rotor": _so_rpy_rotor_dynamics,
    "so_rpy_rotor_drag": _so_rpy_rotor_drag_dynamics,
}


def model_features(model: Callable) -> dict[str, bool]:
    """Get the features of a model."""
    return getattr(model, "__drone_model_features__")
