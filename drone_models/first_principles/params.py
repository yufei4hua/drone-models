"""First principles model parameters."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from array_api_typing import Array


class FirstPrinciplesParams(NamedTuple):
    """Parameters for the FirstPrinciples model."""

    thrust_tau: float
    KF: float
    KM: float
    L: float
    mixing_matrix: Array
    gravity_vec: Array
    mass: float
    J: Array
    J_inv: Array

    @staticmethod
    def load(drone_model: str) -> FirstPrinciplesParams:
        """Load the parameters for the drone model from the params.toml file."""
        with open(Path(__file__).parent / "params.toml", "rb") as f:
            params = tomllib.load(f)
        if drone_model not in params:
            raise KeyError(f"Drone model `{drone_model}` not found in params.toml")
        params = {k: np.asarray(v) for k, v in params[drone_model].items()}
        params["J_inv"] = np.linalg.inv(params["J"])
        return FirstPrinciplesParams(**params)
