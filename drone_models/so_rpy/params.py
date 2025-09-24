"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from drone_models.core import load_params

if TYPE_CHECKING:
    from array_api_typing import Array


class SoRpyParams(NamedTuple):
    """Parameters for the SoRpy model."""

    mass: float
    gravity_vec: Array
    J: Array
    J_inv: Array
    acc_coef: Array
    cmd_f_coef: Array
    rpy_coef: Array
    rpy_rates_coef: Array
    cmd_rpy_coef: Array

    @staticmethod
    def load(drone_model: str) -> SoRpyParams:
        """Load the parameters for the drone model from the params.toml file."""
        params = load_params("so_rpy", drone_model)
        fields = SoRpyParams.__annotations__.keys()
        params = {k: np.asarray(v) for k, v in params.items() if k in fields}
        return SoRpyParams(**params)
