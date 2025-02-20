"""This file contains helper functions for the Crazyflie 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    from lsy_models.utils.constants import Constants

    Array = NDArray | JaxArray | Tensor


def force2pwm(thrust: Array | float, constants: Constants, perMotor: bool = False) -> Array | float:
    """Convert thrust in N to thrust in PWM.

    Args:
        thrust: Array or float of the thrust in N
        constants: Of the drone
        perMotor: Optional, boolean if the calculation is executed per motor, or for total thrust

    Returns:
        thrust: Array or float thrust in PWM
    """
    if not perMotor:
        thrust /= 4
    ratio = thrust / constants.THRUST_MAX
    return ratio * constants.PWM_MAX


def pwm2force(pwm: Array | float, constants: Constants, perMotor: bool = False) -> Array | float:
    """Convert pwm thrust command to actual thrust.

    Args:
        pwm: Array or float of the pwm value
        constants: Of the drone
        perMotor: Optional, boolean if the calculation is executed per motor, or for total thrust

    Returns:
        thrust: Array or float thrust in [N]
    """
    ratio = pwm / constants.PWM_MAX
    if not perMotor:
        ratio *= 4
    return ratio * constants.THRUST_MAX
