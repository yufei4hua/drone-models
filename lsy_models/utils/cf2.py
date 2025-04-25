"""This file contains helper functions for the Crazyflie 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    xp = thrust.__array_namespace__()
    motor_thrust = xp.where(perMotor, thrust, thrust / 4)
    ratio = motor_thrust / constants.THRUST_MAX
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
    xp = pwm.__array_namespace__()
    ratio = pwm / constants.PWM_MAX
    ratio = xp.where(perMotor, ratio, ratio * 4)
    return ratio * constants.THRUST_MAX
