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


def poly(x, p, order):
    y = 0
    for i in range(order + 1):
        y += p[i] * x**i
    return y


def inversepoly(y, param, order):
    # index of param = order, i.e. y = sum ( p[i] * x**i ) for all i
    assert len(param) == order + 1
    if order == 1:
        return (y - param[0]) / param[1]
    elif order == 2:
        return (-param[1] + np.sqrt(param[1] ** 2 - 4 * param[2] * (param[0] - y))) / (2 * param[2])
    elif order == 3:
        # https://math.vanderbilt.edu/schectex/courses/cubic/
        # a = p[3], b = p[2], c = p[1], d = p[0]-thrust
        p = -param[2] / (3 * param[3])
        q = p**3 + (param[2] * param[1] - 3 * param[3] * (param[0] - y)) / (6 * param[3] ** 2)
        r = param[1] / (3 * param[3])

        qrp = np.sqrt(q**2 + (r - p**2) ** 3)
        return np.cbrt(q + qrp) + np.cbrt(q - qrp) + p
    else:
        raise NotImplementedError(f"Inverted polynomial of order {order} not supported.")


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


# def thrust2rpm(thrust, perMotor = False, constants: Constants):
#     """Thrust [N] = KF * RPM^2"""
#     if not perMotor:
#         thrust /= 4
#     return np.sqrt(thrust/constants.KF)
