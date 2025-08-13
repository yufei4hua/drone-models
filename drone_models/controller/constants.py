"""Constants for the controllers."""

from types import ModuleType

import numpy as np
from array_api_typing import Array

# Constants for the controllers
# Same as in the firmware. Do not touch
# Not part of the Constants class, since the values are controller specific and not drone specific

#### Mellinger controller (see controller_mellinger.c)
# Note: The firmware assumes mass=0.027. With battery thats closer to 0.034 though!!!
mass = 0.034  # TODO This is the wrong mass (cf with battery weighs more!)
massThrust = 132000 * 0.034 / 0.027

# XY Position PID
kp_xy = 0.4  # P
kd_xy = 0.2  # D
ki_xy = 0.05  # I
i_range_xy = 2.0

# Z Position
kp_z = 1.25  # P
kd_z = 0.4  # D
ki_z = 0.05  # I
i_range_z = 0.4

# Attitude
kR_xy = 70000.0  # P
kw_xy = 20000.0  # D
ki_m_xy = 0.0  # I
i_range_m_xy = 1.0

# Yaw
kR_z = 60000.0  # P
kw_z = 12000.0  # D
ki_m_z = 500.0  # I
i_range_m_z = 1500.0

# roll and pitch angular velocity
kd_omega_rp = 200.0  # D


def cntrl_const_mel(xp: ModuleType = np) -> dict[str, Array | float]:
    """Returns the controller constants for the Mellinger controller."""
    return {
        "mass": mass,
        "massThrust": massThrust,
        "kp": xp.asarray([kp_xy, kp_xy, kp_z]),
        "kd": xp.asarray([kd_xy, kd_xy, kd_z]),
        "ki": xp.asarray([ki_xy, ki_xy, ki_z]),
        "i_range": xp.asarray([i_range_xy, i_range_xy, i_range_z]),
        "kR": xp.asarray([kR_xy, kR_xy, kR_z]),
        "kw": xp.asarray([kw_xy, kw_xy, kw_z]),
        "ki_m": xp.asarray([ki_m_xy, ki_m_xy, ki_m_z]),
        "kd_omega": xp.asarray([kd_omega_rp, kd_omega_rp, 0.0]),
        "i_range_m": xp.asarray([i_range_m_xy, i_range_m_xy, i_range_m_z]),
        "torque_pwm_range": 32_000.0,
    }
