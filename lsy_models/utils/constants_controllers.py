"""Constants for the numeric controllers."""

import numpy as np

# Constants for the controllers
# Same as in the firmware. Do not touch
# Not part of the Constants class, since the values are controller specific and not drone specific

#### Mellinger controller (see controller_mellinger.c)
# Note: The firmware assumes mass=0.027. With battery thats closer to 0.034 though!!!
mass = 0.027  # TODO This is the wrong mass (cf with battery weighs more!)
massThrust = 132000

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
kR_xy = 70000  # P
kw_xy = 20000  # D
ki_m_xy = 0.0  # I
i_range_m_xy = 1.0

# Yaw
kR_z = 60000  # P
kw_z = 12000  # D
ki_m_z = 500  # I
i_range_m_z = 1500

# roll and pitch angular velocity
kd_omega_rp = 200  # D

cntrl_const_mel = {
    "mass": mass,
    "massThrust": massThrust,
    "kp": np.array([kp_xy, kp_xy, kp_z]),
    "kd": np.array([kd_xy, kd_xy, kd_z]),
    "ki": np.array([ki_xy, ki_xy, ki_z]),
    "i_range": np.array([i_range_xy, i_range_xy, i_range_z]),
    "kR": np.array([kR_xy, kR_xy, kR_z]),
    "kw": np.array([kw_xy, kw_xy, kw_z]),
    "ki_m": np.array([ki_m_xy, ki_m_xy, ki_m_z]),
    "kd_omega": np.array([kd_omega_rp, kd_omega_rp, 0.0]),
    "i_range_m": np.array([i_range_m_xy, i_range_m_xy, i_range_m_z]),
}
