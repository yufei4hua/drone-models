"""This file contains modelled controllers, which can be combined with lower level modesl.

The structure of the low control on the Crazyflie drones is as follows:
    1. Controller (can be cascaded internally): Gets a setpoint type (i.e. RPYT setpoint)
       and returns control type (RPYT command in PWM or in N and Nm)
    2. Power distribution: Distributes the RPYT command to the four motors,
       such that the commmand is executed and total commanded thrust is achieved
    3. Battery compensation: Adjusts the actual PWM to the motors, such that the commanded forces are achieved.
       Not modelled here, since we assume it's working perfectly
    4. Thrust capping: The commanded motor PWM values are capped at PWM_MAX (65535).

All the functions beginning with cntrl are controllers of the crazyflie.
All the functions beginning with fw are steps done in the firmware after the controllers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import lsy_models.utils.cf2 as cf2
import lsy_models.utils.rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor

    from lsy_models.utils.constants import Constants


def cntrl_mellinger_attitude(
    pos: Array, quat: Array, vel: Array, angvel: Array, command_RPYT: Array
) -> Array:
    """Simulates the attitude controller of the Mellinger controller.

    TODO: Think about if we want this to be batchable or not (what makes mor sense in terms of UKF)

    Args:
        command_RPYT: Array of shape (4,) or (N,4), containing commanded values for roll, pitch, yaw (rpy) in degrees and thrust in PWM scaling

    Return:
        Control type as in firmware TODO
    """
    xp = pos.__array_namespace__()
    # From firmware controller_mellinger
    # WARNING: This should be set while calling. For now it is set to the same value as in the firmware
    # Long term we should set it to 200Hz, since that's what the UKF is running at
    dt = 1.0 / 500
    # l. 52
    mass_thrust = 132000
    # l. 66-79
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

    ### Calculate RPMs for first x (mean of UKF estimate) & u to keep RPM command constant
    thrust_des = command_RPYT[..., -1]
    rpy_des = command_RPYT[..., :-1]

    ### From firmware controller_mellinger:
    # l. 220 ff
    rot = R.from_quat(quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=True)
    R_act = rot.as_matrix()
    R_des = rot_des.as_matrix()
    # print(f"R_est={R_act}, R_des={R_des}, thrust_des={thrust_des}")
    print(f"rpy_est={rot.as_euler('xyz', degrees=True)}, rpy_des={rpy_des}")
    eR = 0.5 * (R_des.T @ R_act - R_act.T @ R_des)
    # vee operator (SO3 to R3), the -y is to account for the frame of the crazyflie
    eR = xp.array([eR[2, 1], -eR[0, 2], eR[1, 0]])

    # l.256 ff
    angular_vel_des = xp.zeros_like(pos)  # zero for now (would need to be given as input)
    ew = angular_vel_des - angvel  # if the setpoint is ever != 0 => change sign of setpoint[1]

    # l. 259 ff
    prev_angular_vel_des = xp.zeros_like(pos)  # zero for now (would need to be stored)
    prev_angular_vel = xp.zeros_like(pos)  # zero for now (would need to be stored)
    err_d = (
        (angular_vel_des - prev_angular_vel_des) - (angvel - prev_angular_vel)
    ) / dt  # WARNING: if the setpoint is ever != 0 => change sign of ew.y!
    prev_angular_vel = angvel.copy()

    # l. 268 ff
    i_error_m = xp.zeros_like(pos)  # zero for now (would need to be stored)
    # i_error_m -= eR * dt
    # i_error_m[0:2] = xp.clip(i_error_m[0:2], -i_range_m_xy, i_range_m_xy)
    # i_error_m[2] = xp.clip(i_error_m[2], -i_range_m_z, i_range_m_z)

    # l. 279 ff
    print(f"eR={eR}, ew={ew}")
    Mx = -kR_xy * eR[0] + kw_xy * ew[0] + ki_m_xy * i_error_m[0] + kd_omega_rp * err_d[0]
    My = -kR_xy * eR[1] + kw_xy * ew[1] + ki_m_xy * i_error_m[1] + kd_omega_rp * err_d[1]
    Mz = -kR_z * eR[2] + kw_z * ew[2] + ki_m_z * i_error_m[2]

    # l. 297 ff
    if thrust_des > 0:
        cmd_roll = xp.clip(Mx, -32000, 32000)
        cmd_pitch = xp.clip(My, -32000, 32000)
        cmd_yaw = xp.clip(-Mz, -32000, 32000)
    else:
        cmd_roll = 0
        cmd_pitch = 0
        cmd_yaw = 0

    return {"thrust": thrust_des, "roll": cmd_roll, "pitch": cmd_pitch, "yaw": cmd_yaw}


def fw_power_distribution_flapper():
    """This exists in the firmware for some reason."""
    raise NotImplementedError("Don't know yet what flapper is for")


def fw_power_distribution_legacy(control: dict[str, Array]) -> Array:
    """Legacy power distribution from power_distribution_quadrotor.c working with PWM signals.

    Args:
        control: dictionary of the same form as the control type in the firmware

    Returns:
        Array of the four commanded motor PWMs
    """
    xp = control["thrust"].__array_namespace__()
    roll = control["roll"]
    pitch = control["pitch"]
    m1_pwm = control["thrust"] - roll + pitch + control["yaw"]
    m2_pwm = control["thrust"] - roll - pitch - control["yaw"]
    m3_pwm = control["thrust"] + roll - pitch + control["yaw"]
    m4_pwm = control["thrust"] + roll + pitch - control["yaw"]
    return xp.array([m1_pwm, m2_pwm, m3_pwm, m4_pwm])


def fw_power_distribution_force_torque(control: dict[str, Array], constants: Constants) -> Array:
    """Power distribution from power_distribution_quadrotor.c working with SI units.

    Args:
        control: dictionary of the same form as the control type in the firmware
        constants: constants for the drone

    Returns:
        Array of the four commanded motor PWMs
    """
    xp = control["thrustSI"].__array_namespace__()
    # WARNING: This function only works for flying in "x" configuration
    roll_part = 0.25 / constants.L * control["torqueX"]
    pitch_part = 0.25 / constants.L * control["torqueY"]
    yaw_part = 0.25 * control["torqueZ"] * constants.KF / constants.KM
    thrust_part = 0.25 * control["thrustSI"]

    m1_force = thrust_part - roll_part - pitch_part - yaw_part
    m2_force = thrust_part - roll_part + pitch_part + yaw_part
    m3_force = thrust_part + roll_part + pitch_part - yaw_part
    m4_force = thrust_part + roll_part - pitch_part + yaw_part

    # TODO force 2 pwm
    motor_forces = xp.array([m1_force, m2_force, m3_force, m4_force])
    return cf2.force2pwm(motor_forces, constants=constants, perMotor=True)


def fw_power_distribution_cap(motor_pwm: Array, constants: Constants) -> Constants:
    """TODO."""
    xp = motor_pwm.__array_namespace__()
    # TODO, if 0, let it be 0
    if xp.all(motor_pwm == 0):
        return xp.zeros_like(motor_pwm)
    return xp.clip(motor_pwm, constants.PWM_MIN, constants.PWM_MAX)
