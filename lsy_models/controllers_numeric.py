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

import lsy_models.utils.cf2 as cf2
import lsy_models.utils.rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor

    from lsy_models.utils.constants import Constants


def cntrl_mellinger_position(
    pos: Array,
    quat: Array,
    vel: Array,
    angvel: Array,
    command_PQVW: Array,
    constants: Constants,
    dt: float = 1 / 500,
    i_error: Array | None = None,
) -> Array:
    """The positional part of the mellinger controller.

    Returns a RPYT command
    """
    xp = pos.__array_namespace__()

    setpointPos = command_PQVW[..., 0:3]
    setpointVel = command_PQVW[..., 3:6]
    setpointAcc = command_PQVW[..., 6:9]
    setpointQuat = command_PQVW[..., 9:12]

    # From firmware controller_mellinger
    mass = 0.033  # TODO (CF_MASS,)
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

    # Vectorization:
    kp = xp.array([kp_xy, kp_xy, kp_z])
    kd = xp.array([kd_xy, kd_xy, kd_z])
    ki = xp.array([ki_xy, ki_xy, ki_z])
    i_range = xp.array([i_range_xy, i_range_xy, i_range_z])

    # l. 145 Position Error (ep)
    r_error = setpointPos - pos

    # l. 148 Velocity Error (ev)
    v_error = setpointVel - vel

    # l.151 ff Integral Error
    if i_error is None:
        i_error = xp.zeros_like(pos)
    i_error = xp.clip(i_error + r_error * dt, -i_range, i_range)

    # l. 161 Desired thrust [F_des] TODO correct mode?
    target_thrust = (
        mass * (setpointAcc + constants.GRAVITY_VEC) + kp * r_error + kd * v_error + ki * i_error
    )

    # l. 178 Rate-controlled YAW is moving YAW angle setpoint (skipped)
    desiredYaw = 0  # TODO

    # l. 189 Z-Axis [zB]
    rot = R.from_quat(quat)
    z_axis = rot.as_matrix()[..., :, -1]  # 3rd column or roation matrix is z axis

    # l. 194 yaw correction (skipped)
    # TODO

    # l. 204 Current thrust [F]
    current_thrust = xp.dot(target_thrust, z_axis)

    # l. 207 Calculate axis [zB_des]
    z_axis_desired = target_thrust / xp.linalg.norm(target_thrust)

    # l. 210 [xC_des]
    # x_axis_desired = z_axis_desired x [sin(yaw), cos(yaw), 0]^T
    x_c_des_x = xp.cos(desiredYaw)
    x_c_des_y = xp.sin(desiredYaw)
    x_c_des_z = 0
    x_c_des = xp.stack((x_c_des_x, x_c_des_y, x_c_des_z), axis=-1)
    # [yB_des]
    y_axis_desired = xp.cross(z_axis_desired, x_c_des)
    y_axis_desired = y_axis_desired / xp.linalg.norm(y_axis_desired)
    # [xB_des]
    x_axis_desired = xp.cross(y_axis_desired, z_axis_desired)

    # converting desired axis to rotation matrix and then to RPY
    matrix = xp.stack((x_axis_desired, y_axis_desired, z_axis_desired), axis=-1)
    command_RPY = R.from_matrix(matrix).as_euler("xyz", degrees=True)

    # l. 283
    thrust = massThrust * current_thrust

    command_RPYT = xp.stack((command_RPY, thrust), axis=-1)

    return command_RPYT


def cntrl_mellinger_attitude(
    pos: Array,
    quat: Array,
    vel: Array,
    angvel: Array,
    command_RPYT: Array,
    dt: float = 1 / 500,
    i_error_m: Array | None = None,
    angular_vel_des: Array | None = None,
    prev_angular_vel: Array | None = None,
    prev_angular_vel_des: Array | None = None,
) -> Array:
    """Simulates the attitude controller of the Mellinger controller.

    TODO: Think about if we want this to be batchable or not (what makes mor sense in terms of UKF)

    Args:
        command_RPYT: Array of shape (4,) or (N,4), containing commanded values for roll, pitch, yaw (rpy) in degrees and thrust in PWM scaling

    Return:
        Control type as in firmware TODO
    """
    xp = pos.__array_namespace__()
    thrust_des = command_RPYT[..., -1]
    rpy_des = command_RPYT[..., :-1]
    axis_flip = xp.array([1, -1, 1])
    # From firmware controller_mellinger
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

    # Vectorization
    kR = xp.array([kR_xy, kR_xy, kR_z])
    kw = xp.array([kw_xy, kw_xy, kw_z])
    ki_m = xp.array([ki_m_xy, ki_m_xy, ki_m_z])
    kd_omega = xp.array([kd_omega_rp, kd_omega_rp, 0.0])
    i_range_m = xp.array([i_range_m_xy, i_range_m_xy, i_range_m_z])

    # l. 220 ff [eR]
    # Using the "inefficient" code from the firmware
    rot = R.from_quat(quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=True)
    R_act = rot.as_matrix()
    R_des = rot_des.as_matrix()
    eRM = xp.matmul(xp.swapaxes(R_des, -1, -2), R_act) - xp.matmul(
        xp.swapaxes(R_act, -1, -2), R_des
    )
    # vee operator (SO3 to R3)
    eR = xp.stack((eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1)
    eR = axis_flip * eR  # Sign change to account for crazyflie axis

    # l.248 ff [ew]
    if angular_vel_des is None:
        angular_vel_des = xp.zeros_like(pos)
    if prev_angular_vel_des is None:
        prev_angular_vel_des = xp.zeros_like(pos)
    if prev_angular_vel is None:
        prev_angular_vel = xp.zeros_like(pos)

    ew = angular_vel_des - angvel  # if the setpoint is ever != 0 => change sign of setpoint[1]
    ew = axis_flip * ew  # Sign change to account for crazyflie axis

    err_d = (
        (angular_vel_des - prev_angular_vel_des) - (angvel - prev_angular_vel)
    ) / dt  # WARNING: if the setpoint is ever != 0 => change sign of ew.y!
    err_d = axis_flip * err_d  # Sign change to account for crazyflie axis

    # l. 268 ff Integral Error
    if i_error_m is None:
        i_error_m = xp.zeros_like(pos)
    i_error_m = i_error_m - eR * dt
    i_error_m = xp.clip(i_error_m, -i_range_m, i_error_m)

    # l. 278 ff Moment:
    # print(f"eR={eR}, ew={ew}")
    M = -kR * eR + kw * ew + ki_m * i_error_m + kd_omega * err_d * 0

    # l. 297 ff
    M = xp.where((thrust_des > 0)[..., None], xp.clip(M, -32000, 32000), M * 0)

    control = {"thrust": thrust_des, "roll": M[..., 0], "pitch": M[..., 1], "yaw": -M[..., 2]}

    # INFO:
    # The following part is NOT part of the Mellinger controller itself,
    # but of the firmware. It is how the firmware calculates the motor forces
    # onboard. Actually, the output of this part are motor PWM values.
    # However, with the knowledge of the system, we can directly calculate the
    # corresponding force "commands", which need to pass through the motor dynamics.
    pwms = fw_power_distribution_legacy(control)
    pwms = fw_power_distribution_cap(pwms, constants)
    forces = cf2.pwm2force(pwms, constants, perMotor=True)

    return forces, i_error_m


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
    return xp.stack((m1_pwm, m2_pwm, m3_pwm, m4_pwm), axis=-1)


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
    motor_forces = xp.stack((m1_force, m2_force, m3_force, m4_force), axis=-1)
    return cf2.force2pwm(motor_forces, constants=constants, perMotor=True)


def fw_power_distribution_cap(motor_pwm: Array, constants: Constants) -> Constants:
    """TODO."""
    xp = motor_pwm.__array_namespace__()
    return xp.where(
        xp.all(motor_pwm == 0),
        xp.zeros_like(motor_pwm),
        xp.clip(motor_pwm, constants.PWM_MIN, constants.PWM_MAX),
    )
