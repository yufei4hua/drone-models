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

from array_api_compat import array_namespace

import drone_models.utils.cf2 as cf2
import drone_models.utils.rotation as R
from drone_models.utils.constants_controllers import cntrl_const_mel

if TYPE_CHECKING:
    from array_api_compat.typing import Array

    from drone_models.utils.constants import Constants


def cntrl_mellinger_position(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command_state: Array,
    constants: Constants,
    dt: float = 1 / 500,
    i_error: Array | None = None,
) -> tuple[Array, Array]:
    """The positional part of the mellinger controller.

    Args:
        pos (Array): State of the drone (position), can be batched
        quat (Array): State of the drone (quaternion), can be batched
        vel (Array): State of the drone (velocity), can be batched
        ang_vel (Array): State of the drone (angular velocity) in rad/s, can be batched
        command_state (Array): Full commanded state in SI units (or rad) in the form
            [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
        constants (Constants): Constants of the specific drone
        dt (float, optional): Time since last call. Defaults to 1/500.
        i_error (Array | None, optional): Integral error. Defaults to None.

    Returns:
        tuple[Array, Array]: command_RPYT [rad, rad, rad, N], i_error
    """
    xp = pos.__array_namespace__()

    setpointPos = command_state[..., 0:3]
    setpointVel = command_state[..., 3:6]
    setpointAcc = command_state[..., 6:9]
    setpointYaw = command_state[..., 9]
    # setpointRPY_rates = command_state[..., 10:13]

    # From firmware controller_mellinger

    # l. 145 Position Error (ep)
    r_error = setpointPos - pos

    # l. 148 Velocity Error (ev)
    v_error = setpointVel - vel

    # l.151 ff Integral Error
    if i_error is None:
        i_error = xp.zeros_like(pos)
    i_error = xp.clip(
        i_error + r_error * dt, -cntrl_const_mel["i_range"], cntrl_const_mel["i_range"]
    )

    # l. 161 Desired thrust [F_des]
    # => only one case here, since setpoint is always in absolute mode
    # Note: since we've defined the gravity in z direction, a "-" needs to be added
    target_thrust = (
        cntrl_const_mel["mass"] * (setpointAcc - constants.GRAVITY_VEC)
        + cntrl_const_mel["kp"] * r_error
        + cntrl_const_mel["kd"] * v_error
        + cntrl_const_mel["ki"] * i_error
    )

    # l. 178 Rate-controlled YAW is moving YAW angle setpoint
    # => only one case here, since the setpoint is always in absolute mode
    desiredYaw = setpointYaw

    # l. 189 Z-Axis [zB]
    rot = R.from_quat(quat).as_matrix()
    z_axis = rot[..., -1]  # 3rd column or roation matrix is z axis

    # l. 194 yaw correction (only if position control is not used)
    # => skipped since we always use position control here

    # l. 204 Current thrust [F]
    # Taking the dot product of the last axis:
    # current_thrust = xp.dot(target_thrust, z_axis)  # doesnt work because of dimensions
    current_thrust = xp.einsum("...i,...i->...", target_thrust, z_axis)

    # l. 207 Calculate axis [zB_des]
    z_axis_desired = target_thrust / xp.linalg.norm(target_thrust)

    # l. 210 [xC_des]
    # x_axis_desired = z_axis_desired x [sin(yaw), cos(yaw), 0]^T
    x_c_des_x = xp.cos(desiredYaw)
    x_c_des_y = xp.sin(desiredYaw)
    x_c_des_z = x_c_des_y * 0  # to get zeros in the correct shape
    x_c_des = xp.stack((x_c_des_x, x_c_des_y, x_c_des_z), axis=-1)
    # [yB_des]
    y_axis_desired = xp.linalg.cross(z_axis_desired, x_c_des)
    y_axis_desired = y_axis_desired / xp.linalg.norm(y_axis_desired)
    # [xB_des]
    x_axis_desired = xp.linalg.cross(y_axis_desired, z_axis_desired)

    # converting desired axis to rotation matrix and then to RPY
    matrix = xp.stack((x_axis_desired, y_axis_desired, z_axis_desired), axis=-1)
    command_RPY = R.from_matrix(matrix).as_euler("xyz", degrees=False)

    # l. 283
    thrust = cntrl_const_mel["massThrust"] * current_thrust

    # Transform thrust into N to keep uniform interface
    thrust = cf2.pwm2force(thrust, constants, perMotor=False)

    command_RPYT = xp.concat((command_RPY, thrust[..., None]), axis=-1)

    return command_RPYT, i_error


def cntrl_mellinger_attitude(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command_RPYT: Array,
    constants: Constants,
    dt: float = 1 / 500,
    i_error_m: Array | None = None,
    angular_vel_des: Array | None = None,
    prev_angular_vel: Array | None = None,
    prev_angular_vel_des: Array | None = None,
) -> tuple[Array, Array]:
    """_summary.

    Args:
        pos (Array): State of the drone (position), can be batched
        quat (Array): State of the drone (quaternion), can be batched
        vel (Array): State of the drone (velocity), can be batched
        ang_vel (Array): State of the drone (angular velocity) in rad/s, can be batched
        command_RPYT (Array): Commanded attitude (roll, pitch, yaw) and total thrust [rad, rad, rad, N]
        constants (Constants): Constants of the specific drone
        dt (float, optional): Time since last call. Defaults to 1/500.
        i_error_m (Array | None, optional): Integral error. Defaults to None.
        angular_vel_des (Array | None, optional): Desired angular velocity in rad/s. Defaults to None.
        prev_angular_vel (Array | None, optional): Previous angular velocity in rad/s. Defaults to None.
        prev_angular_vel_des (Array | None, optional): Previous angular velocity command in rad/s. Defaults to None.

    Returns:
        tuple[Array, Array]: 4 Motor forces [N], i_error_m
    """
    xp = pos.__array_namespace__()
    thrust_des = cf2.force2pwm(command_RPYT[..., -1], constants, perMotor=False)
    rpy_des = command_RPYT[..., :-1]
    axis_flip = xp.array([1, -1, 1])  # to change the direction of the y axis
    # From firmware controller_mellinger
    # l. 220 ff [eR]
    # Using the "inefficient" code from the firmware
    rot = R.from_quat(quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=False)
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

    ew = angular_vel_des - ang_vel  # if the setpoint is ever != 0 => change sign of setpoint[1]
    ew = axis_flip * ew  # Sign change to account for crazyflie axis

    err_d = (
        (angular_vel_des - prev_angular_vel_des) - (ang_vel - prev_angular_vel)
    ) / dt  # WARNING: if the setpoint is ever != 0 => change sign of ew.y!
    err_d = axis_flip * err_d  # Sign change to account for crazyflie axis

    # l. 268 ff Integral Error
    if i_error_m is None:
        i_error_m = xp.zeros_like(pos)
    i_error_m = i_error_m - eR * dt
    i_error_m = xp.clip(i_error_m, -cntrl_const_mel["i_range_m"], cntrl_const_mel["i_range_m"])

    # l. 278 ff Moment:
    M = (
        -cntrl_const_mel["kR"] * eR
        + cntrl_const_mel["kw"] * ew
        + cntrl_const_mel["ki_m"] * i_error_m
        + cntrl_const_mel["kd_omega"] * err_d
    )

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


def fw_power_distribution_legacy(roll: Array, pitch: Array, yaw: Array, thrust: Array) -> Array:
    """Legacy power distribution from power_distribution_quadrotor.c working with PWM signals.

    Returns:
        Array of the four commanded motor PWMs
    """
    xp = array_namespace(roll)
    m1_pwm = thrust - roll + pitch + yaw
    m2_pwm = thrust - roll - pitch - yaw
    m3_pwm = thrust + roll - pitch + yaw
    m4_pwm = thrust + roll + pitch - yaw
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
