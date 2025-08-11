"""..."""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

import drone_models.utils.cf2 as cf2
import drone_models.utils.rotation as R
from drone_models.utils.constants_controllers import cntrl_const_mel

if TYPE_CHECKING:
    from array_api_compat.typing import Array

    from drone_models.utils.constants import Constants


def pos2attitude(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    constants: Constants,
    dt: float = 1 / 500,
    i_error: Array | None = None,
) -> tuple[Array, Array]:
    """Compute the positional part of the mellinger controller.

    All controllers are implemented as pure functions. Therefore, integral errors have to be passed
    as an argument and returned as well.

    Note:
        The naming of the variables is based on the original firmware implementation, but converted
        to camel case to follow the Python style guide.

    Args:
        pos: Drone position with shape (..., 3).
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        vel: Drone velocity with shape (..., 3).
        ang_vel: Drone angular drone velocity in rad/s with shape (..., 3).
        cmd: Full state command in SI units and rad with shape (..., 13). The entries are
            [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate].
        constants: Drone specific constants.
        dt: Time since last call.
        i_error: Integral error of the position controller with shape (..., 3).

    Returns:
        tuple[Array, Array]: command_rpyt [rad, rad, rad, N], i_error
    """
    xp = array_namespace(pos)

    setpoint_pos = cmd[..., 0:3]
    setpoint_vel = cmd[..., 3:6]
    setpoint_acc = cmd[..., 6:9]
    setpoint_yaw = cmd[..., 9]
    # setpointRPY_rates = cmd[..., 10:13]
    # From firmware controller_mellinger
    r_error = setpoint_pos - pos  # l. 145 Position Error (ep)
    v_error = setpoint_vel - vel  # l. 148 Velocity Error (ev)
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
        cntrl_const_mel["mass"] * (setpoint_acc - constants.GRAVITY_VEC)
        + cntrl_const_mel["kp"] * r_error
        + cntrl_const_mel["kd"] * v_error
        + cntrl_const_mel["ki"] * i_error
    )
    # l. 178 Rate-controlled YAW is moving YAW angle setpoint
    # => only one case here, since the setpoint is always in absolute mode
    desiredYaw = setpoint_yaw
    # l. 189 Z-Axis [zB]
    rot = R.from_quat(quat).as_matrix()
    z_axis = rot[..., -1]  # 3rd column or roation matrix is z axis
    # l. 194 yaw correction (only if position control is not used)
    # => skipped since we always use position control here

    # l. 204 Current thrust [F]
    # Taking the dot product of the last axis:
    current_thrust = xp.vecdot(target_thrust, z_axis, axis=-1)
    # l. 207 Calculate axis [zB_des]
    z_axis_desired = target_thrust / xp.linalg.norm(target_thrust)
    # l. 210 [xC_des]
    # x_axis_desired = z_axis_desired x [sin(yaw), cos(yaw), 0]^T
    x_c_des_x = xp.cos(desiredYaw)
    x_c_des_y = xp.sin(desiredYaw)
    x_c_des_z = xp.zeros_like(x_c_des_x)
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
    command_rpyt = xp.concat((command_RPY, thrust[..., None]), axis=-1)
    return command_rpyt, i_error


def attitude2force_torque(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command_rpyt: Array,
    constants: Constants,
    dt: float = 1 / 500,
    i_error_m: Array | None = None,
    ang_vel_des: Array | None = None,
    prev_ang_vel: Array | None = None,
    prev_ang_vel_des: Array | None = None,
) -> tuple[Array, Array]:
    """Compute the attitude to desired force-torque part of the Mellinger controller.

    Args:
        pos: Drone position with shape (..., 3).
        quat: Drone orientation as xyzw quaternion with shape (..., 4).
        vel: Drone velocity with shape (..., 3).
        ang_vel: Drone angular drone velocity in rad/s with shape (..., 3).
        command_rpyt: Commanded attitude (roll, pitch, yaw) and total thrust [rad, rad, rad, N]
        constants (Constants): Constants of the specific drone
        dt: Time since last call.
        i_error_m: Integral error.
        ang_vel_des: Desired angular velocity in rad/s.
        prev_ang_vel: Previous angular velocity in rad/s.
        prev_ang_vel_des: Previous angular velocity command in rad/s.

    Returns:
        4 Motor forces [N], i_error_m
    """
    xp = array_namespace(pos)
    force_des = cf2.force2pwm(command_rpyt[..., -1], constants, perMotor=False)
    rpy_des = command_rpyt[..., :-1]
    axis_flip = xp.array([1, -1, 1])  # to change the direction of the y axis
    # From firmware controller_mellinger
    # l. 220 ff [eR]
    # Using the "inefficient" code from the firmware
    rot = R.from_quat(quat)
    rot_des = R.from_euler("xyz", rpy_des, degrees=False)
    R_act = rot.as_matrix()
    R_des = rot_des.as_matrix()
    # TODO: cffirmware does not multiply by 0.5 here, but the original paper does. We replicate the
    # firmware exactly to avoid sim2real issues with the original controller parameters.
    eRM = R_des.mT @ R_act - R_act.mT @ R_des
    # vee operator (SO3 to R3)
    eR = xp.stack((eRM[..., 2, 1], eRM[..., 0, 2], eRM[..., 1, 0]), axis=-1)
    eR = axis_flip * eR  # Sign change to account for crazyflie axis
    # l.248 ff [ew]
    if ang_vel_des is None:
        ang_vel_des = xp.zeros_like(pos)
    if prev_ang_vel_des is None:
        prev_ang_vel_des = xp.zeros_like(pos)
    if prev_ang_vel is None:
        prev_ang_vel = xp.zeros_like(pos)

    ew = ang_vel_des - ang_vel  # if the setpoint is ever != 0 => change sign of setpoint[1]
    ew = axis_flip * ew  # Sign change to account for crazyflie axis
    # WARNING: if the setpoint is ever != 0 => change sign of ew.y!
    err_d = ((ang_vel_des - prev_ang_vel_des) - (ang_vel - prev_ang_vel)) / dt
    err_d = axis_flip * err_d  # Sign change to account for crazyflie axis
    # l. 268 ff Integral Error
    if i_error_m is None:
        i_error_m = xp.zeros_like(pos)
    i_error_m = i_error_m - eR * dt
    i_error_m = xp.clip(i_error_m, -cntrl_const_mel["i_range_m"], cntrl_const_mel["i_range_m"])
    # l. 278 ff Moment:
    torque_des = (
        -cntrl_const_mel["kR"] * eR
        + cntrl_const_mel["kw"] * ew
        + cntrl_const_mel["ki_m"] * i_error_m
        + cntrl_const_mel["kd_omega"] * err_d
    )
    # l. 297 ff
    torque_des = xp.clip(torque_des, -cntrl_const_mel["torque_max"], cntrl_const_mel["torque_max"])
    torque_des = xp.where((force_des > 0)[..., None], torque_des, 0.0)

    control = {
        "thrust": force_des,
        "roll": torque_des[..., 0],
        "pitch": torque_des[..., 1],
        "yaw": -torque_des[..., 2],
    }
    # INFO:
    # The following part is NOT part of the Mellinger controller itself,
    # but of the firmware. It is how the firmware calculates the motor forces
    # onboard. Actually, the output of this part are motor PWM values.
    # However, with the knowledge of the system, we can directly calculate the
    # corresponding force "commands", which need to pass through the motor dynamics.
    pwms = power_distribution_legacy(control, force_des)
    pwms = power_distribution_clip(pwms, constants)
    forces = cf2.pwm2force(pwms, constants, perMotor=True)
    # TODO: Long-term, the Mellinger controller should use the new power distribution which
    # calculates motor forces in Newtons. However, for now the firmware uses the legacy power
    # distribution, so we keep it here for compatibility. To have a single consistent interface for
    # controllers within drone_models, we still want to return SI forces and torques. We thus need
    # to convert the legacy output to SI units.
    # l. 310 ff
    torque_des = (
        forces
        @ constants.SIGN_MATRIX
        * xp.stack([constants.L, constants.L, constants.KM / constants.KF])
    )
    force_des = xp.sum(forces, axis=-1)
    return force_des, torque_des, i_error_m


def force_torque2rotor_speed_legacy():
    """Convert desired collective thrust and torques to rotor speeds using legacy behavior."""
    ...


def power_distribution_legacy(torque: Array, thrust: Array) -> Array:
    """Legacy power distribution from power_distribution_quadrotor.c working with PWM signals.

    Args:
        control: dictionary of the same form as the control type in the firmware

    Returns:
        Array of the four commanded motor PWMs
    """
    xp = array_namespace(torque)
    m1_pwm = thrust - torque[..., 0] + torque[..., 1] + torque[..., 2]
    m2_pwm = thrust - torque[..., 0] - torque[..., 1] - torque[..., 2]
    m3_pwm = thrust + torque[..., 0] - torque[..., 1] + torque[..., 2]
    m4_pwm = thrust + torque[..., 0] + torque[..., 1] - torque[..., 2]
    return xp.stack((m1_pwm, m2_pwm, m3_pwm, m4_pwm), axis=-1)


def force_torque2rotor_speed(force: Array, torque: Array, constants: Constants) -> Array:
    """Convert desired collective thrust and torques to rotor speeds.

    The firmware calculates PWMs for each motor, compensates for the battery voltage, and then
    applies the modified PWMs to the motors. We assume perfect battery compensation here, skip the
    PWM interface except for clipping, and instead return rotor speeds. This allows us to account
    for rotor dynamics.

    Note:
        The equivalent function in the crazyflie firmware is power_distribution from
        power_distribution_quadrotor.c.

    Warning:
        This function assumes an X rotor configuration.

    Args:
        force: Desired thrust in SI units with shape (...,).
        torque: Desired torque in SI units with shape (..., 3).
        constants: constants for the drone

    Returns:
        Array of the four commanded motor PWMs
    """
    xp = array_namespace(torque)
    roll_part = 0.25 / constants.L * torque[..., 0]
    pitch_part = 0.25 / constants.L * torque[..., 1]
    yaw_part = 0.25 * torque[..., 2] * constants.KF / constants.KM
    thrust_part = 0.25 * force
    m1_force = thrust_part - roll_part - pitch_part - yaw_part
    m2_force = thrust_part - roll_part + pitch_part + yaw_part
    m3_force = thrust_part + roll_part + pitch_part - yaw_part
    m4_force = thrust_part + roll_part - pitch_part + yaw_part
    motor_forces = xp.stack((m1_force, m2_force, m3_force, m4_force), axis=-1)
    # Clip motor forces on the thrust instead of PWM level.
    clipped_forces = xp.clip(motor_forces, constants.THRUST_MIN, constants.THRUST_MAX)
    motor_forces = xp.where(xp.all(motor_forces == 0), 0.0, clipped_forces)
    # Assume perfect battery compensation and calculate the desired motor speeds directly
    return motor_force2rotor_speed(motor_forces, constants)


def motor_force2rotor_speed(motor_forces: Array, constants: Constants) -> Array:
    """Convert motor forces to rotor speeds.

    Args:
        motor_forces: Motor forces in SI units with shape (..., 4).
        constants: Constants of the specific drone.

    Returns:
        Array of rotor speeds in rad/s with shape (..., 4).
    """
    xp = array_namespace(motor_forces)
    return xp.sqrt(motor_forces / constants.KF)


def power_distribution_clip(motor_pwm: Array, constants: Constants) -> Constants:
    """TODO."""
    xp = array_namespace(motor_pwm)
    return xp.where(
        xp.all(motor_pwm == 0), 0.0, xp.clip(motor_pwm, constants.PWM_MIN, constants.PWM_MAX)
    )
