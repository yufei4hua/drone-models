"""TODO."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace
from scipy.spatial.transform import Rotation as R

import drone_models.models.symbols as symbols
from drone_models.models.utils import supports
from drone_models.transform import motor_force2rotor_speed
from drone_models.utils import rotation

if TYPE_CHECKING:
    from array_api_typing import Array

    from drone_models.utils.constants import Constants


@supports(rotor_dynamics=True)
def dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    constants: Constants,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        command: Roll pitch yaw (rad) and collective thrust (N) command.
        constants: Containing the constants of the drone.
        rotor_vel: Speed of the 4 motors (rad/s). If None, the commanded thrust is directly
            applied (not recommended). If value is given, rotor dynamics are calculated.
        dist_f: Disturbance force (N) acting on the CoM.
        dist_t: Disturbance torque (Nm) acting on the CoM.

    Returns:
        The derivatives of all state variables.
    """
    xp = array_namespace(pos)
    cmd_f = command[..., -1]
    cmd_rotor_vel = motor_force2rotor_speed(cmd_f / 4, constants.KF)
    cmd_rpy = command[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")
    rpy_rates = rotation.ang_vel2rpy_rates(quat, ang_vel)

    if rotor_vel is None:
        rotor_vel_dot = None
        rotor_vel = cmd_rotor_vel
        warnings.warn("Rotor velocity is not provided, using commanded rotor velocity directly.")
    else:
        rotor_vel_dot = (
            1 / constants.DI_DD_ACC[2] * (cmd_rotor_vel[..., None] - rotor_vel)
            - constants.KM * rotor_vel**2
        )
    forces_motor = xp.sum(constants.KF * rotor_vel**2, axis=-1)
    forces_sum = xp.sum(forces_motor, axis=-1)
    thrust = constants.DI_DD_ACC[0] + constants.DI_DD_ACC[1] * forces_sum

    drone_z_axis = rot.inv().as_matrix()[..., -1, :]

    pos_dot = vel
    vel_dot = (
        1 / constants.MASS * thrust[..., None] * drone_z_axis
        + constants.GRAVITY_VEC
        + 1 / constants.MASS * constants.DI_DD_ACC[2] * vel
        + 1 / constants.MASS * constants.DI_DD_ACC[3] * vel * xp.abs(vel)
    )
    if dist_f is not None:
        vel_dot = vel_dot + dist_f / constants.MASS

    # Rotational equation of motion
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    rpy_rates_dot = (
        constants.DI_DD_PARAMS[:, 0] * euler_angles
        + constants.DI_DD_PARAMS[:, 1] * rpy_rates
        + constants.DI_DD_PARAMS[:, 2] * cmd_rpy
    )
    ang_vel_dot = rotation.rpy_rates2ang_vel(quat, rpy_rates_dot)
    if dist_t is not None:
        # adding disturbances to the state
        # adding torque is a little more complex:
        # angular acceleration can be converted to torque
        torque = xp.matvec(constants.J, ang_vel_dot) - xp.linalg.cross(
            ang_vel, xp.matvec(constants.J, ang_vel)
        )
        # adding torque
        torque = torque + dist_t
        # back to angular acceleration
        ang_vel_dot = xp.matvec(constants.J_INV, torque)

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def dynamics_symbolic(
    constants: Constants,
    calc_rotor_vel: bool = False,
    calc_dist_f: bool = False,
    calc_dist_t: bool = False,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    TODO.
    """
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if calc_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    if calc_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if calc_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = cs.vertcat(symbols.cmd_roll, symbols.cmd_pitch, symbols.cmd_yaw, symbols.cmd_thrust)

    # Defining the dynamics function
    if calc_rotor_vel:
        # Note: Due to the structure of the integrator, we split the commanded thrust into
        # four equal parts and later apply the sum as total thrust again. Those four forces
        # are not the true forces of the motors, but the sum is the true total thrust.
        forces_motor_dot = 1 / constants.DI_D_ACC[2] * (symbols.cmd_thrust / 4 - symbols.rotor_vel)
        thrust = cs.sum1(symbols.rotor_vel)
        # Creating force vector
        forces_motor_vec = cs.vertcat(0, 0, constants.DI_D_ACC[0] + constants.DI_D_ACC[1] * thrust)
    else:
        thrust = symbols.cmd_thrust
        # Creating force vector
        forces_motor_vec = cs.vertcat(0, 0, constants.DI_ACC[0] + constants.DI_ACC[1] * thrust)

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = symbols.rot @ forces_motor_vec / constants.MASS + constants.GRAVITY_VEC
    if calc_dist_f:
        # Adding force disturbances to the state
        vel_dot = vel_dot + symbols.dist_f / constants.MASS

    # Rotational equation of motion
    euler_angles = rotation.cs_quat2euler(symbols.quat)

    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    rpy_rates = rotation.cs_ang_vel2rpy_rates(symbols.quat, symbols.ang_vel)
    if calc_rotor_vel:
        rpy_rates_dot = (
            constants.DI_D_PARAMS[:, 0] * euler_angles
            + constants.DI_D_PARAMS[:, 1] * rpy_rates
            + constants.DI_D_PARAMS[:, 2]
            * cs.vertcat(symbols.cmd_roll, symbols.cmd_pitch, symbols.cmd_yaw)
        )
    else:
        rpy_rates_dot = (
            constants.DI_PARAMS[:, 0] * euler_angles
            + constants.DI_PARAMS[:, 1] * rpy_rates
            + constants.DI_PARAMS[:, 2]
            * cs.vertcat(symbols.cmd_roll, symbols.cmd_pitch, symbols.cmd_yaw)
        )
    ang_vel_dot = rotation.cs_rpy_rates_deriv2ang_vel_deriv(symbols.quat, rpy_rates, rpy_rates_dot)
    if calc_dist_t:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = constants.J @ ang_vel_dot - cs.cross(
            symbols.ang_vel, constants.J @ symbols.ang_vel
        )
        # adding torque
        torque = torque + symbols.dist_t
        # back to angular acceleration
        ang_vel_dot = constants.J_INV @ torque

    if calc_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y
