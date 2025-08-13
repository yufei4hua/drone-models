"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace
from scipy.spatial.transform import Rotation as R

import drone_models.models.symbols as symbols
from drone_models.models.utils import supports
from drone_models.utils import rotation

if TYPE_CHECKING:
    from array_api_typing import Array

    from drone_models.utils.constants import Constants


@supports(rotor_dynamics=False)
def dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
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
        cmd: Roll pitch yaw (rad) and collective thrust (N) command.
        constants: Containing the constants of the drone.
        rotor_vel: Speed of the 4 motors (rad/s). If None, the commanded thrust is directly
            applied. If a value is given, the function raises an error.
        dist_f: Disturbance force (N) acting on the CoM.
        dist_t: Disturbance torque (Nm) acting on the CoM.

    Returns:
        The derivatives of all state variables.
    """
    xp = array_namespace(pos)
    cmd_f = cmd[..., -1]
    cmd_rpy = cmd[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")

    rotor_vel_dot = None
    thrust = constants.DI_ACC[0] + constants.DI_ACC[1] * cmd_f

    drone_z_axis = rot.as_matrix()[..., -1]

    pos_dot = vel
    vel_dot = 1.0 / constants.MASS * thrust[..., None] * drone_z_axis + constants.GRAVITY_VEC
    if dist_f is not None:
        # Adding force disturbances to the state
        vel_dot = vel_dot + dist_f / constants.MASS
    vel_dot = xp.asarray(vel_dot)

    # Rotational equation of motion
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    rpy_rates = rotation.ang_vel2rpy_rates(quat, ang_vel)
    rpy_rates_dot = (
        constants.DI_PARAMS[:, 0] * euler_angles
        + constants.DI_PARAMS[:, 1] * rpy_rates
        + constants.DI_PARAMS[:, 2] * cmd_rpy
    )
    ang_vel_dot = rotation.rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)
    if dist_t is not None:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque given the inertia matrix
        torque = ang_vel_dot @ constants.J.mT + xp.linalg.cross(ang_vel, ang_vel @ constants.J.mT)

        # adding torque
        torque = torque + dist_t  # TODO rotation into body frame
        # back to angular acceleration
        ang_vel_dot = (
            torque - xp.linalg.cross(ang_vel, ang_vel @ constants.J.T)
        ) @ constants.J_INV.T

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
    # Creating force vector
    forces_motor_vec = cs.vertcat(
        0, 0, constants.DI_ACC[0] + constants.DI_ACC[1] * symbols.cmd_thrust
    )

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
        torque = constants.J @ ang_vel_dot + cs.cross(
            symbols.ang_vel, constants.J @ symbols.ang_vel
        )
        # adding torque
        torque = torque + symbols.dist_t  # TODO rotation into body frame
        # back to angular acceleration
        ang_vel_dot = constants.J_INV @ (
            torque - cs.cross(symbols.ang_vel, constants.J @ symbols.ang_vel)
        )

    X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y
