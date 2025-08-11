"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace
from scipy.spatial.transform import Rotation as R

import drone_models.models.symbols as symbols
from drone_models.utils import rotation

if TYPE_CHECKING:
    from array_api_typing import Array

    from drone_models.utils.constants import Constants


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
    r"""First principles model for a quatrotor.

    The input consists of four forces in [N]. TODO more detail.

    Based on the quaternion model from https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-robot20.pdf

    Args:
        pos: Position of the drone (m)
        quat: Quaternion of the drone (xyzw)
        vel: Velocity of the drone (m/s)
        ang_vel: Angular velocity of the drone (rad/s)
        cmd: RPYT command (roll, pitch, yaw in rad, thrust in N)
        constants: Containing the constants of the drone
        rotor_vel: Angular velocity of the 4 motors (rad/s). Defaults to None.
            If None, the commanded thrust is directly applied. If value is given, thrust dynamics are calculated.
        dist_f: Disturbance force acting on the CoM. Defaults to None.
        dist_t: Disturbance torque acting on the CoM. Defaults to None.

    .. math::
        \sum_{i=1}^{\\infty} x_{i} TODO

    Warning:
        Do not use quat_dot directly for integration! Only usage of ang_vel is mathematically correct.
        If you still decide to use quat_dot to integrate, ensure unit length!
        More information https://ahrs.readthedocs.io/en/latest/filters/angular.html
    """
    xp = array_namespace(pos)
    rot = R.from_quat(quat)
    # Thrust dynamics
    if rotor_vel is None:
        rotor_vel_dot = None
        rotor_vel = cmd
    else:
        rotor_vel_dot = (
            1 / constants.ROTOR_TAU * (cmd - rotor_vel) - 1 / constants.ROTOR_D * rotor_vel**2
        )
    # Creating force and torque vector
    forces_motor = constants.KF * rotor_vel**2
    forces_motor_tot = xp.sum(forces_motor, axis=-1)
    zeros = xp.zeros_like(forces_motor_tot)
    forces_motor_vec = xp.stack((zeros, zeros, forces_motor_tot), axis=-1)
    # Torques in x & y are simply the force x distance.
    # Because there currently is no way to identify the z torque in relation to the thrust,
    # we rely on a old identified value that can compute rpm to torque.
    # force = kf * rpm², torque = km * rpm² => torque = km/kf*force TODO
    torques_motor_vec = xp.matmul(forces_motor, constants.SIGN_MATRIX) * xp.stack(
        [constants.L, constants.L, constants.KM / constants.KF]
    )

    # Linear equation of motion
    forces_motor_vec_world = rot.apply(forces_motor_vec)

    force_world_frame = forces_motor_vec_world + constants.GRAVITY_VEC * constants.MASS
    if dist_f is not None:
        force_world_frame = force_world_frame + dist_f

    pos_dot = vel
    vel_dot = force_world_frame / constants.MASS

    # Rotational equation of motion
    torques = torques_motor_vec
    if dist_t is not None:
        # paper: rot.as_matrix() @ torques_dist
        torques = torques + rot.apply(dist_t)
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    ang_vel_dot = xp.matmul(
        torques - xp.linalg.cross(ang_vel, xp.matmul(ang_vel, xp.asarray(constants.J).T)),
        xp.asarray(constants.J_INV).T,
    )

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def dynamics_sybolic(
    constants: Constants,
    calc_rotor_vel: bool = True,
    calc_dist_f: bool = False,
    calc_dist_t: bool = False,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """TODO take from numeric."""
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if calc_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    if calc_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if calc_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = symbols.cmd_rotor_vel

    # Defining the dynamics function
    if calc_rotor_vel:
        # Thrust dynamics
        rotor_vel_dot = (
            1 / constants.ROTOR_TAU * (U - symbols.rotor_vel)
            - 1 / constants.ROTOR_D * symbols.rotor_vel**2
        )
        forces_motor = constants.KF * symbols.rotor_vel**2
    else:
        forces_motor = constants.KF * U**2

    # Creating force and torque vector
    forces_motor_vec = cs.vertcat(0, 0, cs.sum1(forces_motor))
    torques_motor_vec = (
        constants.SIGN_MATRIX.T
        @ forces_motor
        * cs.vertcat(constants.L, constants.L, constants.KM / constants.KF)
    )

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = symbols.rot @ forces_motor_vec / constants.MASS + constants.GRAVITY_VEC
    if calc_dist_f:
        # Adding force disturbances to the state
        vel_dot = vel_dot + symbols.dist_f / constants.MASS

    # Rotational equation of motion
    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    ang_vel_dot = constants.J_INV @ (
        torques_motor_vec - cs.cross(symbols.ang_vel, constants.J @ symbols.ang_vel)
    )
    if calc_dist_t:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = constants.J @ ang_vel_dot - cs.cross(
            symbols.ang_vel, constants.J @ symbols.ang_vel
        )
        # adding torque
        torque = torque + symbols.torques_dist
        # back to angular acceleration
        ang_vel_dot = constants.J_INV @ torque

    if calc_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y
