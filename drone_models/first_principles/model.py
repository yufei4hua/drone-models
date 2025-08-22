"""TODO."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace, device
from scipy.spatial.transform import Rotation as R

import drone_models.symbols as symbols
from drone_models.core import register_model_parameters, supports
from drone_models.first_principles.params import FirstPrinciplesParams
from drone_models.utils import rotation, to_xp

if TYPE_CHECKING:
    from array_api_typing import Array


@register_model_parameters(FirstPrinciplesParams)
@supports(rotor_dynamics=True)
def dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
    *,
    thrust_tau: float,
    KF: float,
    KM: float,
    L: float,
    mixing_matrix: Array,
    gravity_vec: Array,
    mass: float,
    J: Array,
    J_inv: Array,
) -> tuple[Array, Array, Array, Array, Array | None]:
    r"""First principles model for a quatrotor.

    The input consists of four forces in [N]. TODO more detail.

    Based on the quaternion model from https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-robot20.pdf

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        cmd: Motor speeds (rad/s).
        constants: Containing the constants of the drone.
        rotor_vel: Angular velocity of the 4 motors (rad/s). If None, the commanded thrust is
            directly applied. If value is given, thrust dynamics are calculated.
        dist_f: Disturbance force acting on the CoM (N).
        dist_t: Disturbance torque acting on the CoM (Nm).

    .. math::
        \sum_{i=1}^{\\infty} x_{i} TODO

    Warning:
        Do not use quat_dot directly for integration! Only usage of ang_vel is mathematically correct.
        If you still decide to use quat_dot to integrate, ensure unit length!
        More information https://ahrs.readthedocs.io/en/latest/filters/angular.html
    """
    xp = array_namespace(pos)
    mass, gravity_vec, KF, KM, L, mixing_matrix, J, J_inv = to_xp(
        mass, gravity_vec, KF, KM, L, mixing_matrix, J, J_inv, xp=xp, device=device(pos)
    )
    rot = R.from_quat(quat)
    # Rotor dynamics
    if rotor_vel is None:
        rotor_vel_dot = None
        rotor_vel = cmd
        warnings.warn("Rotor velocity is not provided, using commanded rotor velocity directly.")
    else:
        rotor_vel_dot = 1 / thrust_tau * (cmd - rotor_vel) - 1 / KM * rotor_vel**2
    # Creating force and torque vector
    forces_motor = KF * rotor_vel**2
    forces_motor_tot = xp.sum(forces_motor, axis=-1)
    zeros = xp.zeros_like(forces_motor_tot)
    forces_motor_vec = xp.stack((zeros, zeros, forces_motor_tot), axis=-1)
    # Torques in x & y are simply the force x distance.
    # Because there currently is no way to identify the z torque in relation to the thrust,
    # we rely on a old identified value that can compute rpm to torque.
    # force = kf * rpm², torque = km * rpm² => torque = km/kf*force TODO
    torques_motor_vec = forces_motor @ mixing_matrix * xp.stack([L, L, KM / KF])

    # Linear equation of motion
    forces_motor_vec_world = rot.apply(forces_motor_vec)
    forces_sum = forces_motor_vec_world + gravity_vec * mass
    if dist_f is not None:
        forces_sum = forces_sum + dist_f

    pos_dot = vel
    vel_dot = forces_sum / mass

    # Rotational equation of motion
    torques_sum = torques_motor_vec
    if dist_t is not None:
        torques_sum = torques_sum + rot.apply(dist_t, inverse=True)
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    ang_vel_dot = J_inv @ (torques_sum - xp.linalg.cross(ang_vel, J @ ang_vel))

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def dynamics_symbolic(
    calc_rotor_vel: bool = True,
    calc_dist_f: bool = False,
    calc_dist_t: bool = False,
    *,
    thrust_tau: float,
    KF: float,
    KM: float,
    L: float,
    sign_matrix: Array,
    gravity_vec: Array,
    mass: float,
    J: Array,
    J_inv: Array,
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
        rotor_vel_dot = 1 / thrust_tau * (U - symbols.rotor_vel) - 1 / KM * symbols.rotor_vel**2
        forces_motor = KF * symbols.rotor_vel**2
    else:
        forces_motor = KF * U**2

    # Creating force and torque vector
    forces_motor_vec = cs.vertcat(0, 0, cs.sum1(forces_motor))
    torques_motor_vec = forces_motor @ sign_matrix * cs.vertcat(L, L, KM / KF)

    # Linear equation of motion
    forces_motor_vec_world = symbols.rot @ forces_motor_vec
    forces_sum = forces_motor_vec_world + gravity_vec * mass
    if calc_dist_f is True:
        forces_sum = forces_sum + symbols.dist_f

    pos_dot = symbols.vel
    vel_dot = forces_sum / mass

    # Rotational equation of motion
    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    torques_sum = torques_motor_vec
    if calc_dist_t:
        torques_sum = torques_sum + symbols.rot.T @ symbols.dist_t
    ang_vel_dot = J_inv @ (torques_sum - cs.cross(symbols.ang_vel, J @ symbols.ang_vel))

    if calc_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y
