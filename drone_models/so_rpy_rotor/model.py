"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace
from array_api_compat import device as xp_device
from scipy.spatial.transform import Rotation as R

import drone_models.symbols as symbols
from drone_models.core import register_model_parameters, supports
from drone_models.so_rpy_rotor.params import SoRpyRotorParams
from drone_models.transform import motor_force2rotor_vel
from drone_models.utils import rotation, to_xp

if TYPE_CHECKING:
    from array_api_typing import Array


@register_model_parameters(SoRpyRotorParams)
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
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    KF: Array,
    KM: Array,
    rotor_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        cmd: Roll pitch yaw (rad) and collective thrust (N) command.
        rotor_vel: Speed of the 4 motors (RPMs). If None, the commanded thrust is directly
            applied (not recommended). If value is given, rotor dynamics are calculated.
        dist_f: Disturbance force (N) in the world frame acting on the CoM.
        dist_t: Disturbance torque (Nm) in the world frame acting on the CoM.

        mass: Mass of the drone (kg).
        gravity_vec: Gravity vector (m/s^2). We assume the gravity vector points downwards, e.g.
            [0, 0, -9.81].
        J: Inertia matrix (kg m^2).
        J_inv: Inverse inertia matrix (1/kg m^2).
        KF: Motor force constant (N/rad^2).
        KM: Motor torque constant (Nm/rad^2).
        rotor_coef: Coefficient for the rotor dynamics (1/s).
        acc_coef: Coefficient for the acceleration (1/s^2).
        cmd_f_coef: Coefficient for the collective thrust (N/rad^2).
        rpy_coef: Coefficient for the roll pitch yaw dynamics (1/s).
        rpy_rates_coef: Coefficient for the roll pitch yaw rates dynamics (1/s^2).
        cmd_rpy_coef: Coefficient for the roll pitch yaw command dynamics (1/s).

    Returns:
        tuple[Array, Array, Array, Array, Array | None]: _description_
    """
    xp = array_namespace(pos)
    # Convert constants to the correct framework and device
    device = xp_device(pos)
    mass, gravity_vec, KF, KM, J, J_inv = to_xp(
        mass, gravity_vec, KF, KM, J, J_inv, xp=xp, device=device
    )
    rotor_coef, acc_coef, cmd_f_coef = to_xp(rotor_coef, acc_coef, cmd_f_coef, xp=xp, device=device)
    rpy_coef, rpy_rates_coef, cmd_rpy_coef = to_xp(
        rpy_coef, rpy_rates_coef, cmd_rpy_coef, xp=xp, device=device
    )

    cmd_f = cmd[..., -1]
    cmd_rotor_vel = motor_force2rotor_vel(cmd_f / 4, KF)
    cmd_rpy = cmd[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")

    if rotor_vel is None:
        rotor_vel, rotor_vel_dot = cmd_rotor_vel[..., None], None
    else:
        rotor_vel_dot = 1 / rotor_coef * (cmd_rotor_vel[..., None] - rotor_vel) - KM * rotor_vel**2

    forces_motor = KF * xp.sum(rotor_vel**2, axis=-1)
    thrust = acc_coef + cmd_f_coef * forces_motor

    drone_z_axis = rot.as_matrix()[..., -1]

    pos_dot = vel
    vel_dot = 1.0 / mass * thrust[..., None] * drone_z_axis + gravity_vec
    if dist_f is not None:
        # Adding force disturbances to the state
        vel_dot = vel_dot + dist_f / mass
    vel_dot = xp.asarray(vel_dot)

    # Rotational equation of motion
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    rpy_rates = rotation.ang_vel2rpy_rates(quat, ang_vel)
    rpy_rates_dot = rpy_coef * euler_angles + rpy_rates_coef * rpy_rates + cmd_rpy_coef * cmd_rpy
    ang_vel_dot = rotation.rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)
    if dist_t is not None:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque given the inertia matrix
        torque = (J @ ang_vel_dot[..., None])[..., 0]
        torque = torque + xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        # adding torque. TODO: This should be a linear transformation. Can't we just transform the
        # disturbance torque to an ang_vel_dot summand directly?
        torque = torque + rot.apply(dist_t, inverse=True)
        # back to angular acceleration
        torque = torque - xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        ang_vel_dot = (J_inv @ torque[..., None])[..., 0]

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


@register_model_parameters(SoRpyRotorParams)
def symbolic_dynamics(
    model_rotor_vel: bool = False,
    model_dist_f: bool = False,
    model_dist_t: bool = False,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    KF: Array,
    KM: Array,
    rotor_coef: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    TODO.
    """
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    if model_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if model_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = cs.vertcat(symbols.cmd_roll, symbols.cmd_pitch, symbols.cmd_yaw, symbols.cmd_thrust)
    cmd_rpy = cs.vertcat(symbols.cmd_roll, symbols.cmd_pitch, symbols.cmd_yaw)

    # Defining the dynamics function
    if model_rotor_vel:
        # motor_force2rotor_vel
        cmd_rotor_vel = cs.sqrt(symbols.cmd_thrust / 4 / KF)
        rotor_vel_dot = (
            1 / rotor_coef * (cmd_rotor_vel - symbols.rotor_vel) - KM * symbols.rotor_vel**2
        )
        forces_motor = KF * cs.sum1(symbols.rotor_vel**2)
    else:
        forces_motor = symbols.cmd_thrust
    # Creating force vector
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * forces_motor)

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = symbols.rot @ forces_motor_vec / mass + gravity_vec
    if model_dist_f:
        # Adding force disturbances to the state
        vel_dot = vel_dot + symbols.dist_f / mass

    # Rotational equation of motion
    euler_angles = rotation.cs_quat2euler(symbols.quat)

    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    rpy_rates = rotation.cs_ang_vel2rpy_rates(symbols.quat, symbols.ang_vel)
    rpy_rates_dot = rpy_coef * euler_angles + rpy_rates_coef * rpy_rates + cmd_rpy_coef * cmd_rpy
    ang_vel_dot = rotation.cs_rpy_rates_deriv2ang_vel_deriv(symbols.quat, rpy_rates, rpy_rates_dot)
    if model_dist_t:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = J @ ang_vel_dot + cs.cross(symbols.ang_vel, J @ symbols.ang_vel)
        # adding torque
        torque = torque + symbols.rot.T @ symbols.dist_t
        # back to angular acceleration
        ang_vel_dot = J_inv @ (torque - cs.cross(symbols.ang_vel, J @ symbols.ang_vel))

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y
