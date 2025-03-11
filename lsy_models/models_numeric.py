"""This file contains all the numeric models for a generic quatrotor drone. The parameters need to be stored in the corresponding xml file."""

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


def quat_dot_from_angvel(quat: Array, angvel: Array) -> Array:
    """Calculates the quaternion derivative based on an angular velocity."""
    xp = quat.__array_namespace__()
    x, y, z = xp.split(angvel, 3, axis=-1)
    angvel_skew = xp.stack(
        [
            xp.concat((xp.zeros_like(x), -z, y), axis=-1),
            xp.concat((z, xp.zeros_like(x), -x), axis=-1),
            xp.concat((-y, x, xp.zeros_like(x)), axis=-1),
        ],
        axis=-2,
    )
    xi1 = xp.insert(-angvel, 0, 0, axis=-1)  # First line of xi
    xi2 = xp.concat((xp.expand_dims(angvel.T, axis=0).T, -angvel_skew), axis=-1)
    xi = xp.concat((xp.expand_dims(xi1, axis=-2), xi2), axis=-2)
    return 0.5 * xp.matvec(xi, quat)
    # return 0.5 * (xi @ quat[..., None]).squeeze(axis=-1)


def f_first_principles(
    pos: Array,
    quat: Array,
    vel: Array,
    angvel: Array,
    command: Array,
    constants: Constants,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """First principles model for a quatrotor.

    The input consists of four forces in [N]. TODO more detail.

    Based on the quaternion model from https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-robot20.pdf

    Warning:
        Do not use quat_dot directly for integration! Only usage of angvel is mathematically correct.
        If you still decide to use quat_dot to integrate, ensure unit length!
        More information https://ahrs.readthedocs.io/en/latest/filters/angular.html

    forces_motor are the four indiviual forces of the propellers in body frame
    forces_dist and torques_dist are vectors in world frame
    """
    xp = pos.__array_namespace__()  # This looks into the type of the position array and decides what implementation to use (numpy, jax, etc)
    rot = R.from_quat(quat)

    # Thrust dynamics
    if forces_motor is None:
        forces_motor_dot = None
        forces_motor = command
    else:
        forces_motor_dot = constants.THRUST_TAU * (command - forces_motor)  # TODO add dt = 1/200
    # Creating force and torque vector
    forces_motor_tot = xp.sum(forces_motor, axis=-1)
    # forces_motor_tot = xp.sum(
    #     command, axis=-1
    # )  # Without motor dynamics TODO make motor forces None
    zeros = xp.zeros_like(forces_motor_tot)
    forces_motor_vec = xp.stack((zeros, zeros, forces_motor_tot), axis=-1)
    # Torques in x & y are simply the force x distance.
    # Because there currently is no way to identify the z torque in relation to the thrust,
    # we rely on a old identified value that can compute rpm to torque.
    # force = kf * rpm², torque = km * rpm² => torque = km/kf*force
    torques_motor_vec = xp.vecmat(forces_motor, constants.SIGN_MATRIX) * xp.array(
        [constants.L, constants.L, constants.KM / constants.KF]
    )

    # Linear equation of motion
    forces_motor_vec_world = rot.apply(forces_motor_vec)

    force_world_frame = forces_motor_vec_world + constants.GRAVITY_VEC * constants.MASS
    if forces_dist is not None:
        force_world_frame = force_world_frame + forces_dist

    pos_dot = vel
    vel_dot = force_world_frame / constants.MASS

    # Rotational equation of motion

    torques = torques_motor_vec
    if torques_dist is not None:
        # paper: rot.as_matrix() @ torques_dist
        torques = torques + rot.apply(torques_dist)
    quat_dot = quat_dot_from_angvel(quat, angvel)
    angvel_dot = xp.matvec(
        constants.J_INV, torques - xp.cross(angvel, xp.matvec(constants.J, angvel))
    )

    return pos_dot, quat_dot, vel_dot, angvel_dot, forces_motor_dot


def f_fitted_DI_rpy(
    pos: Array,
    quat: Array,
    vel: Array,
    angvel: Array,
    command: Array,
    constants: Constants,
    forces_motor: Array | None = None,
    forces_dist: Array | None = None,
    torques_dist: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """TODO."""
    xp = pos.__array_namespace__()
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")
    rpy_rates = rot.apply(angvel)  # is this correct?

    # TODO thrust dynamics?
    if forces_motor is not None:
        raise NotImplementedError("Thrust dynamics can currently not be simulated")

    # command = command * 0
    # Linear equation of motion
    forces_motor_tot = cf2.pwm2force(command[..., -1], constants)
    coeff = constants.DI_ACC[0] * forces_motor_tot + constants.DI_ACC[1]
    cos_x3 = xp.cos(euler_angles[..., 0])  # roll
    sin_x3 = xp.sin(euler_angles[..., 0])  # roll
    cos_x4 = xp.cos(euler_angles[..., 1])  # pitch
    sin_x4 = xp.sin(euler_angles[..., 1])  # pitch
    cos_x5 = xp.cos(euler_angles[..., 2])  # yaw
    sin_x5 = xp.sin(euler_angles[..., 2])  # yaw
    rotation_matrix = xp.stack(
        [
            cos_x3 * sin_x4 * cos_x5 + sin_x3 * sin_x5,
            cos_x3 * sin_x4 * sin_x5 - sin_x3 * cos_x5,
            cos_x3 * cos_x4,
        ],
        axis=-1,
    )
    pos_dot = vel
    vel_dot = coeff * rotation_matrix + constants.GRAVITY_VEC

    # Rotational equation of motion
    quat_dot = quat_dot_from_angvel(quat, angvel)
    rpy_rates_dot = (
        constants.DI_PARAMS[:, 0] * euler_angles
        + constants.DI_PARAMS[:, 1] * rpy_rates
        + constants.DI_PARAMS[:, 2] * command[..., 0:3]
    )
    angvel_dot = rot.apply(rpy_rates_dot, inverse=True)  # is this correct?

    # WARNING: This is the surrogate addition to the model and not very realistic!
    # adding disturbances to the state
    if forces_dist is not None:
        vel_dot = vel_dot + forces_dist / constants.MASS
    if torques_dist is not None:
        # adding disturbances to the state
        # adding torque is a little more complex:
        # angular acceleration can be converted to torque
        torque = xp.matvec(constants.J, angvel_dot) - xp.cross(
            angvel, xp.matvec(constants.J, angvel)
        )
        # adding torque
        torque = torque + torques_dist
        # back to angular acceleration
        angvel_dot = xp.matvec(constants.J_INV, torque)

    return pos_dot, quat_dot, vel_dot, angvel_dot, None


# f = model_dynamics("cf2x+", "analytical")

# # Single value
# f(np.array([1,1,1]), np.array([1,1,1]), np.array([0,0,0,1]), np.array([1,1,1]), np.array([1,1,1,1]), np.array([1,1,1,1]))

# # Batched value
# f(np.array([[1,1,1],[1,1,1]]), np.array([[1,1,1],[1,1,1]]), np.array([[0,0,0,1],[0,0,0,1]]),
#   np.array([[1,1,1],[1,1,1]]), np.array([[1,1,1,1],[1,1,1,1]]), np.array([[1,1,1,1],[1,1,1,1]]))
