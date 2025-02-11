"""This file contains all the numeric models for a generic quatrotor drone. The parameters need to be stored in the corresponding xml file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lsy_models.utils.rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor

    from lsy_models.utils.constants import Constants


def f_first_principles(
    pos: Array,
    quat: Array,
    vel: Array,
    angvel: Array,
    forces_motor: Array,
    command: Array,
    constants: Constants,
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
    forces_motor_dot = constants.KD * (command - forces_motor)  # TODO add dt = 1/200
    # forces_motor_dot = forces_motor * 0  # Without motor dynamics TODO make motor forces None
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
    torques_motor_vec = (forces_motor @ constants.SIGN_MATRIX) * xp.array(
        [constants.L, constants.L, constants.KM / constants.KF]
    )

    # Linear equation of motion
    forces_motor_vec_world = rot.apply(forces_motor_vec)
    if forces_dist is not None:
        force_world_frame = (
            forces_motor_vec_world + constants.GRAVITY_VEC * constants.MASS + forces_dist
        )
    else:
        force_world_frame = forces_motor_vec_world + constants.GRAVITY_VEC * constants.MASS
    pos_dot = vel
    vel_dot = force_world_frame / constants.MASS

    # Rotational equation of motion
    x, y, z = xp.split(angvel, 3, axis=-1)
    angvel_skew = xp.stack(
        [
            xp.concat((xp.zeros_like(x), -z, y), axis=-1),
            xp.concat((z, xp.zeros_like(x), -x), axis=-1),
            xp.concat((-y, x, xp.zeros_like(x)), axis=-1),
        ],
        axis=-2,
    )  # .squeeze() # from jaxsim.math.skew
    xi1 = xp.insert(-angvel, 0, 0, axis=-1)  # First line of xi
    xi2 = xp.concat((xp.expand_dims(angvel.T, axis=0).T, -angvel_skew), axis=-1)
    xi = xp.concat((xp.expand_dims(xi1, axis=-2), xi2), axis=-2)
    if torques_dist is not None:
        # paper: rot.as_matrix() @ torques_dist
        torques = torques_motor_vec + rot.apply(torques_dist)
    else:
        torques = torques_motor_vec
    quat_dot = 0.5 * (xi @ quat[..., None]).squeeze(axis=-1)
    angvel_dot = (
        torques - xp.cross(angvel, angvel @ constants.J)
    ) @ constants.J_inv  # batchable version

    return pos_dot, quat_dot, vel_dot, angvel_dot, forces_motor_dot


# def f_fitted_DI(
#     pos: NDArray[np.floating],
#     quat: NDArray[np.floating],
#     vel: NDArray[np.floating],
#     angvel: NDArray[np.floating],
#     forces_motor: NDArray[np.floating],
#     RPYT_cmd: NDArray[np.floating],
#     constants: Constants,
#     forces_dist: NDArray[np.floating] | None = None,
#     torques_dist: NDArray[np.floating] | None = None,
# ) -> tuple[
#     NDArray[np.floating],
#     NDArray[np.floating],
#     NDArray[np.floating],
#     NDArray[np.floating],
#     NDArray[np.floating],
# ]:
#     """TODO."""
#     xp = pos.__array_namespace__()
#     rot = R.from_quat(quat)
#     euler_angles = rot.as_euler("xyz")

#     # Linear equation of motion
#     coeff = constants.DI_ACC[0] * RPYT_cmd[..., 3] + constants.DI_ACC[1]
#     cos_x3 = np.cos(x[:, 3])
#     sin_x3 = np.sin(x[:, 3])
#     cos_x4 = np.cos(x[:, 4])
#     sin_x4 = np.sin(x[:, 4])
#     cos_x5 = np.cos(x[:, 5])
#     sin_x5 = np.sin(x[:, 5])
#     if forces_dist is not None:
#         force_world_frame = (
#             forces_motor_vec_world + constants.GRAVITY_VEC * constants.MASS + forces_dist
#         )
#     else:
#         force_world_frame = forces_motor_vec_world + constants.GRAVITY_VEC * constants.MASS
#     pos_dot = vel
#     vel_dot = force_world_frame / constants.MASS

#     return pos_dot, vel_dot, quat_dot, angvel_dot, forces_motor_dot


# f = model_dynamics("cf2x+", "analytical")

# # Single value
# f(np.array([1,1,1]), np.array([1,1,1]), np.array([0,0,0,1]), np.array([1,1,1]), np.array([1,1,1,1]), np.array([1,1,1,1]))

# # Batched value
# f(np.array([[1,1,1],[1,1,1]]), np.array([[1,1,1],[1,1,1]]), np.array([[0,0,0,1],[0,0,0,1]]),
#   np.array([[1,1,1],[1,1,1]]), np.array([[1,1,1,1],[1,1,1,1]]), np.array([[1,1,1,1],[1,1,1,1]]))
