"""This file contains all the numeric models for a generic quatrotor drone. The parameters need to be stored in the corresponding xml file"""


from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

import lsy_models.const as const
from lsy_models import rotation as R

if TYPE_CHECKING:
    from types import FunctionType

    from numpy.typing import NDArray

# C = const.Constants.create("data/cf2x_-B250.xml")

def model_dynamics(model: str, method: str) -> callable:
    match model: # TODO make constants in jp or np
        case "cf2x+":
            C = const.Constants.create("data/cf2x_-B250.xml")
        case "cf2x-":
            C = const.Constants.create("data/cf2x_+B250.xml")
        case _:
            raise ValueError(f"Model '{model}' is not supported")
        
    match method:
        case "analytical":
            return partial(f_analytical, C=C)
        case "fit_DI":
            ...
        case _:
            raise ValueError(f"Method '{method}' is not supported")

def f_analytical(pos: NDArray[np.floating], vel: NDArray[np.floating], quat: NDArray[np.floating], angvel: NDArray[np.floating],
                 forces_motor: NDArray[np.floating], forces_cmd: NDArray[np.floating], C: const.Constants, 
                 forces_dist: NDArray[np.floating] | None = None, torques_dist: NDArray[np.floating] | None = None):
    """The input consists of four forces. TODO more detail.
    
    Based on the quaternion model from https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-robot20.pdf

    forces_motor are the four indiviual forces of the propellers in body frame
    forces_dist and torques_dist are vectors in world frame
    """
    xp = pos.__array_namespace__() # This looks into the type of the position array and decides what implementation to use (numpy, jax, etc)
    rot = R.from_quat(quat)

    # Thrust dynamics
    force_motor_tot = xp.sum(forces_motor, axis=-1)
    zeros = xp.zeros_like(force_motor_tot)
    force_motor_tot = xp.stack((zeros, zeros, force_motor_tot), axis=-1)
    forces_motor_dot = C.KD*(forces_cmd-forces_motor)
    # Because there currently is no way to identify the yaw torque in relation to the thrust,
    # we rely on a old identified value that can compute rpm to torque.
    torque_motor_tot = xp.sum(C.KM/C.KF*forces_motor, axis=-1) # force = kf * rpm², torque = km * rpm² => torque = km/kf*force
    torque_motor_tot = xp.stack((zeros, zeros, torque_motor_tot), axis=-1)   

    # Linear equation of motion
    force_motor_tot_world = rot.apply(force_motor_tot)
    if forces_dist is not None:
        force_world_frame = force_motor_tot_world + C.GRAVITY_VEC*C.MASS + forces_dist
    else:
        force_world_frame = force_motor_tot_world + C.GRAVITY_VEC*C.MASS
    pos_dot = vel
    vel_dot = force_world_frame / C.MASS

    # Rotational equation of motion
    # print(f"angvel.shape={angvel.shape}")
    # angvel_skew = xp.linalg.cross(xp.reshape(angvel, (-1,1,3)), xp.identity(3) * -1, axis=-1) # creates the skew symmetric matrix
    # angvel_skew = xp.triu(angvel) - xp.triu(angvel).T
    x, y, z = xp.split(angvel, 3, axis=-1)
    angvel_skew = xp.stack(
            [
                xp.concatenate([xp.zeros_like(x), -z, y], axis=-1),
                xp.concatenate([z, xp.zeros_like(x), -x], axis=-1),
                xp.concatenate([-y, x, xp.zeros_like(x)], axis=-1),
            ],
            axis=-2,
        ).squeeze() # from jaxsim.math.skew
    xi1 = xp.insert(angvel, 0, 0, axis=-1) # First line of xi
    xi2 = xp.concat((xp.expand_dims(angvel.T, axis=0).T, angvel_skew), axis=-1)
    xi = xp.concat((xp.expand_dims(xi1, axis=-2), xi2), axis=-2)
    if torques_dist is not None:
        torques = torque_motor_tot + rot.as_matrix() @ torques_dist # TODO this rotation can be done better
    else:
        torques = torque_motor_tot
    quat_dot = 0.5*(xi @ quat.T)
    # angvel_dot = C.J_inv @ (torques - xp.cross(angvel, C.J@angvel)) # non batchable, but easy version
    angvel_dot = (torques - xp.cross(angvel, angvel@C.J)) @ C.J_inv # batchable version

    return pos_dot, vel_dot, quat_dot, angvel_dot, forces_motor_dot
    
f = model_dynamics("cf2x+", "analytical")

# Single value
f(np.array([1,1,1]), np.array([1,1,1]), np.array([0,0,0,1]), np.array([1,1,1]), np.array([1,1,1,1]), np.array([1,1,1,1]))

# Batched value 
f(np.array([[1,1,1],[1,1,1]]), np.array([[1,1,1],[1,1,1]]), np.array([[0,0,0,1],[0,0,0,1]]), 
  np.array([[1,1,1],[1,1,1]]), np.array([[1,1,1,1],[1,1,1,1]]), np.array([[1,1,1,1],[1,1,1,1]]))