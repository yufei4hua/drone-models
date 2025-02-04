"""This file contains all the numeric models for a generic quatrotor drone. The parameters need to be stored in the corresponding xml file"""


from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R
import const

if TYPE_CHECKING:
    from types import FunctionType

    from numpy.typing import NDArray

C = const.Constants.create("cf2x_-B250.xml")

def f_analytical(pos: NDArray[np.floating], vel: NDArray[np.floating], quat: NDArray[np.floating], omega: NDArray[np.floating],
                 force_motor: NDArray[np.floating], torque_motor: NDArray[np.floating], cmd: NDArray[np.floating], 
                 force_dist: NDArray[np.floating] | None = None, torque_dist: NDArray[np.floating] | None = None):
    """The input consists of four forces. TODO proper description."""
    xp = pos.__array_namespace__() # This looks into the type of the position array and decides what implementation to use (numpy, jax, etc)
    rot = R.from_quat(quat)

    # Thrust dynamics
    force_motor_deriv = C.KD*(cmd-force_motor)
    # Because there currently is no way to identify the yaw torque in relation to the thrust,
    # we rely on a old identified value that can compute rpm to torque.
    torque_motor = C.KM/C.KF*force_motor # force = kf * rpm², torque = km * rpm² => torque = km/kf*force

    # Equation of motion
    thrust = xp.sum(force_motor[..., :]) # TODO make both batched and non batches version
    thrust
    thrust_world_frame = rot.apply(thrust)
    if force_dist is not None:
        force_world_frame = thrust_world_frame + C.GRAVITY_VEC*C.MASS + force_dist
    else:
        force_world_frame = thrust_world_frame + C.GRAVITY_VEC*C.MASS
    pos_dot = vel
    vel_dot = force_world_frame / C.MASS

    # Rotation
    # TODO
    

f_analytical(np.array([1,1,1]), np.array([1,1,1]), np.array([0,0,0,1]), np.array([1,1,1]), np.array([1,1,1,1]), np.array([1,1,1,1]), np.array([0,0,0,0]))