"""Contains important dataclasses."""

from __future__ import absolute_import, annotations, division, print_function

from typing import TYPE_CHECKING, Callable

import numpy as np
from flax.struct import dataclass, field

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class QuadrotorState:  # TODO use in the models
    """TODO."""

    pos: NDArray[np.floating]  # Position in world frame
    quat: NDArray[np.floating]  # quaternion of drone in world frame
    vel: NDArray[np.floating]  # Velocity in world frame
    angvel: NDArray[np.floating]  # angvel of drone in body frame
    forces_motor: NDArray[np.floating]  # motor forces in body frame
    forces_dist: NDArray[np.floating] | None  # disturbance forces in world frame
    torques_dist: NDArray[np.floating] | None  # disturbance torques in world frame

    @classmethod
    def create(
        cls,
        pos: NDArray[np.floating],
        quat: NDArray[np.floating],
        vel: NDArray[np.floating],
        angvel: NDArray[np.floating],
        forces_motor: NDArray[np.floating],
        forces_dist: NDArray[np.floating] | None = None,
        torques_dist: NDArray[np.floating] | None = None,
    ) -> QuadrotorState:
        """TODO."""  # TODO check shapes.
        return cls(pos, quat, vel, angvel, forces_motor, forces_dist, torques_dist)

    @classmethod  # TODO add information to create (maybe to other function) like pos: NDArray[np.floating] | None
    def create_empty(cls, forces_dist: bool = False, torques_dist: bool = False) -> QuadrotorState:
        """TODO."""
        pos = np.array([0.0, 0.0, 0.0])
        vel = np.array([0.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        angvel = np.array([0.0, 0.0, 0.0])
        forces_motor = np.array([0.0, 0.0, 0.0, 0.0])
        if forces_dist:
            forces_dist = np.array([0.0, 0.0, 0.0])
        else:
            forces_dist = None
        if torques_dist:
            torques_dist = np.array([0.0, 0.0, 0.0])
        else:
            torques_dist = None

        return cls(pos, quat, vel, angvel, forces_motor, forces_dist, torques_dist)

    @classmethod
    def as_array(cls, state: QuadrotorState) -> NDArray[np.floating]:
        """TODO."""
        xp = state.pos.__array_namespace__()
        state_array = xp.concat(
            (state.pos, state.quat, state.vel, state.angvel, state.forces_motor), axis=-1
        )
        if state.forces_dist is not None:
            state_array = xp.concat((state_array, state.forces_dist), axis=-1)
        if state.torques_dist is not None:
            state_array = xp.concat((state_array, state.torques_dist), axis=-1)
        return state_array

    @classmethod
    def from_array(cls, state: QuadrotorState, array: NDArray[np.floating]) -> QuadrotorState:
        """TODO."""
        pos = array[..., 0:3]
        quat = array[..., 3:7]
        vel = array[..., 7:10]
        angvel = array[..., 10:13]
        forces_motor = array[..., 13:17]
        forces_dist = None
        torques_dist = None
        if state.forces_dist is not None:
            forces_dist = array[..., 17:20]
        if state.torques_dist is not None:
            if forces_dist is None:
                torques_dist = array[..., 17:20]
            else:
                torques_dist = array[..., 20:23]
        return cls(pos, quat, vel, angvel, forces_motor, forces_dist, torques_dist)

    @classmethod
    def state_dim(cls, state: QuadrotorState) -> int:
        """TODO."""
        dim = 17  # Base case, pos, quat, vel, angvel, forces_motor = 17
        if state.forces_dist is not None:
            dim += 3
        if state.torques_dist is not None:
            dim += 3
        return dim

    @classmethod  # TODO doesn't check if state is batched
    def get_entry(cls, state: QuadrotorState, i: int) -> QuadrotorState:
        """Returns the ith entry of the state, if batched."""
        pos = state.pos[i]
        quat = state.quat[i]
        vel = state.vel[i]
        angvel = state.angvel[i]
        forces_motor = state.forces_motor[i]
        forces_dist = state.forces_dist
        if forces_dist is not None:
            forces_dist = forces_dist[i]
        torques_dist = state.torques_dist
        if torques_dist is not None:
            torques_dist = torques_dist[i]

        return cls(pos, quat, vel, angvel, forces_motor, forces_dist, torques_dist)
