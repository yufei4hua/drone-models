"""Testing the selfimplemented rotations against scipy rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jp
import numpy as np
import pytest

import lsy_models.utils.rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor

tol = 1e-6  # Since Jax by default works with 32 bit, the precision is worse


def create_uniform_quats(N: int = 100, scale: float = 10) -> list:
    """Creates an (n, 4) list with random quaternions."""
    # larger range because the function should be able to handle wrong length quaternions
    quats = np.random.uniform(
        -np.array([1, 1, 1, 1]) * scale, np.array([1, 1, 1, 1]) * scale, size=(N, 4)
    )
    return quats.tolist()


def create_lock_quats() -> list:
    """Create the quaternions which would lead to gimbal lock in rpy represenation."""
    quats = [
        [1, 1, 1, 1]  # TODO fill in all the gimbal lock quaternions
    ]
    return quats


def create_uniform_ang_vel(N: int = 100, scale: float = 10) -> list:
    """Creates an (n, 4) list with random quaternions."""
    # larger range because the function should be able to handle wrong length quaternions
    ang_vel = np.random.uniform(
        -np.array([1, 1, 1]) * scale, np.array([1, 1, 1]) * scale, size=(N, 3)
    )
    return ang_vel.tolist()


@pytest.mark.unit
def test_rot_from_quat():
    """Testing Quaternion to Euler angle with individual arrays."""
    quats = create_uniform_quats()
    quats.extend(create_lock_quats())

    # Testing individual
    for q in quats:
        R.from_quat(np.array(q))
        R.from_quat(jp.array(q))

    # Testing batched
    R.from_quat(np.array(quats))
    R.from_quat(jp.array(quats))


@pytest.mark.unit
def test_rot_from_euler():
    """Testing Quaternion to Euler angle with individual arrays."""
    ...  # TODO


def simple_ang_vel2rpy_rates(ang_vel: Array, quat: Array) -> Array:
    """Convert angular velocity to rpy rates.

    Args:
        ang_vel: The angular velocity in the body frame.
        quat: The current orientation.

    Returns:
        The rpy rates in the body frame, following the 'xyz' convention.
    """
    xp = quat.__array_namespace__()
    rpy = R.from_quat(quat).as_euler("xyz")
    sin_phi, cos_phi = xp.sin(rpy[0]), xp.cos(rpy[0])
    cos_theta, tan_theta = xp.cos(rpy[1]), xp.tan(rpy[1])
    conv_mat = xp.array(
        [
            [1, sin_phi * tan_theta, cos_phi * tan_theta],
            [0, cos_phi, -sin_phi],
            [0, sin_phi / cos_theta, cos_phi / cos_theta],
        ]
    )
    return conv_mat @ ang_vel


def simple_rpy_rates2ang_vel(rpy_rates: Array, quat: Array) -> Array:
    """Convert rpy rates to angular velocity."""
    xp = quat.__array_namespace__()
    rpy = R.from_quat(quat).as_euler("xyz")
    sin_phi, cos_phi = xp.sin(rpy[0]), xp.cos(rpy[0])
    cos_theta, tan_theta = xp.cos(rpy[1]), xp.tan(rpy[1])
    conv_mat = xp.array(
        [
            [1, 0, -cos_theta * tan_theta],
            [0, cos_phi, sin_phi * cos_theta],
            [0, -sin_phi, cos_phi * cos_theta],
        ]
    )
    return conv_mat @ rpy_rates


@pytest.mark.unit
def test_ang_vel2rpy_rates_two_way():
    """TODO."""
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())

    rpy_rates_two_way = R.ang_vel2rpy_rates(ang_vels, quats)
    ang_vels_two_way = R.rpy_rates2ang_vel(rpy_rates_two_way, quats)
    assert np.allclose(ang_vels, ang_vels_two_way), "Two way transform results are off."


@pytest.mark.unit
def test_ang_vel2rpy_rates_batching():
    """TODO."""
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())

    rpy_rates = R.ang_vel2rpy_rates(ang_vels, quats)

    for i in range(len(ang_vels)):
        rpy_rates_simple = simple_ang_vel2rpy_rates(ang_vels[i], quats[i])
        assert np.allclose(rpy_rates_simple, rpy_rates[i]), "Batched ang_vel2rpy_rates is wrong."
