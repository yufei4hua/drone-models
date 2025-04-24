"""Testing the selfimplemented rotations against scipy rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
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


@pytest.mark.unit
def test_ang_vel2rpy_rates_two_way():
    """TODO."""
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())

    rpy_rates_two_way = R.ang_vel2rpy_rates(quats, ang_vels)
    ang_vels_two_way = R.rpy_rates2ang_vel(quats, rpy_rates_two_way)
    assert np.allclose(ang_vels, ang_vels_two_way), "Two way transform results are off."


@pytest.mark.unit
def test_ang_vel2rpy_rates_batching():
    """TODO."""
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())

    # Calculate batched version
    rpy_rates = R.ang_vel2rpy_rates(quats, ang_vels)

    # Compare to casadi implementation
    cs_quat, cs_ang_vel, cs_rpy_rates = R.casadi_ang_vel2rpy_rates()
    ang_vel2rpy_rates = cs.Function("ang_vel2rpy_rates", [cs_quat, cs_ang_vel], [cs_rpy_rates])
    for i in range(len(ang_vels)):
        # TODO test against casadi implementation
        rpy_rates_cs = np.array(ang_vel2rpy_rates(quats[i], ang_vels[i])).flatten()
        assert np.allclose(rpy_rates_cs, rpy_rates[i]), "Batched ang_vel2rpy_rates is wrong."
