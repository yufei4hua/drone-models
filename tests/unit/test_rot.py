"""Testing the selfimplemented rotations against scipy rotations."""

import jax.numpy as jp
import numpy as np
import pytest

import lsy_models.utils.rotation as R

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
