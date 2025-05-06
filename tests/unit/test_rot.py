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


def create_uniform_quats(N: int = 1000, scale: float = 10) -> list:
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


def create_uniform_ang_vel(N: int = 1000, scale: float = 10) -> list:
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
    rpy_rates_batched = R.ang_vel2rpy_rates(quats, ang_vels)

    # Compare to non-batched version
    for i in range(len(ang_vels)):
        rpy_rates_non_batched = R.ang_vel2rpy_rates(quats[i], ang_vels[i])
        assert np.allclose(rpy_rates_non_batched, rpy_rates_batched[i]), (
            "Batched and non-batched results differ."
        )


@pytest.mark.unit
def test_rpy_rates2ang_vel_batching():
    """TODO."""
    quats = np.array(create_uniform_quats())
    rpy_rates = np.array(create_uniform_ang_vel())

    # Calculate batched version
    ang_vel_batched = R.rpy_rates2ang_vel(quats, rpy_rates)

    # Compare to non-batched version
    for i in range(len(rpy_rates)):
        ang_vel_non_batched = R.rpy_rates2ang_vel(quats[i], rpy_rates[i])
        assert np.allclose(ang_vel_non_batched, ang_vel_batched[i]), (
            "Batched and non-batched results differ."
        )


@pytest.mark.unit
def test_ang_vel2rpy_rates_symbolic():
    """TODO."""
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())

    # Calculate numeric version
    rpy_rates = R.ang_vel2rpy_rates(quats, ang_vels)

    # Compare to casadi implementation
    cs_quat, cs_ang_vel, cs_rpy_rates = R.cs_ang_vel2rpy_rates()
    cs_ang_vel2rpy_rates = cs.Function("ang_vel2rpy_rates", [cs_quat, cs_ang_vel], [cs_rpy_rates])
    for i in range(len(ang_vels)):
        rpy_rates_cs = np.array(cs_ang_vel2rpy_rates(quats[i], ang_vels[i])).flatten()
        assert np.allclose(rpy_rates_cs, rpy_rates[i]), "Symbolic and numeric results differ."


@pytest.mark.unit
def test_rpy_rates2ang_vel_symbolic():
    """TODO."""
    quats = np.array(create_uniform_quats())
    rpy_rates = np.array(create_uniform_ang_vel())

    # Calculate numeric version
    ang_vels = R.rpy_rates2ang_vel(quats, rpy_rates)

    # Compare to casadi implementation
    cs_quat, cs_rpy_rates, cs_ang_vel = R.cs_rpy_rates2ang_vel()
    cs_rpy_rates2ang_vel = cs.Function("rpy_rates2ang_vel", [cs_quat, cs_rpy_rates], [cs_ang_vel])
    for i in range(len(rpy_rates)):
        ang_vel_cs = np.array(cs_rpy_rates2ang_vel(quats[i], rpy_rates[i])).flatten()
        assert np.allclose(ang_vel_cs, ang_vels[i]), "Symbolic and numeric results differ."


@pytest.mark.unit
def test_ang_vel_deriv2rpy_rates_deriv_two_way():
    """TODO."""
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())
    ang_vels_deriv = np.array(create_uniform_ang_vel())
    rpy_rates = R.ang_vel2rpy_rates(quats, ang_vels)

    rpy_rates_deriv_two_way = R.ang_vel_deriv2rpy_rates_deriv(quats, ang_vels, ang_vels_deriv)
    ang_vels_deriv_two_way = R.rpy_rates_deriv2ang_vel_deriv(
        quats, rpy_rates, rpy_rates_deriv_two_way
    )
    assert np.allclose(ang_vels_deriv, ang_vels_deriv_two_way), "Two way transform results are off."


@pytest.mark.unit
def test_ang_vel_deriv2rpy_rates_deriv_batching():
    """TODO."""
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())
    ang_vels_deriv = np.array(create_uniform_ang_vel())

    # Calculate batched version
    rpy_rates_deriv_batched = R.ang_vel_deriv2rpy_rates_deriv(quats, ang_vels, ang_vels_deriv)

    # Compare to non-batched version
    for i in range(len(ang_vels)):
        rpy_rates_deriv_non_batched = R.ang_vel_deriv2rpy_rates_deriv(
            quats[i], ang_vels[i], ang_vels_deriv[i]
        )
        assert np.allclose(rpy_rates_deriv_non_batched, rpy_rates_deriv_batched[i]), (
            "Batched and non-batched results differ."
        )


@pytest.mark.unit
def test_rpy_rates_deriv2ang_vel_deriv_batching():
    """TODO."""
    quats = np.array(create_uniform_quats())
    rpy_rates = np.array(create_uniform_ang_vel())
    rpy_rates_deriv = np.array(create_uniform_ang_vel())

    # Calculate batched version
    ang_vels_deriv_batched = R.rpy_rates_deriv2ang_vel_deriv(quats, rpy_rates, rpy_rates_deriv)

    # Compare to non-batched version
    for i in range(len(rpy_rates)):
        ang_vels_deriv_non_batched = R.rpy_rates_deriv2ang_vel_deriv(
            quats[i], rpy_rates[i], rpy_rates_deriv[i]
        )
        assert np.allclose(ang_vels_deriv_non_batched, ang_vels_deriv_batched[i]), (
            "Batched and non-batched results differ."
        )


# @pytest.mark.unit
# def test_ang_vel_deriv2rpy_rates_deriv_symbolic():
#     """TODO."""
#     quats = np.array(create_uniform_quats())
#     ang_vels = np.array(create_uniform_ang_vel())
#     ang_vels_deriv = np.array(create_uniform_ang_vel())
#     rpy_rates = R.ang_vel2rpy_rates(quats, ang_vels)

#     # Calculate batched version
#     rpy_rates_deriv = R.ang_vel_deriv2rpy_rates_deriv(quats, ang_vels, ang_vels_deriv)

#     # Compare to casadi implementation
#     cs_quat, cs_ang_vel, cs_rpy_rates = R.cs_ang_vel2rpy_rates()
#     ang_vel2rpy_rates = cs.Function("ang_vel2rpy_rates", [cs_quat, cs_ang_vel], [cs_rpy_rates])
#     for i in range(len(ang_vels)):
#         # TODO test against casadi implementation
#         rpy_rates_cs = np.array(ang_vel2rpy_rates(quats[i], ang_vels[i])).flatten()
#         assert np.allclose(rpy_rates_cs, rpy_rates[i]), "Batched and non-batched results differ."
