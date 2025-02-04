"""Testing the selfimplemented rotations against scipy rotations."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import lsy_models.rotation as rot

tol = 1e-6 # Since Jax by default works with 32 bit, the precision is worse

def create_uniform_quats(n=10, range=10) -> list:
    """Creates an (n, 4) list with random quaternions."""
    # larger range because the function should be able to handle wrong length quaternions
    quats = np.random.uniform(-np.array([1,1,1,1])*range, np.array([1,1,1,1])*range, size=(n,4)) 
    return quats.tolist()

def create_lock_quats() -> list:
    """Create the quaternions which would lead to gimbal lock in rpy represenation."""
    quats = [
            [1,1,1,1], # TODO fill in all the gimbal lock quaternions
            ]
    return quats

@pytest.mark.unit
def test_quat2euler1D():
    """Testing Quaternion to Euler angle with individual arrays."""
    quats = create_uniform_quats()
    quats.extend(create_lock_quats())

    for q in quats:
        rotation = R.from_quat(q)
        assert np.allclose(rotation.as_euler("xyz"), rot.quat2euler(np.array(q), "xyz"))
        assert np.allclose(rotation.as_euler("XYZ"), rot.quat2euler(np.array(q), "XYZ"))
        assert jnp.allclose(rotation.as_euler("xyz"), rot.quat2euler(jnp.array(q), "xyz"), atol=tol)
        assert jnp.allclose(rotation.as_euler("XYZ"), rot.quat2euler(jnp.array(q), "XYZ"), atol=tol)


@pytest.mark.unit
def test_quat2euler2D():
    """Testing Quaternion to Euler angle with batched arrays."""
    quats = create_uniform_quats()
    # quats /= np.linalg.norm(quats) # normalization

    rotations = R.from_quat(quats)

    assert np.allclose(rotations.as_euler("xyz"), rot.quat2euler(np.array(quats), "xyz"))
    assert np.allclose(rotations.as_euler("XYZ"), rot.quat2euler(np.array(quats), "XYZ"))
    assert jnp.allclose(rotations.as_euler("xyz"), rot.quat2euler(jnp.array(quats), "xyz"), atol=tol)
    assert jnp.allclose(rotations.as_euler("XYZ"), rot.quat2euler(jnp.array(quats), "XYZ"), atol=tol)


a = create_uniform_quats()
b = create_lock_quats()

print(a,b)
a.extend(b)
print(a)