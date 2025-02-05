"""Tests of the numeric models."""

import casadi as cs
import jax.numpy as jp
import numpy as np
import pytest

import lsy_models.numeric
import lsy_models.symbolic

N = 1000
pos = np.random.uniform(-5, 5, (N,3))
vel = np.random.uniform(-5, 5, (N,3))
quat = np.random.uniform(-5, 5, (N,4))
quat = quat / np.expand_dims(np.linalg.norm(quat, axis=1), -1) # unit quaternions!
angvel = np.random.uniform(-2, 2, (N,3))
forces_motor = np.random.uniform(0, 0.15, (N,4))
forces_cmd = np.random.uniform(0, 0.15, (N,4))

@pytest.mark.unit
def test_symbolic2numeric():
    """Tests if casadi numeric prediction is the same as the numpy one."""
    for method in lsy_models.numeric.methods:
        f_numeric = lsy_models.numeric.model_dynamics("cf2x-", method)
        f_symbolic2numeric = lsy_models.symbolic.model_dynamics("cf2x-", method)

        for i in range(N): # casadi only supports non batched calls
            x_dot_numeric = f_numeric(pos[i], vel[i], quat[i], angvel[i], forces_motor[i], forces_cmd[i])
            x_dot_numeric = np.concat(x_dot_numeric)
            x_dot_symbolic2numeric = np.array(f_symbolic2numeric(pos[i], vel[i], quat[i], angvel[i], forces_motor[i], forces_cmd[i])).squeeze()
            assert np.allclose(x_dot_numeric, x_dot_symbolic2numeric)

@pytest.mark.unit
def test_numeric_batching():
    """Tests if batching works and if the results are identical to the non-batched version."""
    for method in lsy_models.numeric.methods:
        f_numeric = lsy_models.numeric.model_dynamics("cf2x-", method)
        
        batched = f_numeric(pos, vel, quat, angvel, forces_motor, forces_cmd)
        batched_1 = [] # testing with batch size 1 (has led to problems earlier)
        non_batched = []

        for i in range(N):
            batched_1.append(np.hstack(f_numeric(pos[None,i], vel[None,i], quat[None,i], angvel[None,i], forces_motor[None,i], forces_cmd[None,i])))
            non_batched.append(np.concat(f_numeric(pos[i], vel[i], quat[i], angvel[i], forces_motor[i], forces_cmd[i])))

        batched = np.hstack(batched)
        batched_1 = np.vstack(batched_1)
        non_batched = np.array(non_batched)

        assert np.allclose(batched, batched_1)
        assert np.allclose(batched, non_batched)


# TODO: test for numpy and jax
@pytest.mark.unit
def test_numeric_arrayAPI():
    assert True

# TODO test if external wrench gets applied properly
@pytest.mark.unit
def test_external_wrench():
    assert True

