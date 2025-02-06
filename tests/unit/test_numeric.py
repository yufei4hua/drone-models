"""Tests of the numeric models."""

import casadi as cs
import jax
import jax.numpy as jp
import numpy as np
import pytest

import lsy_models.models as models

# For all tests to pass, we need the same precsion in jax as in np
jax.config.update("jax_enable_x64", True)

N = 1000

def create_rnd_states_inputs(N=1000) -> np.ndarray: # TODO return type
    """TODO."""
    pos = np.random.uniform(-5, 5, (N,3))
    vel = np.random.uniform(-5, 5, (N,3))
    quat = np.random.uniform(-5, 5, (N,4)) # all rotation libraries should be normalizing automatically
    angvel = np.random.uniform(-2, 2, (N,3))
    forces_motor = np.random.uniform(0, 0.15, (N,4))
    forces_cmd = np.random.uniform(0, 0.15, (N,4))
    return pos, vel, quat, angvel, forces_motor, forces_cmd

@pytest.mark.unit
def test_symbolic2numeric():
    """Tests if casadi numeric prediction is the same as the numpy one."""
    pos, vel, quat, angvel, forces_motor, forces_cmd = create_rnd_states_inputs()

    for model in models.available_models:
        f_numeric = models.dynamics(model, "cf2x-")
        f_symbolic2numeric = models.dynamics(model, "cf2x-", symbolic=True)

        for i in range(N): # casadi only supports non batched calls
            x_dot_numeric = f_numeric(pos[i], vel[i], quat[i], angvel[i], forces_motor[i], forces_cmd[i])
            x_dot_numeric = np.concat(x_dot_numeric)
            
            X = np.concat((pos[i], quat[i], vel[i], angvel[i], forces_motor[i]))
            print(X)
            U = forces_cmd[i]
            x_dot_symbolic2numeric = np.array(f_symbolic2numeric(X, U)).squeeze()
            print(f"diff = {x_dot_numeric-x_dot_symbolic2numeric}")
            assert np.allclose(x_dot_numeric, x_dot_symbolic2numeric)

@pytest.mark.unit
def test_numeric_batching():
    """Tests if batching works and if the results are identical to the non-batched version."""
    pos, vel, quat, angvel, forces_motor, forces_cmd = create_rnd_states_inputs(N=N)
    
    for model in models.available_models:
        f_numeric = models.dynamics(model, "cf2x-")
        
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
    nppos, npvel, npquat, npangvel, npforces_motor, npforces_cmd = create_rnd_states_inputs(N=N)
    jppos, jpvel, jpquat  = jp.array(nppos), jp.array(npvel), jp.array(npquat), 
    jpangvel, jpforces_motor, jpforces_cmd = jp.array(npangvel), jp.array(npforces_motor), jp.array(npforces_cmd)

    for model in models.available_models:
        f_numeric = models.dynamics(model, "cf2x-")

        npresults = f_numeric(nppos, npvel, npquat, npangvel, npforces_motor, npforces_cmd)
        jpresults = f_numeric(jppos, jpvel, jpquat, jpangvel, jpforces_motor, jpforces_cmd)

        assert isinstance(npresults[0], np.ndarray)
        assert isinstance(jpresults[0], jp.ndarray)
        npresults = np.hstack(npresults)
        jpresults = jp.hstack(jpresults)
        assert np.allclose(npresults, jpresults)

# TODO test if external wrench gets applied properly
@pytest.mark.unit
def test_external_wrench():
    assert True


# TODO test if all possible configs work => maybe just restructure and add to method testing
@pytest.mark.unit
def test_configs():
    assert True


test_numeric_arrayAPI()