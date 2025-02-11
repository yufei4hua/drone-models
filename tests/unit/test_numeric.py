"""Tests of the numeric models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
import numpy as np
import pytest

from lsy_models.models import available_models, dynamic_numeric_from_symbolic, dynamics_numeric
from lsy_models.utils.constants import Constants

if TYPE_CHECKING:
    from numpy.typing import NDArray

# For all tests to pass, we need the same precsion in jax as in np
jax.config.update("jax_enable_x64", True)

N = 1000


def create_rnd_states(N: int = 1000) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Creates N random states."""
    pos = np.random.uniform(-5, 5, (N, 3))
    quat = np.random.uniform(
        -5, 5, (N, 4)
    )  # all rotation libraries should be normalizing automatically
    vel = np.random.uniform(-5, 5, (N, 3))
    angvel = np.random.uniform(-2, 2, (N, 3))
    forces_motor = np.random.uniform(0, 0.15, (N, 4))
    return pos, quat, vel, angvel, forces_motor


def create_rnd_commands(N: int = 1000, dim: int = 4) -> NDArray[np.floating]:
    """Creates N random inputs with size dim."""
    return np.random.uniform(0, 0.15, (N, dim))


@pytest.mark.unit
@pytest.mark.parametrize("model", available_models)
@pytest.mark.parametrize("config", Constants.available_configs)
def test_symbolic2numeric(model: str, config: str):
    """Tests if casadi numeric prediction is the same as the numpy one."""
    pos, quat, vel, angvel, forces_motor = create_rnd_states((N))
    commands = create_rnd_commands(N, 4)  # TODO make dependent on model

    f_numeric = dynamics_numeric(model, config)
    f_symbolic2numeric = dynamic_numeric_from_symbolic(model, config)

    for i in range(N):  # casadi only supports non batched calls
        x_dot_numeric = f_numeric(pos[i], quat[i], vel[i], angvel[i], forces_motor[i], commands[i])
        x_dot_numeric = np.concat(x_dot_numeric)

        X = np.concat((pos[i], quat[i], vel[i], angvel[i], forces_motor[i]))

        U = commands[i]
        x_dot_symbolic2numeric = np.array(f_symbolic2numeric(X, U)).squeeze()
        print(f"diff = {x_dot_numeric - x_dot_symbolic2numeric}")
        assert np.allclose(x_dot_numeric, x_dot_symbolic2numeric), (
            "Symbolic and numeric model have different output"
        )


@pytest.mark.unit
@pytest.mark.parametrize("model", available_models)
@pytest.mark.parametrize("config", Constants.available_configs)
def test_numeric_batching(model: str, config: str):
    """Tests if batching works and if the results are identical to the non-batched version."""
    pos, quat, vel, angvel, forces_motor = create_rnd_states((N))
    commands = create_rnd_commands(N, 4)  # TODO make dependent on model

    f_numeric = dynamics_numeric(model, config)

    batched = f_numeric(pos, quat, vel, angvel, forces_motor, commands)
    batched_1 = []  # testing with batch size 1 (has led to problems earlier)
    non_batched = []

    for i in range(N):
        batched_1.append(
            np.hstack(
                f_numeric(
                    pos[None, i],
                    quat[None, i],
                    vel[None, i],
                    angvel[None, i],
                    forces_motor[None, i],
                    commands[None, i],
                )
            )
        )
        non_batched.append(
            np.concat(f_numeric(pos[i], quat[i], vel[i], angvel[i], forces_motor[i], commands[i]))
        )

    batched = np.hstack(batched)
    batched_1 = np.vstack(batched_1)
    non_batched = np.array(non_batched)

    assert np.allclose(batched, batched_1), "Batching failed for batch size 1"
    assert np.allclose(batched, non_batched), "Non-batched and batched results are not the same"


@pytest.mark.unit
@pytest.mark.parametrize("model", available_models)
@pytest.mark.parametrize("config", Constants.available_configs)
def test_numeric_arrayAPI(model: str, config: str):
    """Tests is the functions are jitable and if the results are identical to the numpy ones."""
    nppos, npquat, npvel, npangvel, npforces_motor = create_rnd_states(N=N)
    npcommands = create_rnd_commands(N, 4)

    jppos, jpvel, jpquat = jp.array(nppos), jp.array(npvel), jp.array(npquat)
    jpangvel, jpforces_motor = (jp.array(npangvel), jp.array(npforces_motor))
    jpcommands = jp.array(npcommands)

    f_numeric = dynamics_numeric(model, config)
    f_jit_numeric = jax.jit(f_numeric)

    npresults = f_numeric(nppos, npquat, npvel, npangvel, npforces_motor, npcommands)
    jpresults = f_jit_numeric(jppos, jpquat, jpvel, jpangvel, jpforces_motor, jpcommands)

    assert isinstance(npresults[0], np.ndarray), "Results are not numpy arrays"
    assert isinstance(jpresults[0], jp.ndarray), "Results are not jax arrays"
    npresults = np.hstack(npresults)
    jpresults = jp.hstack(jpresults)
    assert np.allclose(npresults, jpresults), "numpy and jax results differ"


# TODO test if external wrench gets applied properly. But how to test it?
# -> maybe apply and predict based on mass how much higher the acceleration should be
# same for torque
@pytest.mark.unit
def test_external_wrench():
    assert True
