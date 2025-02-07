"""Tests of the numeric models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
import numpy as np
import pytest

from lsy_models.dataclasses import QuadrotorState
from lsy_models.models import available_models, dynamic_numeric_from_symbolic, dynamics_numeric
from lsy_models.utils.constants import Constants

if TYPE_CHECKING:
    from numpy.typing import NDArray

# For all tests to pass, we need the same precsion in jax as in np
jax.config.update("jax_enable_x64", True)

N = 1000


def create_rnd_states(N: int = 1000) -> QuadrotorState:
    """Creates N random states."""
    pos = np.random.uniform(-5, 5, (N, 3))
    quat = np.random.uniform(
        -5, 5, (N, 4)
    )  # all rotation libraries should be normalizing automatically
    vel = np.random.uniform(-5, 5, (N, 3))
    angvel = np.random.uniform(-2, 2, (N, 3))
    forces_motor = np.random.uniform(0, 0.15, (N, 4))
    return QuadrotorState.create(pos, quat, vel, angvel, forces_motor)


def create_rnd_commands(N: int = 1000, dim: int = 4) -> NDArray[np.floating]:
    """Creates N random inputs with size dim."""
    return np.random.uniform(0, 0.15, (N, dim))


@pytest.mark.unit
@pytest.mark.parametrize("model", available_models)
@pytest.mark.parametrize("config", Constants.available_configs)
def test_symbolic2numeric(model: str, config: str):
    """Tests if casadi numeric prediction is the same as the numpy one."""
    states = create_rnd_states(N)
    commands = create_rnd_commands(N, 4)  # TODO make dependent on model

    f_numeric = dynamics_numeric(model, config)
    f_symbolic2numeric = dynamic_numeric_from_symbolic(model, config)

    for i in range(N):  # casadi only supports non batched calls
        x_dot_numeric = f_numeric(QuadrotorState.get_entry(states, i), commands[i])
        x_dot_numeric = QuadrotorState.as_array(x_dot_numeric)

        X = np.concat(
            (states.pos[i], states.quat[i], states.vel[i], states.angvel[i], states.forces_motor[i])
        )
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
    states = create_rnd_states(N)
    commands = create_rnd_commands(N, 4)  # TODO make dependent on model

    f_numeric = dynamics_numeric(model, config)

    batched = f_numeric(states, commands)
    batched = QuadrotorState.as_array(batched)
    batched_1 = []  # testing with batch size 1 (has led to problems earlier)
    non_batched = []

    for i in range(N):
        state_i_2D = QuadrotorState.create(
            states.pos[None, i],
            states.quat[None, i],
            states.vel[None, i],
            states.angvel[None, i],
            states.forces_motor[None, i],
        )

        batched_1.append(QuadrotorState.as_array(f_numeric(state_i_2D, commands[None, i])))
        non_batched.append(
            QuadrotorState.as_array(f_numeric(QuadrotorState.get_entry(states, i), commands[i]))
        )

    batched_1 = np.vstack(batched_1)
    non_batched = np.array(non_batched)

    assert np.allclose(batched, batched_1), "Batching failed for batch size 1"
    assert np.allclose(batched, non_batched), "Non-batched and batched results are not the same"


@pytest.mark.unit
@pytest.mark.parametrize("model", available_models)
@pytest.mark.parametrize("config", Constants.available_configs)
def test_numeric_arrayAPI(model: str, config: str):
    """Tests is the functions are jitable and if the results are identical to the numpy ones."""
    states_np = create_rnd_states(N)
    commands_np = create_rnd_commands(N, 4)  # TODO make dependent on model

    states_jp = QuadrotorState.from_array(states_np, jp.array(QuadrotorState.as_array(states_np)))
    commands_jp = jp.array(commands_np)

    f_numeric = dynamics_numeric(model, config)
    f_jit_numeric = jax.jit(f_numeric)

    npresults = QuadrotorState.as_array(f_numeric(states_np, commands_np))
    jpresults = QuadrotorState.as_array(f_jit_numeric(states_jp, commands_jp))

    assert isinstance(npresults, np.ndarray), "Results are not numpy arrays"
    assert isinstance(jpresults, jp.ndarray), "Results are not jax arrays"
    assert np.allclose(npresults, jpresults), "numpy and jax results differ"


# TODO test if external wrench gets applied properly. But how to test it?
# -> maybe apply and predict based on mass how much higher the acceleration should be
# same for torque
@pytest.mark.unit
def test_external_wrench():
    assert True
