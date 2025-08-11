"""Tests of the numeric models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_compat.numpy as np
import array_api_strict as xp
import jax
import jax.numpy as jp

# import numpy as np
import pytest

from drone_models.models import available_models, dynamic_numeric_from_symbolic, dynamics_numeric
from drone_models.utils.constants import Constants

if TYPE_CHECKING:
    from array_api_strict import Array

# For all tests to pass, we need the same precsion in jax as in np
jax.config.update("jax_enable_x64", True)

N = 100


def create_rnd_states(N: int = 1000) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Creates N random states."""
    pos = xp.asarray(np.random.uniform(-5, 5, (N, 3)))
    quat = xp.asarray(np.random.uniform(-1, 1, (N, 4)))  # Libraries normalize automatically
    vel = xp.asarray(np.random.uniform(-5, 5, (N, 3)))
    ang_vel = xp.asarray(np.random.uniform(-2, 2, (N, 3)))
    forces_motor = xp.asarray(np.random.uniform(0, 0.2, (N, 4)))
    forces_dist = xp.asarray(np.random.uniform(-0.2, 0.2, (N, 3)))
    torques_dist = xp.asarray(np.random.uniform(-0.05, 0.05, (N, 3)))
    return pos, quat, vel, ang_vel, forces_motor, forces_dist, torques_dist


def create_rnd_commands(N: int = 1000, dim: int = 4) -> Array:
    """Creates N random inputs with size dim."""
    return xp.asarray(np.random.uniform(0, 0.2, (N, dim)))


@pytest.mark.unit
@pytest.mark.parametrize("model", available_models.keys())
@pytest.mark.parametrize("config", Constants.available_configs)
def test_symbolic2numeric(model: str, config: str):
    """Tests if casadi numeric prediction is the same as the numpy one."""
    pos, quat, vel, ang_vel, forces_motor, _, _ = create_rnd_states((N))
    if model == "fitted_DI_rpyt":
        forces_motor = None
    commands = create_rnd_commands(N, 4)  # TODO make dependent on model

    f_numeric = dynamics_numeric(model, config, xp)
    f_symbolic2numeric = dynamic_numeric_from_symbolic(model, config)

    for i in range(N):  # casadi only supports non batched calls
        x_dot_numeric = f_numeric(
            pos[i, ...],
            quat[i, ...],
            vel[i, ...],
            ang_vel[i, ...],
            commands[i, ...],
            forces_motor=forces_motor[i, ...] if forces_motor is not None else None,
        )
        print(x_dot_numeric)
        if forces_motor is not None:
            x_dot_numeric = xp.concat(x_dot_numeric)
        else:
            x_dot_numeric = xp.concat(x_dot_numeric[:-1])

        if forces_motor is not None:
            X = xp.concat(
                (pos[i, ...], quat[i, ...], vel[i, ...], ang_vel[i, ...], forces_motor[i, ...])
            )
        else:
            X = xp.concat((pos[i, ...], quat[i, ...], vel[i, ...], ang_vel[i, ...]))

        U = commands[i, ...]
        print(X.shape, U.shape)
        x_dot_symbolic2numeric = xp.asarray(f_symbolic2numeric(X._array, U._array))
        x_dot_symbolic2numeric = xp.squeeze(x_dot_symbolic2numeric, axis=-1)
        print(f"i={i}, diff={x_dot_numeric - x_dot_symbolic2numeric}")
        print(f"{x_dot_numeric=}, {x_dot_symbolic2numeric=}")
        assert np.allclose(x_dot_numeric, x_dot_symbolic2numeric), (
            "Symbolic and numeric model have different output"
        )


# @pytest.mark.unit
# @pytest.mark.parametrize("model", available_models.keys())
# @pytest.mark.parametrize("config", Constants.available_configs)
# def test_numeric_batching(model: str, config: str):
#     """Tests if batching works and if the results are identical to the non-batched version."""
#     pos, quat, vel, ang_vel, forces_motor, _, _ = create_rnd_states(N)
#     commands = create_rnd_commands(N, 4)  # TODO make dependent on model

#     f_numeric = dynamics_numeric(model, config)
#     if model == "fitted_DI_rpyt":
#         forces_motor = None

#     batched = f_numeric(pos, quat, vel, ang_vel, commands, forces_motor=forces_motor)
#     batched_1 = []  # testing with batch size 1 (has led to problems earlier)
#     non_batched = []

#     for i in range(N):
#         if forces_motor is not None:
#             pos_bat, quat_bat, vel_bat, ang_vel_bat, forces_motor_bat = f_numeric(
#                 pos[None, i],
#                 quat[None, i],
#                 vel[None, i],
#                 ang_vel[None, i],
#                 commands[None, i],
#                 forces_motor=forces_motor[None, i],
#             )
#             batched_1.append(np.hstack((pos_bat, quat_bat, vel_bat, ang_vel_bat, forces_motor_bat)))

#             pos_non_bat, quat_non_bat, vel_non_bat, ang_vel_non_bat, forces_motor_non_bat = (
#                 f_numeric(
#                     pos[i], quat[i], vel[i], ang_vel[i], commands[i], forces_motor=forces_motor[i]
#                 )
#             )
#             non_batched.append(
#                 np.hstack(
#                     (pos_non_bat, quat_non_bat, vel_non_bat, ang_vel_non_bat, forces_motor_non_bat)
#                 )
#             )
#         else:
#             pos_bat, quat_bat, vel_bat, ang_vel_bat, forces_motor_bat = f_numeric(
#                 pos[None, i], quat[None, i], vel[None, i], ang_vel[None, i], commands[None, i]
#             )
#             batched_1.append(np.hstack((pos_bat, quat_bat, vel_bat, ang_vel_bat)))

#             pos_non_bat, quat_non_bat, vel_non_bat, ang_vel_non_bat, forces_motor_non_bat = (
#                 f_numeric(pos[i], quat[i], vel[i], ang_vel[i], commands[i])
#             )
#             non_batched.append(np.hstack((pos_non_bat, quat_non_bat, vel_non_bat, ang_vel_non_bat)))

#     if batched[-1] is not None:
#         batched = np.hstack(batched)
#     else:
#         batched = np.hstack(batched[:-1])
#     batched_1 = np.vstack(batched_1)
#     non_batched = np.array(non_batched)

#     assert np.allclose(batched, batched_1), "Batching failed for batch size 1"
#     assert np.allclose(batched, non_batched), "Non-batched and batched results are not the same"


@pytest.mark.unit
@pytest.mark.parametrize("model", available_models.keys())
@pytest.mark.parametrize("config", Constants.available_configs)
def test_numeric_jit(model: str, config: str):
    """Tests is the models are jitable and if the results are identical to the numpy ones."""
    nppos, npquat, npvel, npang_vel, npforces_motor, _, _ = create_rnd_states(N=N)
    if model == "fitted_DI_rpyt":
        npforces_motor = None
    npcommands = create_rnd_commands(N, 4)

    jppos, jpquat = jp.array(nppos._array), jp.array(npquat._array)
    jpvel, jpang_vel = jp.array(npvel._array), jp.array(npang_vel._array)
    if model == "fitted_DI_rpyt":
        jpforces_motor = None
    else:
        jpforces_motor = jp.array(npforces_motor._array)
    jpcommands = jp.array(npcommands._array)

    f_numeric = dynamics_numeric(model, config, xp)
    f_jit_numeric = jax.jit(dynamics_numeric(model, config, jp))

    npresults = f_numeric(nppos, npquat, npvel, npang_vel, npcommands, forces_motor=npforces_motor)
    jpresults = f_jit_numeric(
        jppos, jpquat, jpvel, jpang_vel, jpcommands, forces_motor=jpforces_motor
    )

    # assert isinstance(npresults[0], np.ndarray), "Results are not numpy arrays"
    assert isinstance(jpresults[0], jp.ndarray), "Results are not jax arrays"
    if npresults[-1] is not None:
        npresults = np.hstack(npresults)
    else:
        npresults = np.hstack(npresults[:-1])
    if jpresults[-1] is not None:
        jpresults = np.hstack(jpresults)
    else:
        jpresults = np.hstack(jpresults[:-1])
    assert np.allclose(npresults, jpresults), "numpy and jax results differ"


# # TODO test if external wrench gets applied properly. But how to test it?
# # -> maybe apply and predict based on mass how much higher the acceleration should be
# # same for torque
# @pytest.mark.unit
# def test_external_wrench():
#     assert True
