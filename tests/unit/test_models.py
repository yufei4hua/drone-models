"""Tests of the numeric models."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable

import array_api_compat.numpy as np
import array_api_strict as xp
import casadi as cs
import jax
import jax.numpy as jp
import pytest
from array_api_compat import device as xp_device

from drone_models.models import available_models, model_features
from drone_models.utils.constants import Constants

if TYPE_CHECKING:
    from array_api_typing import Array

# For all tests to pass, we need the same precsion in jax as in np
jax.config.update("jax_enable_x64", True)


def create_rnd_states(
    shape: tuple[int, ...] = (),
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Creates N random states."""
    pos = xp.asarray(np.random.uniform(-5, 5, shape + (3,)))
    quat = xp.asarray(np.random.uniform(-1, 1, shape + (4,)))  # Libraries normalize automatically
    vel = xp.asarray(np.random.uniform(-5, 5, shape + (3,)))
    ang_vel = xp.asarray(np.random.uniform(-2, 2, shape + (3,)))
    rotor_vel = xp.asarray(np.random.uniform(0, 0.2, shape + (4,)))
    forces_dist = xp.asarray(np.random.uniform(-0.2, 0.2, shape + (3,)))
    torques_dist = xp.asarray(np.random.uniform(-0.05, 0.05, shape + (3,)))
    return pos, quat, vel, ang_vel, rotor_vel, forces_dist, torques_dist


def create_rnd_commands(shape: tuple[int, ...] = (), dim: int = 4) -> Array:
    """Creates N random inputs with size dim."""
    return xp.asarray(np.random.uniform(0, 0.2, shape + (dim,)))


def skip_models_without_features(model: Callable, features: list[str]):
    """Skip the model if it does not have the required features."""
    for feature in features:
        if not model_features(model)[feature]:
            pytest.skip(f"Model {model.__name__} does not have the feature '{feature}'.")


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
def test_model_features(model_name: str, model: Callable):
    """Tests if the model features are correctly set."""
    assert hasattr(model, "__drone_model_features__"), (
        f"Model function {model_name} does not have __drone_model_features__ attribute"
    )
    features = model_features(model)
    assert isinstance(features, dict), (
        f"model features should be a dict, got {type(features)} for {model_name}"
    )
    assert "rotor_dynamics" in features, (
        f"model features should contain 'rotor_dynamics' key for {model_name}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_name", Constants.available_configs)
def test_model_single_no_rotor_dynamics(model_name: str, model: Callable, drone_name: str):
    pos, quat, vel, ang_vel, _, _, _ = create_rnd_states()
    cmd = create_rnd_commands(dim=4)  # TODO make dependent on model
    if model_features(model)["rotor_dynamics"]:
        with pytest.warns(UserWarning, match="Rotor velocity is not provided"):
            dpos, dquat, dvel, dang_vel, drotor_vel = model(
                pos, quat, vel, ang_vel, cmd, Constants.from_config(drone_name, xp), rotor_vel=None
            )
    else:
        dpos, dquat, dvel, dang_vel, drotor_vel = model(
            pos, quat, vel, ang_vel, cmd, Constants.from_config(drone_name, xp), rotor_vel=None
        )
    assert drotor_vel is None, "Model should not return rotor velocities without rotor_vel input"
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip([dpos, dquat, dvel, dang_vel], [pos, quat, vel, ang_vel], strict=True):
        assert isinstance(dx, type(x))
        assert xp_device(dx) == xp_device(x)
        assert dx.shape == x.shape


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_name", Constants.available_configs)
def test_model_single_rotor_dynamics(model_name: str, model: Callable, drone_name: str):
    skip_models_without_features(model, ["rotor_dynamics"])

    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states()
    cmd = create_rnd_commands(dim=4)  # TODO make dependent on model
    dpos, dquat, dvel, dang_vel, drotor_vel = model(
        pos, quat, vel, ang_vel, cmd, Constants.from_config(drone_name, xp), rotor_vel=rotor_vel
    )
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip(
        [dpos, dquat, dvel, dang_vel, drotor_vel], [pos, quat, vel, ang_vel, rotor_vel], strict=True
    ):
        assert isinstance(dx, type(x))
        assert xp_device(dx) == xp_device(x)
        assert dx.shape == x.shape


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_name", Constants.available_configs)
def test_model_batched_no_rotor_dynamics(model_name: str, model: Callable, drone_name: str):
    batch_shape = (10,)
    pos, quat, vel, ang_vel, _, _, _ = create_rnd_states(batch_shape)
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model
    if model_features(model)["rotor_dynamics"]:
        with pytest.warns(UserWarning, match="Rotor velocity is not provided"):
            dpos, dquat, dvel, dang_vel, drotor_vel = model(
                pos, quat, vel, ang_vel, cmd, Constants.from_config(drone_name, xp), rotor_vel=None
            )
    else:
        dpos, dquat, dvel, dang_vel, drotor_vel = model(
            pos, quat, vel, ang_vel, cmd, Constants.from_config(drone_name, xp), rotor_vel=None
        )
    assert drotor_vel is None, "Model should not return rotor velocities without rotor_vel input"
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip([dpos, dquat, dvel, dang_vel], [pos, quat, vel, ang_vel], strict=True):
        assert isinstance(dx, type(x))
        assert xp_device(dx) == xp_device(x)
        assert dx.shape == x.shape


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_name", Constants.available_configs)
def test_model_batched_rotor_dynamics(model_name: str, model: Callable, drone_name: str):
    skip_models_without_features(model, ["rotor_dynamics"])

    batch_shape = (10,)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model
    dpos, dquat, dvel, dang_vel, drotor_vel = model(
        pos, quat, vel, ang_vel, cmd, Constants.from_config(drone_name, xp), rotor_vel=rotor_vel
    )
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip(
        [dpos, dquat, dvel, dang_vel, drotor_vel], [pos, quat, vel, ang_vel, rotor_vel], strict=True
    ):
        assert isinstance(dx, type(x))
        assert xp_device(dx) == xp_device(x)
        assert dx.shape == x.shape


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("config", Constants.available_configs)
def test_symbolic2numeric(model_name: str, model: Callable, config: str):
    batch_shape = (10,)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model

    # Create numeric model from symbolic model
    dynamics_symbolic = getattr(sys.modules[model.__module__], "dynamics_symbolic")
    X_dot, X, U, _ = dynamics_symbolic(Constants.from_config(config, np))
    model_symbolic2numeric = cs.Function(model_name, [X, U], [X_dot])

    for i in np.ndindex(np.shape(pos)[:-1]):  # casadi only supports non batched calls
        print(f"{i=}, {np.shape(pos)=}, {pos[i+(slice(None),)]=}")  #
        x_dot = model(
            pos[i + (slice(None),)],
            quat[i + (slice(None),)],
            vel[i + (slice(None),)],
            ang_vel[i + (slice(None),)],
            cmd[i + (slice(None),)],
            Constants.from_config(config, xp),
            rotor_vel=rotor_vel[i + (slice(None),)] if rotor_vel is not None else None,
        )
        x_dot = xp.concat([x for x in x_dot if x is not None])

        if rotor_vel is not None:
            X = xp.concat(
                (
                    pos[i + (slice(None),)],
                    quat[i + (slice(None),)],
                    vel[i + (slice(None),)],
                    ang_vel[i + (slice(None),)],
                    rotor_vel[i + (slice(None),)],
                )
            )
        else:
            X = xp.concat(
                (
                    pos[i + (slice(None),)],
                    quat[i + (slice(None),)],
                    vel[i + (slice(None),)],
                    ang_vel[i + (slice(None),)],
                )
            )

        U = cmd[i + (slice(None),)]
        x_dot_symbolic2numeric = xp.asarray(model_symbolic2numeric(X._array, U._array))
        x_dot_symbolic2numeric = xp.squeeze(x_dot_symbolic2numeric, axis=-1)
        assert np.allclose(x_dot, x_dot_symbolic2numeric), (
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
