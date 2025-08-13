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
from drone_models.utils.constants import Constants, available_configs

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
@pytest.mark.parametrize("drone_name", available_configs)
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
@pytest.mark.parametrize("drone_name", available_configs)
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
@pytest.mark.parametrize("drone_name", available_configs)
def test_model_single_external_wrench(model_name: str, model: Callable, drone_name: str):
    pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t = create_rnd_states()
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(dim=4)  # TODO make dependent on model
    dpos, dquat, dvel, dang_vel, drotor_vel = model(
        pos,
        quat,
        vel,
        ang_vel,
        cmd,
        Constants.from_config(drone_name, xp),
        rotor_vel=rotor_vel,
        dist_f=dist_f,
        dist_t=dist_t,
    )
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip([dpos, dquat, dvel, dang_vel], [pos, quat, vel, ang_vel], strict=True):
        assert isinstance(dx, type(x))
        assert xp_device(dx) == xp_device(x)
        assert dx.shape == x.shape


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_name", available_configs)
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
@pytest.mark.parametrize("drone_name", available_configs)
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
@pytest.mark.parametrize("drone_name", available_configs)
def test_model_batched_external_wrench(model_name: str, model: Callable, drone_name: str):
    batch_shape = (10,)
    pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model
    dpos, dquat, dvel, dang_vel, drotor_vel = model(
        pos,
        quat,
        vel,
        ang_vel,
        cmd,
        Constants.from_config(drone_name, xp),
        rotor_vel=rotor_vel,
        dist_f=dist_f,
        dist_t=dist_t,
    )
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip([dpos, dquat, dvel, dang_vel], [pos, quat, vel, ang_vel], strict=True):
        assert isinstance(dx, type(x))
        assert xp_device(dx) == xp_device(x)
        assert dx.shape == x.shape


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("config", available_configs)
def test_symbolic2numeric_no_external_wrench(model_name: str, model: Callable, config: str):
    batch_shape = (10,)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model

    # Create numeric model from symbolic model
    dynamics_symbolic = getattr(sys.modules[model.__module__], "dynamics_symbolic")
    X_dot, X, U, _ = dynamics_symbolic(
        Constants.from_config(config, np), calc_rotor_vel=True if rotor_vel is not None else False
    )
    model_symbolic2numeric = cs.Function(model_name, [X, U], [X_dot])

    for i in np.ndindex(np.shape(pos)[:-1]):  # casadi only supports non batched calls
        x_dot = model(
            pos[i + (...,)],
            quat[i + (...,)],
            vel[i + (...,)],
            ang_vel[i + (...,)],
            cmd[i + (...,)],
            Constants.from_config(config, xp),
            rotor_vel=rotor_vel[i + (...,)] if rotor_vel is not None else None,
        )
        x_dot = xp.concat([x for x in x_dot if x is not None], axis=-1)

        if rotor_vel is not None:
            X = xp.concat(
                (
                    pos[i + (...,)],
                    quat[i + (...,)],
                    vel[i + (...,)],
                    ang_vel[i + (...,)],
                    rotor_vel[i + (...,)],
                ),
                axis=-1,
            )
        else:
            X = xp.concat(
                (pos[i + (...,)], quat[i + (...,)], vel[i + (...,)], ang_vel[i + (...,)]), axis=-1
            )

        U = cmd[i + (...,)]
        x_dot_symbolic2numeric = xp.asarray(model_symbolic2numeric(np.asarray(X), np.asarray(U)))
        x_dot_symbolic2numeric = xp.squeeze(x_dot_symbolic2numeric, axis=-1)
        assert np.allclose(x_dot, x_dot_symbolic2numeric), (
            "Symbolic and numeric model have different output"
        )


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("config", available_configs)
def test_symbolic2numeric_external_wrench(model_name: str, model: Callable, config: str):
    batch_shape = (10,)
    pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model

    # Create numeric model from symbolic model
    dynamics_symbolic = getattr(sys.modules[model.__module__], "dynamics_symbolic")
    X_dot, X, U, _ = dynamics_symbolic(
        Constants.from_config(config, np),
        calc_rotor_vel=True if rotor_vel is not None else False,
        calc_dist_f=True,
        calc_dist_t=True,
    )
    model_symbolic2numeric = cs.Function(model_name, [X, U], [X_dot])

    for i in np.ndindex(np.shape(pos)[:-1]):  # casadi only supports non batched calls
        x_dot = model(
            pos[i + (...,)],
            quat[i + (...,)],
            vel[i + (...,)],
            ang_vel[i + (...,)],
            cmd[i + (...,)],
            Constants.from_config(config, xp),
            rotor_vel=rotor_vel[i + (...,)] if rotor_vel is not None else None,
            dist_f=dist_f[i + (...,)],
            dist_t=dist_t[i + (...,)],
        )
        x_dot = xp.concat([x for x in x_dot if x is not None], axis=-1)

        if rotor_vel is not None:
            X = xp.concat(
                (
                    pos[i + (...,)],
                    quat[i + (...,)],
                    vel[i + (...,)],
                    ang_vel[i + (...,)],
                    rotor_vel[i + (...,)],
                    dist_f[i + (...,)],
                    dist_t[i + (...,)],
                ),
                axis=-1,
            )
        else:
            X = xp.concat(
                (
                    pos[i + (...,)],
                    quat[i + (...,)],
                    vel[i + (...,)],
                    ang_vel[i + (...,)],
                    dist_f[i + (...,)],
                    dist_t[i + (...,)],
                ),
                axis=-1,
            )

        U = cmd[i + (...,)]
        x_dot_symbolic2numeric = xp.asarray(model_symbolic2numeric(np.asarray(X), np.asarray(U)))
        x_dot_symbolic2numeric = xp.squeeze(x_dot_symbolic2numeric, axis=-1)
        assert np.allclose(x_dot, x_dot_symbolic2numeric), (
            "Symbolic and numeric model have different output"
        )


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("config", available_configs)
def test_numeric_batching(model_name: str, model: Callable, config: str):
    """Tests if batching works and if the results are identical to the non-batched version."""
    batch_shape = (10,)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model
    constants = Constants.from_config(config, xp)

    x_dot_batched = model(pos, quat, vel, ang_vel, cmd, constants, rotor_vel=rotor_vel)
    x_dot_batched = xp.concat([x for x in x_dot_batched if x is not None], axis=-1)

    for i in np.ndindex(np.shape(pos)[:-1]):
        # Batch size 1
        x_dot_batched_1 = model(
            pos[(None,) + i + (...,)],
            quat[(None,) + i + (...,)],
            vel[(None,) + i + (...,)],
            ang_vel[(None,) + i + (...,)],
            cmd[(None,) + i + (...,)],
            constants,
            rotor_vel=rotor_vel[(None,) + i + (...,)] if rotor_vel is not None else None,
        )
        x_dot_batched_1 = xp.concat([x for x in x_dot_batched_1 if x is not None], axis=-1)

        # No batching
        x_dot_non_batched = model(
            pos[i + (...,)],
            quat[i + (...,)],
            vel[i + (...,)],
            ang_vel[i + (...,)],
            cmd[i + (...,)],
            constants,
            rotor_vel=rotor_vel[i + (...,)] if rotor_vel is not None else None,
        )
        x_dot_non_batched = xp.concat([x for x in x_dot_non_batched if x is not None], axis=-1)

        assert np.allclose(x_dot_batched[i + (...,)], x_dot_batched_1[0, ...], atol=2e-8), (
            "Batching failed for batch size 1"
        )
        assert np.allclose(x_dot_batched[i + (...,)], x_dot_non_batched, atol=2e-8), (
            "Non-batched and batched results are not the same"
        )


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("config", available_configs)
def test_numeric_jit(model_name: str, model: Callable, config: str):
    """Tests if the models are jitable and if the results are identical to the array API ones."""
    batch_shape = (10,)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model

    xp_dot = model(
        pos,
        quat,
        vel,
        ang_vel,
        cmd,
        Constants.from_config(config, xp),
        rotor_vel=rotor_vel if rotor_vel is not None else None,
    )

    jppos, jpquat = jp.asarray(np.asarray(pos)), jp.asarray(np.asarray(quat))
    jpvel, jpang_vel = jp.asarray(np.asarray(vel)), jp.asarray(np.asarray(ang_vel))
    if rotor_vel is None:
        jprotor_vel = None
    else:
        jprotor_vel = jp.asarray(np.asarray(rotor_vel))
    jpcmd = jp.asarray(np.asarray(cmd))

    model_jit = jax.jit(model)

    jp_dot = model_jit(
        jppos,
        jpquat,
        jpvel,
        jpang_vel,
        jpcmd,
        Constants.from_config(config, jp),
        rotor_vel=jprotor_vel if jprotor_vel is not None else None,
    )

    assert isinstance(jp_dot[0], jp.ndarray), "Results are not jax arrays"

    xp_dot = xp.concat([x for x in xp_dot if x is not None], axis=-1)
    jp_dot = jp.concat([x for x in jp_dot if x is not None], axis=-1)

    assert np.allclose(xp_dot, jp_dot), "numpy and jax results differ"


# TODO test if external wrench gets applied properly. But how to test it?
# -> maybe apply and predict based on mass how much higher the acceleration should be
# same for torque
@pytest.mark.unit
def test_external_wrench():
    assert True
