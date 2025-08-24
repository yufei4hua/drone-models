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

from drone_models import available_models, model_features
from drone_models.core import parametrize
from drone_models.drones import available_drones

if TYPE_CHECKING:
    from array_api_typing import Array

# For all tests to pass, we need the same precsion in jax as in np
jax.config.update("jax_enable_x64", True)


def create_rnd_states(shape: tuple[int, ...] = ()) -> tuple[Array, Array, Array, Array]:
    x = np.random.randn(*shape, 3 + 4 + 3 + 3 + 4 + 3 + 3)
    pos = xp.asarray(x[..., :3])
    quat = xp.asarray(x[..., 3:7])
    vel = xp.asarray(x[..., 7:10])
    ang_vel = xp.asarray(x[..., 10:13])
    rotor_vel = xp.abs(xp.asarray(x[..., 13:17]))  # Rotor velocities must be positive
    dist_f = xp.asarray(x[..., 17:20])
    dist_t = xp.asarray(x[..., 20:23])
    return pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t


def create_rnd_commands(shape: tuple[int, ...] = (), dim: int = 4) -> Array:
    """Creates N random inputs with size dim."""
    return xp.abs(xp.asarray(np.random.randn(*shape, dim)))  # Motor forces must be positive


def assert_array_meta(x: Array | None, y: Array | None):
    """Assert the output is on the correct device, has the correct type and shape."""
    if x is None and y is None:
        return
    assert isinstance(x, type(y)), f"Output type {type(x)} does not match expected {type(y)}"
    assert xp_device(x) == xp_device(y), (
        f"Output device {xp_device(x)} does not match expected {xp_device(y)}"
    )
    assert x.shape == y.shape, f"Output shape {x.shape} does not match expected {y.shape}"
    assert np.all(np.isnan(x) == np.isnan(y)), "Derivative of non-nan values are NaN"


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
@pytest.mark.parametrize("drone_type", available_drones)
def test_model_single_no_rotor_dynamics(model_name: str, model: Callable, drone_type: str):
    pos, quat, vel, ang_vel, _, _, _ = create_rnd_states()
    model = parametrize(model, drone_type)
    cmd = create_rnd_commands(dim=4)  # TODO make dependent on model
    if model_features(model)["rotor_dynamics"]:
        with pytest.warns(UserWarning, match="Rotor velocity not provided"):
            dpos, dquat, dvel, dang_vel, drotor_vel = model(pos, quat, vel, ang_vel, cmd)
    else:
        dpos, dquat, dvel, dang_vel, drotor_vel = model(pos, quat, vel, ang_vel, cmd)
    assert drotor_vel is None, "Model should not return rotor velocities without rotor_vel input"
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip([dpos, dquat, dvel, dang_vel], [pos, quat, vel, ang_vel], strict=True):
        assert_array_meta(dx, x)


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_model_single_rotor_dynamics(model_name: str, model: Callable, drone_type: str):
    skip_models_without_features(model, ["rotor_dynamics"])
    model = parametrize(model, drone_type)

    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states()
    cmd = create_rnd_commands(dim=4)
    dpos, dquat, dvel, dang_vel, drotor_vel = model(pos, quat, vel, ang_vel, cmd, rotor_vel)
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip(
        [dpos, dquat, dvel, dang_vel, drotor_vel], [pos, quat, vel, ang_vel, rotor_vel], strict=True
    ):
        assert_array_meta(dx, x)


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_model_single_external_wrench(model_name: str, model: Callable, drone_type: str):
    model = parametrize(model, drone_type)
    pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t = create_rnd_states()
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(dim=4)  # TODO make dependent on model
    dpos, dquat, dvel, dang_vel, drotor_vel = model(
        pos, quat, vel, ang_vel, cmd, rotor_vel, dist_f, dist_t
    )
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip([dpos, dquat, dvel, dang_vel], [pos, quat, vel, ang_vel], strict=True):
        assert_array_meta(dx, x)


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_model_batched_no_rotor_dynamics(model_name: str, model: Callable, drone_type: str):
    model = parametrize(model, drone_type)
    batch_shape = (10, 5)
    pos, quat, vel, ang_vel, _, _, _ = create_rnd_states(batch_shape)
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model
    if model_features(model)["rotor_dynamics"]:
        with pytest.warns(UserWarning, match="Rotor velocity not provided"):
            dpos, dquat, dvel, dang_vel, drotor_vel = model(pos, quat, vel, ang_vel, cmd, None)
    else:
        dpos, dquat, dvel, dang_vel, drotor_vel = model(pos, quat, vel, ang_vel, cmd, None)
    assert drotor_vel is None, "Model should not return rotor velocities without rotor_vel input"
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip([dpos, dquat, dvel, dang_vel], [pos, quat, vel, ang_vel], strict=True):
        assert_array_meta(dx, x)


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_model_batched_rotor_dynamics(model_name: str, model: Callable, drone_type: str):
    skip_models_without_features(model, ["rotor_dynamics"])
    model = parametrize(model, drone_type)

    batch_shape = (10, 5)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    cmd = create_rnd_commands(batch_shape, dim=4)
    dpos, dquat, dvel, dang_vel, drotor_vel = model(pos, quat, vel, ang_vel, cmd, rotor_vel)
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip(
        [dpos, dquat, dvel, dang_vel, drotor_vel], [pos, quat, vel, ang_vel, rotor_vel], strict=True
    ):
        assert_array_meta(dx, x)


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_model_batched_external_wrench(model_name: str, model: Callable, drone_type: str):
    model = parametrize(model, drone_type)

    batch_shape = (10, 5)
    pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model
    dpos, dquat, dvel, dang_vel, drotor_vel = model(
        pos, quat, vel, ang_vel, cmd, rotor_vel, dist_f=dist_f, dist_t=dist_t
    )
    # Check if the output is on the correct device, has the correct type and shape
    for dx, x in zip(
        [dpos, dquat, dvel, dang_vel, drotor_vel], [pos, quat, vel, ang_vel, rotor_vel], strict=True
    ):
        assert_array_meta(dx, x)


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_symbolic_dynamics(model_name: str, model: Callable, drone_type: str):
    symbolic_dynamics = getattr(sys.modules[model.__module__], "symbolic_dynamics")
    symbolic_dynamics = parametrize(symbolic_dynamics, drone_type)
    model = parametrize(model, drone_type)

    batch_shape = (10, 5)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)

    # Create symbolic model from dynamics
    X_dot, X, U, _ = symbolic_dynamics(model_rotor_vel=rotor_vel is not None)
    model_symbolic2numeric = cs.Function(model_name, [X, U], [X_dot])

    for i in np.ndindex(np.shape(pos)[:-1]):  # casadi only supports non batched calls
        x_dot = model(
            pos[i + (...,)],
            quat[i + (...,)],
            vel[i + (...,)],
            ang_vel[i + (...,)],
            cmd[i + (...,)],
            rotor_vel=rotor_vel[i + (...,)] if rotor_vel is not None else None,
        )
        x_dot = xp.concat([x for x in x_dot if x is not None], axis=-1)
        X = xp.concat(
            [x[i + (...,)] for x in [pos, quat, vel, ang_vel, rotor_vel] if x is not None], axis=-1
        )
        U = cmd[i + (...,)]
        x_dot_symbolic2numeric = xp.asarray(model_symbolic2numeric(np.asarray(X), np.asarray(U)))
        x_dot_symbolic2numeric = xp.squeeze(x_dot_symbolic2numeric, axis=-1)
        assert np.allclose(x_dot, x_dot_symbolic2numeric), (
            "Symbolic and numeric model have different output"
        )


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_symbolic_dynamics_external_wrench(model_name: str, model: Callable, drone_type: str):
    symbolic_dynamics = getattr(sys.modules[model.__module__], "symbolic_dynamics")
    symbolic_dynamics = parametrize(symbolic_dynamics, drone_type)
    model = parametrize(model, drone_type)

    batch_shape = (10, 5)
    pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model

    # Create numeric model from symbolic model
    X_dot, X, U, _ = symbolic_dynamics(
        model_rotor_vel=rotor_vel is not None, model_dist_f=True, model_dist_t=True
    )
    model_symbolic2numeric = cs.Function(model_name, [X, U], [X_dot])

    for i in np.ndindex(np.shape(pos)[:-1]):  # casadi only supports non batched calls
        x_dot = model(
            pos[i + (...,)],
            quat[i + (...,)],
            vel[i + (...,)],
            ang_vel[i + (...,)],
            cmd[i + (...,)],
            rotor_vel=rotor_vel[i + (...,)] if rotor_vel is not None else None,
            dist_f=dist_f[i + (...,)],
            dist_t=dist_t[i + (...,)],
        )
        x_dot = xp.concat([x for x in x_dot if x is not None], axis=-1)
        X = xp.concat(
            [
                x[i + (...,)]
                for x in [pos, quat, vel, ang_vel, rotor_vel, dist_f, dist_t]
                if x is not None
            ],
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
@pytest.mark.parametrize("drone_type", available_drones)
def test_compare_batched_non_batched(model_name: str, model: Callable, drone_type: str):
    """Tests if batching works and if the results are identical to the non-batched version."""
    model = parametrize(model, drone_type)

    batch_shape = (10, 5)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model

    x_dot_batched = model(pos, quat, vel, ang_vel, cmd, rotor_vel)
    x_dot_batched = xp.concat([x for x in x_dot_batched if x is not None], axis=-1)

    for i in np.ndindex(np.shape(pos)[:-1]):
        x_dot_non_batched = model(
            pos[i + (...,)],
            quat[i + (...,)],
            vel[i + (...,)],
            ang_vel[i + (...,)],
            cmd[i + (...,)],
            rotor_vel[i + (...,)] if rotor_vel is not None else None,
        )
        x_dot_non_batched = xp.concat([x for x in x_dot_non_batched if x is not None], axis=-1)

        assert np.allclose(x_dot_batched[i + (...,)], x_dot_non_batched, atol=1e-5), (
            "Non-batched and batched results are not the same"
        )


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_numeric_jit(model_name: str, model: Callable, drone_type: str):
    """Tests if the models are jitable and if the results are identical to the array API ones."""
    model = parametrize(model, drone_type)

    batch_shape = (10, 5)
    pos, quat, vel, ang_vel, rotor_vel, _, _ = create_rnd_states(batch_shape)
    if not model_features(model)["rotor_dynamics"]:
        rotor_vel = None
    cmd = create_rnd_commands(batch_shape, dim=4)  # TODO make dependent on model

    xp_dot = model(pos, quat, vel, ang_vel, cmd, rotor_vel)

    pos, quat = jp.asarray(np.asarray(pos)), jp.asarray(np.asarray(quat))
    vel, ang_vel = jp.asarray(np.asarray(vel)), jp.asarray(np.asarray(ang_vel))
    rotor_vel = jp.asarray(np.asarray(rotor_vel)) if rotor_vel is not None else None
    cmd = jp.asarray(np.asarray(cmd))

    model_jit = jax.jit(model)
    jp_dot = model_jit(pos, quat, vel, ang_vel, cmd, rotor_vel)

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
