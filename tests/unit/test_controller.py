"""Tests of the numeric models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import array_api_compat.numpy as np
import array_api_strict as xp
import jax
import pytest

from drone_models.controller import available_controller
from drone_models.controller.constants import cntrl_const_mel
from drone_models.utils.constants import Constants, available_drone_types

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


@pytest.mark.unit
@pytest.mark.parametrize("controller_name, controller", available_controller.items())
@pytest.mark.parametrize("drone_type", available_drone_types)
def test_controller(controller_name: str, controller: Callable, drone_type: str):
    """TODO."""
    constants = Constants.from_config(drone_type, xp)
    batch_shape = (10,)
    pos, quat, vel, ang_vel, _, _, _ = create_rnd_states(batch_shape)
    cmd = create_rnd_commands(batch_shape, dim=13)  # TODO make dependent on controller

    parameters = cntrl_const_mel(xp)

    controller(pos, quat, vel, ang_vel, cmd, constants, parameters)
