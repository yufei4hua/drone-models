"""In this file, some important rotations from scipy.spatial.transform.rotation are reimplemented in the Array API to be useable with jax, numpy, etc.

https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


def quat2euler(quat: NDArray[np.floating], seq: str, degrees: bool = False) -> NDArray[np.floating]:
    if len(seq) != 3:
      raise ValueError(f"Expected 3 axes, got {seq}.")
    intrinsic = (re.match(r'^[XYZ]{1,3}$', seq) is not None)
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    if not (intrinsic or extrinsic):
      raise ValueError("Expected axes from `seq` to be from "
                       "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                       "got {}".format(seq))
    if any(seq[i] == seq[i+1] for i in range(2)):
      raise ValueError("Expected consecutive axes to be different, "
                       "got {}".format(seq))
    
    xp = quat.__array_namespace__()

    angle_first = xp.where(extrinsic, 0, 2)
    angle_third = xp.where(extrinsic, 2, 0)
    axes = xp.array([_elementary_basis_index(x) for x in seq.lower()])
    axes = xp.where(extrinsic, axes, axes[::-1])
    i = axes[0]
    j = axes[1]
    k = axes[2]
    symmetric = i == k
    k = xp.where(symmetric, 3 - i - j, k)
    sign = xp.array((i - j) * (j - k) * (k - i) // 2, dtype=quat.dtype)
    eps = 1e-7
    a = xp.where(symmetric, quat[..., 3], quat[..., 3] - quat[..., j])
    b = xp.where(symmetric, quat[..., i], quat[..., i] + quat[..., k] * sign)
    c = xp.where(symmetric, quat[..., j], quat[..., j] + quat[..., 3])
    d = xp.where(symmetric, quat[..., k] * sign, quat[..., k] * sign - quat[..., i])
    # print(f"a.shape={a.shape}")
    # angles = xp.empty(3, dtype=quat.dtype)
    angles = xp.empty(quat.shape[:-1]+(3,), dtype=quat.dtype)
    # angles = angles.at[1].set(2 * xp.arctan2(xp.hypot(c, d), xp.hypot(a, b)))
    angles = _at(angles, 1, 2 * xp.arctan2(xp.hypot(c, d), xp.hypot(a, b)))
    case = xp.where(xp.abs(angles[..., 1] - xp.pi) <= eps, 2, 0)
    case = xp.where(xp.abs(angles[..., 1]) <= eps, 1, case)
    half_sum = xp.arctan2(b, a)
    half_diff = xp.arctan2(d, c)
    # print(f"case.shape={case.shape}, half_sum.shape={half_sum.shape}, half_diff.shape={half_diff.shape}")
    # print(2 * half_sum)
    # angles = angles.at[0].set(xp.where(case == 1, 2 * half_sum, 2 * half_diff * xp.where(extrinsic, -1, 1)))  
    angles = _at(angles, 0, xp.where(case == 1, 2 * half_sum, 2 * half_diff * xp.where(extrinsic, -1, 1)))  # any degenerate case
    # angles = angles.at[angle_first].set(xp.where(case == 0, half_sum - half_diff, angles[angle_first]))
    angles = _at(angles, angle_first, xp.where(case == 0, half_sum - half_diff, angles[..., angle_first]))
    # angles = angles.at[angle_third].set(xp.where(case == 0, half_sum + half_diff, angles[angle_third]))
    angles = _at(angles, angle_third, xp.where(case == 0, half_sum + half_diff, angles[..., angle_third]))
    # angles = angles.at[angle_third].set(xp.where(symmetric, angles[angle_third], angles[angle_third] * sign))
    angles = _at(angles, angle_third, xp.where(symmetric, angles[..., angle_third], angles[..., angle_third] * sign))
    # angles = angles.at[1].set(xp.where(symmetric, angles[1], angles[1] - xp.pi / 2))
    angles = _at(angles, 1, xp.where(symmetric, angles[..., 1], angles[..., 1] - xp.pi / 2))
    angles = (angles + xp.pi) % (2 * xp.pi) - xp.pi
    return xp.where(degrees, xp.rad2deg(angles), angles)

def _at(array: NDArray, index: int, new_value: float):
    xp = array.__array_namespace__()
    # print(f"array.shape={array.shape}, new_value.shape={xp.expand_dims(new_value, axis=-1).shape}")
    return xp.concat([array[..., :index], xp.expand_dims(new_value, axis=-1), array[..., index + 1:]], axis=-1)

def _elementary_basis_index(axis: str) -> int:
    if axis == 'x':
        return 0
    elif axis == 'y':
        return 1
    elif axis == 'z':
        return 2
    raise ValueError(f"Expected axis to be from ['x', 'y', 'z'], got {axis}")