"""In this file, some important rotations from scipy.spatial.transform.rotation are reimplemented in the Array API to be useable with jax, numpy, etc.

https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
import jax.numpy as jp
from jax.scipy.spatial.transform import Rotation as JR
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


def from_quat(quat: Array, scalar_first: bool = False) -> R:
    """Creates a rotation object compatible with the type of the given quat."""
    if isinstance(quat, jp.ndarray):
        if scalar_first:
            raise ValueError("scalar_first is not supported by jax rotations")
        return JR.from_quat(quat)
    else:
        return R.from_quat(quat, scalar_first=scalar_first)


def from_rotvec(rotvec: Array, degrees: bool = False) -> R:
    """Creates a rotation object compatible with the type of the given rotvec."""
    if isinstance(rotvec, jp.ndarray):
        return JR.from_rotvec(rotvec, degrees)
    else:
        return R.from_rotvec(rotvec, degrees)


def from_euler(seq: str, angles: Array, degrees: bool = False) -> R:
    """Creates a rotation object compatible with the type of the given angles."""
    if isinstance(angles, jp.ndarray):
        return JR.from_euler(seq, angles, degrees)
    else:
        return R.from_euler(seq, angles, degrees)


def from_matrix(matrix: Array) -> R:
    """Creates a rotation objecte compatible with the type of the given rotation matrix."""
    if isinstance(matrix, jp.ndarray):
        return JR.from_matrix(matrix)
    else:
        return R.from_matrix(matrix)


def casadi_quat2matrix(quat: cs.MX) -> cs.MX:
    """Creates a symbolic rotation matrix based on a symbolic quaternion.

    From https://github.com/cmower/spatial-casadi/blob/master/spatial_casadi/spatial.py
    """
    x = quat[0] / cs.norm_2(quat)
    y = quat[1] / cs.norm_2(quat)
    z = quat[2] / cs.norm_2(quat)
    w = quat[3] / cs.norm_2(quat)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = cs.horzcat(
        cs.vertcat(x2 - y2 - z2 + w2, 2.0 * (xy + zw), 2.0 * (xz - yw)),
        cs.vertcat(2.0 * (xy - zw), -x2 + y2 - z2 + w2, 2.0 * (yz + xw)),
        cs.vertcat(2.0 * (xz + yw), 2.0 * (yz - xw), -x2 - y2 + z2 + w2),
    )

    return matrix
