"""In this file, some important rotations from scipy.spatial.transform.rotation are reimplemented in the Array API to be useable with jax, numpy, etc.

https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jp
import numpy as np
import casadi as cs
from jax.scipy.spatial.transform import Rotation as JR
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

def from_quat(quat: NDArray[np.floating], scalar_first: bool = False) -> R:
    if isinstance(quat, np.ndarray):
        return R.from_quat(quat, scalar_first=scalar_first)
    if isinstance(quat, jp.ndarray):
        return JR.from_quat(quat)
    raise ValueError(f"Expected numpy or jax array, got {type(quat)}")


def casadi_quat2matrix(quat: cs.MX) -> cs.MX:
    """TODO.
    
    From https://github.com/cmower/spatial-casadi/blob/master/spatial_casadi/spatial.py
    """
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

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
        cs.vertcat(
            x2 - y2 - z2 + w2,
            2.0 * (xy + zw),
            2.0 * (xz - yw),
        ),
        cs.vertcat(
            2.0 * (xy - zw),
            -x2 + y2 - z2 + w2,
            2.0 * (yz + xw),
        ),
        cs.vertcat(
            2.0 * (xz + yw),
            2.0 * (yz - xw),
            -x2 - y2 + z2 + w2,
        ),
    )

    return matrix