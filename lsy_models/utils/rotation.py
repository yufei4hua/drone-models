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
    import numpy as np
    from numpy.typing import NDArray

def from_quat(quat: NDArray[np.floating], scalar_first: bool = False) -> R:
    """Creates a rotation object compatible with the type of the given quat."""
    if isinstance(quat, jp.ndarray):#
        if scalar_first:
            raise ValueError("scalar_first is not supported by jax rotations")
        return JR.from_quat(quat)
    else:
        return R.from_quat(quat, scalar_first=scalar_first)


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