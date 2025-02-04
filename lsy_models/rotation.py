"""In this file, some important rotations from scipy.spatial.transform.rotation are reimplemented in the Array API to be useable with jax, numpy, etc.

https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jp
import numpy as np
from jax.scipy.spatial.transform import Rotation as JR
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


def quat2euler(quat: NDArray[np.floating], seq: str, degrees: bool = False) -> NDArray[np.floating]:
    if isinstance(quat, np.ndarray):
        return R.from_quat(quat).as_euler(seq, degrees=degrees)
    if isinstance(quat, jp.ndarray):
        return JR.from_quat(quat).as_euler(seq, degrees=degrees)
    raise ValueError(f"Expected numpy or jax array, got {type(quat)}")
