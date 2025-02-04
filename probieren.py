import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import lsy_models.rotation as rot

# For testing
quats = [
    [1, 1, 1, 1],
    [0, 0, 0, -1],
    [0.5, 0, 0, 1],
    [0, 0.5, 0, 1],
    [0, 0, 0.5, 1],
    [1, 1, 1, 1],
    [0, 0, 0, -1],
    [0.5, 0, 0, 1],
    [0, 0.5, 0, 1],
    [0, 0, 0.5, 1],
    [1, 1, 1, 1],
    [0, 0, 0, -1],
    [0.5, 0, 0, 1],
    [0, 0.5, 0, 1],
    [0, 0, 0.5, 1],
    [1, 1, 1, 1],
    [0, 0, 0, -1],
    [0.5, 0, 0, 1],
    [0, 0.5, 0, 1],
    [0, 0, 0.5, 1],
    [1, 1, 1, 1],
    [0, 0, 0, -1],
    [0.5, 0, 0, 1],
    [0, 0.5, 0, 1],
    [0, 0, 0.5, 1],
    [1, 1, 1, 1],
    [0, 0, 0, -1],
    [0.5, 0, 0, 1],
    [0, 0.5, 0, 1],
    [0, 0, 0.5, 1],
]
# quats /= np.linalg.norm(quats) # normalization

for q in quats:
    rotation = R.from_quat(q)
    print(rotation.as_euler("XYZ"), rot.quat2euler(np.array(q), "XYZ"))
    assert np.allclose(rotation.as_euler("xyz"), rot.quat2euler(np.array(q), "xyz"))
    # assert np.allclose(rotation.as_euler("XYZ"), rot.quat2euler(np.array(q), "XYZ"))
    print(q)

# rotations = R.from_quat(quats)
# euler_xyz = rotations.as_euler("xyz")
# euler_XYZ = rotations.as_euler("XYZ")

# print(euler_xyz, euler_XYZ)

# print(quat2euler(quats, "xyz"), quat2euler(quats, "XYZ"))

# assert np.allclose(rotations.as_euler("xyz"), quat2euler(quats, "xyz"))
# assert np.allclose(rotations.as_euler("XYZ"), quat2euler(quats, "XYZ"))


# timings:
t1 = time.perf_counter()
rot.quat2euler(np.array(quats[0]), "xyz")
t2 = time.perf_counter()
print(f"time for xp quat2euler 1D: {(t2-t1)*1000:.3f}ms")

t1 = time.perf_counter()
rot.quat2euler(np.array(quats), "xyz")
t2 = time.perf_counter()
print(f"time for xp quat2euler 2D: {(t2-t1)*1000:.3f}ms")


# compared to roation library:
t1 = time.perf_counter()
R.from_quat(np.array(quats[0])).as_euler("xyz")
t2 = time.perf_counter()
print(f"time for scipy quat2euler 1D: {(t2-t1)*1000:.3f}ms")

t1 = time.perf_counter()
R.from_quat(np.array(quats)).as_euler("xyz")
t2 = time.perf_counter()
print(f"time for scipy quat2euler 2D: {(t2-t1)*1000:.3f}ms")

# Using timeit for more accurate benchmarking
import timeit

# XP quat2euler 1D
t = timeit.timeit(lambda: rot.quat2euler(np.array(quats[0]), "xyz"), number=1000)
print(f"time for xp quat2euler 1D (avg of 1000): {(t/1000)*1000:.3f}ms")

# XP quat2euler 2D
t = timeit.timeit(lambda: rot.quat2euler(np.array(quats), "xyz"), number=1000)
print(f"time for xp quat2euler 2D (avg of 1000): {(t/1000)*1000:.3f}ms")

# Scipy quat2euler 1D
t = timeit.timeit(lambda: R.from_quat(np.array(quats[0])).as_euler("xyz"), number=1000)
print(f"time for scipy quat2euler 1D (avg of 1000): {(t/1000)*1000:.3f}ms")

# Scipy quat2euler 2D
t = timeit.timeit(lambda: R.from_quat(np.array(quats)).as_euler("xyz"), number=1000)
print(f"time for scipy quat2euler 2D (avg of 1000): {(t/1000)*1000:.3f}ms")
