"""Utility functions for the drone models and controllers."""

from types import ModuleType
from typing import Any

from array_api_typing import Array


def to_xp(*args: Any, xp: ModuleType, device: Any) -> tuple[Array, ...] | Array:
    """Convert all arrays in the argument list to the given xp framework and device."""
    result = tuple(xp.asarray(x, device=device) for x in args)
    if len(result) == 1:
        return result[0]
    return result
