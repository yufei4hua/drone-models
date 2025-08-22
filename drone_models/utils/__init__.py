"""Utility functions for the drone models and controllers."""

from types import ModuleType
from typing import Any

from array_api_typing import Array


def to_xp(*args: Any, xp: ModuleType, device: Any) -> tuple[Array, ...] | Array:
    result = tuple(xp.asarray(x, device=device) for x in args)
    if len(result) == 1:
        return result[0]
    return result
