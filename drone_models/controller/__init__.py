"""Implementations of onboard drone controllers in Python.

All controllers are implemented using the array API standard. This means that every controller is
agnostic to the choice of framework and supports e.g. NumPy, JAX, or PyTorch. We also implement all
controllers as pure functions to ensure that users can jit-compile them. All controllers use
broadcasting to support batching of arbitrary leading dimensions.
"""
