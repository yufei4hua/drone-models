"""This file is loads all constants for a specific drone, based on the drone type, and stores it in a dataclass."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor

    Array = NDArray | JaxArray | Tensor


@dataclass
class Constants:
    """This is a dataclass for all necessary constants in the models."""

    GRAVITY: float
    GRAVITY_VEC: Array
    MASS: float
    J: Array
    J_INV: Array
    L: float
    MIX_MATRIX: Array
    SIGN_MATRIX: Array

    PWM_MIN: float
    PWM_MAX: float
    KF: float
    KM: float
    THRUST_MIN: float
    THRUST_MAX: float
    THRUST_TAU: float

    # System Identification (SI) parameters
    SI_ROLL: Array
    SI_PITCH: Array
    SI_YAW: Array
    SI_PARAMS: Array
    SI_ACC: Array

    # System Identification parameters for the double integrator (DI) model
    DI_ROLL: Array
    DI_PITCH: Array
    DI_YAW: Array
    DI_PARAMS: Array
    DI_ACC: Array

    # Configs (used in testing)
    available_configs: tuple[str] = ("cf2x_L250", "cf2x_P250", "cf2x_T350")

    @classmethod
    def from_file(cls, path: str) -> Constants:
        """Creates constants based on the xml file at the given location.

        The constants are supposed to be under the costum/numeric category.
        """
        # Constants
        drone_path = Path(__file__).parents[1] / path
        # read in all parameters from xml
        params = ET.parse(drone_path).findall(".//custom/numeric")
        # create a dict from parameters containing array of floats
        params = {p.get("name"): np.array(list(map(float, p.get("data").split()))) for p in params}

        GRAVITY = params["gravity"][0]
        GRAVITY_VEC = np.array([0, 0, -GRAVITY])
        MASS = params["mass"][0]
        J = params["J"].reshape((3, 3))
        J_inv = np.linalg.inv(J)
        L = params["arm"][0]
        MIX_MATRIX = params["mix_matrix"].reshape((4, 3))
        SIGN_MATRIX = np.sign(MIX_MATRIX)

        PWM_MIN = params["PWM_MIN"][0]
        PWM_MAX = params["PWM_MAX"][0]
        KF = params["kf"][0]
        KM = params["km"][0]
        THRUST_MIN = params["THRUST_MIN"][0]
        THRUST_MAX = params["THRUST_MAX"][0]
        THRUST_TAU = params["THRUST_TAU"][0]

        # System Identification (SI) parameters
        SI_ROLL = params["SI_roll"]
        SI_PITCH = params["SI_pitch"]
        SI_YAW = params["SI_yaw"]
        SI_PARAMS = np.vstack((SI_ROLL, SI_PITCH, SI_YAW))
        SI_ACC = params["SI_acc"]

        # System Identification parameters for the double integrator (DI) model
        DI_ROLL = params["DI_roll"]
        DI_PITCH = params["DI_pitch"]
        DI_YAW = params["DI_yaw"]
        DI_PARAMS = np.vstack((DI_ROLL, DI_PITCH, DI_YAW))
        DI_ACC = params["DI_acc"]

        return cls(
            GRAVITY,
            GRAVITY_VEC,
            MASS,
            J,
            J_inv,
            L,
            MIX_MATRIX,
            SIGN_MATRIX,
            PWM_MIN,
            PWM_MAX,
            KF,
            KM,
            THRUST_MIN,
            THRUST_MAX,
            THRUST_TAU,
            SI_ROLL,
            SI_PITCH,
            SI_YAW,
            SI_PARAMS,
            SI_ACC,
            DI_ROLL,
            DI_PITCH,
            DI_YAW,
            DI_PARAMS,
            DI_ACC,
        )

    @classmethod
    def from_config(cls, config: str) -> Constants:
        """Creates constants based on the give configuration.

        For available configs see Constants.available_configs
        """
        match config:
            case "cf2x_L250":
                return Constants.from_file("data/cf2x_L250.xml")
            case "cf2x_P250":
                return Constants.from_file("data/cf2x_P250.xml")
            case "cf2x_T350":
                return Constants.from_file("data/cf2x_T350.xml")
            case _:
                raise ValueError(f"Drone config '{config}' is not supported")
