"""This file is loads all constants for a specific drone, based on the drone type, and stores it in a dataclass."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Constants:
    """This is a dataclass for all necessary constants in the models."""
    GRAVITY: np.floating
    GRAVITY_VEC: NDArray[np.floating]
    MASS: np.floating
    J: NDArray[np.floating]
    J_inv: NDArray[np.floating]
    L: np.floating
    PWM_MIN: np.floating
    PWM_MAX: np.floating

    KF: np.floating
    KM: np.floating
    KD: np.floating
    THRUST_MIN: np.floating
    THRUST_MAX: np.floating

    # System Identification (SI) parameters
    SI_ROLL: NDArray[np.floating]
    SI_PITCH: NDArray[np.floating]
    SI_YAW: NDArray[np.floating]
    SI_PARAMS: NDArray[np.floating]
    SI_ACC: NDArray[np.floating]

    # System Identification parameters for the double integrator (DI) model
    DI_ROLL: NDArray[np.floating]
    DI_PITCH: NDArray[np.floating]
    DI_YAW: NDArray[np.floating]
    DI_PARAMS: NDArray[np.floating]
    DI_ACC: NDArray[np.floating]

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
        GRAVITY_VEC = np.array([0,0,-GRAVITY])
        MASS = params["mass"][0]
        J = params["J"].reshape((3,3))
        J_inv = np.linalg.inv(J)
        L = params["arm"][0]
        PWM_MIN = params["PWM_MIN"][0]
        PWM_MAX = params["PWM_MAX"][0]

        KF = params["kf"][0]
        KM = params["km"][0]
        KD = params["kd"][0]
        THRUST_MIN = params["THRUST_MIN"][0]
        THRUST_MAX = params["THRUST_MAX"][0]

        # System Identification (SI) parameters
        SI_ROLL = params["SI_roll"]
        SI_PITCH = params["SI_pitch"]
        SI_YAW = params["SI_yaw"]
        SI_PARAMS = np.vstack((SI_ROLL, SI_PITCH, SI_YAW))
        SI_ACC = params["DI_acc"] # same parameters for both models

        # System Identification parameters for the double integrator (DI) model
        DI_ROLL = params["DI_roll"]
        DI_PITCH = params["DI_pitch"]
        DI_YAW = params["DI_yaw"]
        DI_PARAMS = np.vstack((DI_ROLL, DI_PITCH, DI_YAW))
        DI_ACC = params["DI_acc"]

        return cls(GRAVITY, GRAVITY_VEC, MASS, J, J_inv, L, PWM_MIN, PWM_MAX, KF, KM, KD, THRUST_MIN, THRUST_MAX, 
                   SI_ROLL, SI_PITCH, SI_YAW, SI_PARAMS, SI_ACC, 
                   DI_ROLL, DI_PITCH, DI_YAW, DI_PARAMS, DI_ACC)