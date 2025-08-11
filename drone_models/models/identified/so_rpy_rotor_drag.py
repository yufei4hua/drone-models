"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace
from scipy.spatial.transform import Rotation as R

from drone_models.models.utils import supports
from drone_models.transform import motor_force2rotor_speed
from drone_models.utils import rotation

if TYPE_CHECKING:
    from array_api_typing import Array

    from drone_models.utils.constants import Constants


@supports(rotor_dynamics=True)
def dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    command: Array,
    constants: Constants,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted double integrator (DI) model with optional motor delay (D).

    Args:
        pos (Array): Position of the drone (m)
        quat (Array): Quaternion of the drone (xyzw)
        vel (Array): Velocity of the drone (m/s)
        ang_vel (Array): Angular velocity of the drone (rad/s)
        command (Array): RPYT command (roll, pitch, yaw in rad, thrust in N)
        constants (Constants): Containing the constants of the drone
        rotor_vel (Array | None, optional): Thrust of the 4 motors in N. Defaults to None.
            If None, the commanded thrust is directly applied. If value is given, thrust dynamics are calculated.
        dist_f: Disturbance force acting on the CoM. Defaults to None.
        dist_t: Disturbance torque acting on the CoM. Defaults to None.

    Returns:
        tuple[Array, Array, Array, Array, Array | None]: _description_
    """
    xp = array_namespace(pos)
    cmd_f = command[..., -1]
    cmd_rotor_vel = motor_force2rotor_speed(cmd_f, constants.KF)
    cmd_rpy = command[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")
    rpy_rates = rotation.ang_vel2rpy_rates(quat, ang_vel)

    rotor_vel_dot = (
        xp.asarray(1 / constants.DI_DD_ACC[2] * (cmd_rotor_vel / 4 - rotor_vel))
        - constants.DI_DD_ACC[3] * rotor_vel**2
    )
    forces_motor = xp.sum(constants.KF * rotor_vel**2, axis=-1)
    forces_sum = xp.sum(forces_motor, axis=-1)
    thrust = constants.DI_DD_ACC[0] + constants.DI_DD_ACC[1] * forces_sum

    drone_z_axis = rot.inv().as_matrix()[..., -1, :]

    pos_dot = vel
    vel_dot = (
        1 / constants.MASS * thrust[..., None] * drone_z_axis
        + constants.GRAVITY_VEC
        + 1 / constants.MASS * constants.DI_DD_ACC[2] * vel
        + 1 / constants.MASS * constants.DI_DD_ACC[3] * vel * xp.abs(vel)
    )
    if dist_f is not None:
        vel_dot = vel_dot + dist_f / constants.MASS

    # Rotational equation of motion
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    rpy_rates_dot = (
        constants.DI_DD_PARAMS[:, 0] * euler_angles
        + constants.DI_DD_PARAMS[:, 1] * rpy_rates
        + constants.DI_DD_PARAMS[:, 2] * cmd_rpy
    )
    ang_vel_dot = rotation.rpy_rates2ang_vel(quat, rpy_rates_dot)
    if dist_t is not None:
        # adding disturbances to the state
        # adding torque is a little more complex:
        # angular acceleration can be converted to torque
        torque = xp.matvec(constants.J, ang_vel_dot) - xp.linalg.cross(
            ang_vel, xp.matvec(constants.J, ang_vel)
        )
        # adding torque
        torque = torque + dist_t
        # back to angular acceleration
        ang_vel_dot = xp.matvec(constants.J_INV, torque)

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def symbolic_dynamics(): ...
