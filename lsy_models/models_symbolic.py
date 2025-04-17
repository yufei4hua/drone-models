"""TODO.

In large parts taken from
https://github.com/middleyuan/safe-control-gym/blob/d5f4f4f1cea112cab84031453b28c9ef65537ff0/safe_control_gym/envs/gym_pybullet_drones/quadrotor.py
"""

import casadi as cs

import lsy_models.utils.rotation as R
from lsy_models.utils.constants import Constants


def first_principles(constants: Constants) -> cs.Function:
    """TODO take from numeric."""
    # nx, nu = 13, 4

    # States
    px = cs.MX.sym("px")
    py = cs.MX.sym("py")
    pz = cs.MX.sym("pz")
    pos = cs.vertcat(px, py, pz)  # Position
    vx = cs.MX.sym("vx")
    vy = cs.MX.sym("vy")
    vz = cs.MX.sym("vz")
    vel = cs.vertcat(vx, vy, vz)  # Velocity
    qw = cs.MX.sym("qw")
    qx = cs.MX.sym("qx")
    qy = cs.MX.sym("qy")
    qz = cs.MX.sym("qz")
    quat = cs.vertcat(qw, qx, qy, qz)  # Quaternions
    rot = R.casadi_quat2matrix(quat)  # Rotation matrix from body to world frame
    p = cs.MX.sym("p")
    q = cs.MX.sym("q")
    r = cs.MX.sym("r")
    ang_vel = cs.vertcat(p, q, r)  # Quaternions
    f1 = cs.MX.sym("f1")
    f2 = cs.MX.sym("f2")
    f3 = cs.MX.sym("f3")
    f4 = cs.MX.sym("f4")
    forces_motor = cs.vertcat(f1, f2, f3, f4)  # Motor thrust
    X = cs.vertcat(pos, quat, vel, ang_vel, forces_motor)

    # Inputs
    f1_cmd = cs.MX.sym("f1_cmd")
    f2_cmd = cs.MX.sym("f2_cmd")
    f3_cmd = cs.MX.sym("f3_cmd")
    f4_cmd = cs.MX.sym("f4_cmd")
    U = cs.vertcat(f1_cmd, f2_cmd, f3_cmd, f4_cmd)  # U

    # Defining the dynamics function
    # Thrust dynamics
    forces_motor_dot = constants.THRUST_TAU * (U - forces_motor)
    # Creating force and torque vector
    forces_motor_vec = cs.vertcat(0, 0, cs.sum1(forces_motor))
    torques_motor_vec = (
        constants.SIGN_MATRIX.T
        @ forces_motor
        * cs.vertcat(constants.L, constants.L, constants.KM / constants.KF)
    )

    # Linear equation of motion
    pos_dot = vel
    vel_dot = (
        rot @ forces_motor_vec / constants.MASS + constants.GRAVITY_VEC
    )  # TODO add disturbance force

    # Rotational equation of motion
    xi = cs.vertcat(cs.horzcat(0, -ang_vel.T), cs.horzcat(ang_vel, -cs.skew(ang_vel)))
    quat_dot = 0.5 * (xi @ quat)
    ang_vel_dot = constants.J_INV @ (
        torques_motor_vec - cs.cross(ang_vel, constants.J @ ang_vel)
    )  # TODO add disturbance torque (rotated!)

    X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, forces_motor_dot)
    Y = cs.vertcat(pos, quat)

    return X_dot, X, U, Y
