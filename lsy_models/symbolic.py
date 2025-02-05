"""TODO.

In large parts taken from 
https://github.com/middleyuan/safe-control-gym/blob/d5f4f4f1cea112cab84031453b28c9ef65537ff0/safe_control_gym/envs/gym_pybullet_drones/quadrotor.py
"""

import casadi as cs
import numpy as np

import lsy_models.utils.const as const
import lsy_models.utils.rotation as R

# C = const.Constants.create("data/cf2x_-B250.xml")

# TODO move model dynamics from numeric and symbolic into one file
def model_dynamics(model: str, method: str) -> callable:
    match model: # TODO make constants in jp or np
        case "cf2x+":
            C = const.Constants.create("data/cf2x_-B250.xml")
        case "cf2x-":
            C = const.Constants.create("data/cf2x_+B250.xml")
        case _:
            raise ValueError(f"Model '{model}' is not supported")
        
    match method:
        case "first_principles":
            return first_principles(C)
        case "fit_SI":
            raise ValueError(f"Method '{method}' is not supported") # TODO
        case "fit_DI":
            raise ValueError(f"Method '{method}' is not supported") # TODO
        case _:
            raise ValueError(f"Method '{method}' is not supported")

def first_principles(C):
    nx, nu = 13, 4

    # States
    px = cs.MX.sym('px')
    py = cs.MX.sym('py')
    pz = cs.MX.sym('pz')
    pos = cs.vertcat(px, py, pz) # Position
    vx = cs.MX.sym('vx')
    vy = cs.MX.sym('vy')
    vz = cs.MX.sym('vz')
    vel = cs.vertcat(vx, vy, vz) # Velocity
    qw = cs.MX.sym('qw') 
    qx = cs.MX.sym('qx')
    qy = cs.MX.sym('qy')
    qz = cs.MX.sym('qz')
    quat = cs.vertcat(qw, qx, qy, qz) # Quaternions
    rot = R.casadi_quat2matrix(quat) # Rotation matrix from body to world frame
    p = cs.MX.sym('p')
    q = cs.MX.sym('q')
    r = cs.MX.sym('r')
    angvel = cs.vertcat(p, q, r) # Quaternions
    f1 = cs.MX.sym('f1')
    f2 = cs.MX.sym('f2')
    f3 = cs.MX.sym('f3')
    f4 = cs.MX.sym('f4')
    forces_motor = cs.vertcat(f1, f2, f3, f4) # Motor thrust
    X = cs.vertcat(pos, quat, vel, angvel, forces_motor)

    # Inputs
    f1_cmd = cs.MX.sym('f1_cmd')
    f2_cmd = cs.MX.sym('f2_cmd')
    f3_cmd = cs.MX.sym('f3_cmd')
    f4_cmd = cs.MX.sym('f4_cmd')
    forces_cmd = cs.vertcat(f1_cmd, f2_cmd, f3_cmd, f4_cmd) # U


    # Defining the dynamics function
    # Thrust dynamics
    forces_motor_dot = C.KD*(forces_cmd-forces_motor)
    # Creating force vector 
    forces_motor_vec = cs.vertcat(0,0,cs.sum1(forces_motor)) # TODO check if sum1 or sum2
    torque_motor_vec = cs.vertcat(
                        (f1 + f2 - f3 - f4) * C.L / cs.sqrt(2),
                        (-f1 + f2 + f3 - f4) * C.L / cs.sqrt(2),
                        C.KM/C.KF*cs.sum1(forces_motor)
                        ) # force = kf * rpm², torque = km * rpm² => torque = km/kf*force 

    # Linear equation of motion
    pos_dot = vel
    vel_dot = rot @ forces_motor_vec / C.MASS + C.GRAVITY_VEC # TODO add disturbance force

    # Rotational equation of motion
    xi = cs.vertcat( cs.horzcat(0, -angvel.T), cs.horzcat(angvel, -cs.skew(angvel)) )
    print(xi)
    quat_dot = 0.5*(xi @ quat)
    angvel_dot = C.J_inv @ (torque_motor_vec - cs.cross(angvel, C.J@angvel)) # TODO add disturbance torque (rotated!)

    X_dot = cs.vertcat(pos_dot, vel_dot, quat_dot, angvel_dot, forces_motor_dot)
    Y = cs.vertcat(pos, quat)

    return cs.Function('first_principles', [pos, vel, quat, angvel, forces_motor, forces_cmd], [X_dot])