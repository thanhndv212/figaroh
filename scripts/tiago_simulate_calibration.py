import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *

from sys import argv
import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import least_squares
import numpy as np
import random

import pandas as pd
import csv

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num

from tiago_mocap_calib_fun_def import get_PEE_var, get_geoOffset, get_jointOffset, add_eemarker_frame, get_PEE, Calculate_kinematics_model, Calculate_identifiable_kinematics_model


# load robot

robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data
'''
# convert rpy to quaternion for free-flyer
q = robot.q0
p = np.array([0.1, 0.1, 0.1])
rpy = np.array([0.1, 0.1, 0.1])
R = pin.rpy.rpyToMatrix(rpy)
some_placement = pin.SE3(R, p)
q[:7] = pin.SE3ToXYZQUAT(some_placement)
pin.framesForwardKinematics(model, data, q)
# add frame
p = np.array([0, 0, 0])
rpy = np.array([0, 0, 0])
R = pin.rpy.rpyToMatrix(rpy)
frame_placement = pin.SE3(R, p)
print(type(frame_placement))
parent_joint = model.getJointId("arm_7_joint")
prev_frame = model.getFrameId("arm_7_joint")
ee_frame_id = model.addFrame(
    pin.Frame("end_effector", parent_joint, prev_frame,
              frame_placement, pin.FrameType(0), pin.Inertia.Zero()), False)
pin.updateGlobalPlacements(model, data)
# print(model.getFrameId('end_effector'))

# check added frame
for i, frame in enumerate(model.frames):
    print('%d' % i, frame)
print(len(data.oMf[79]))
'''
joint_names = [name for i, name in enumerate(model.names)]
geo_params = get_geoOffset(joint_names)
joint_offs = get_jointOffset(joint_names)
NbSample = 20

# create values storing dictionary 'param'
param = {
    'q0': np.array(robot.q0),
    'x_opt_prev': np.zeros([8]),
    'IDX_TOOL': model.getFrameId('ee_marker_joint'),
    'NbSample': NbSample,
    'eps': 1e-3,
    'Ind_joint': np.arange(8),
    'PLOT': 0,
    'calibration_index': 6
}

# create artificial offsets
offset_factor = 0.01
active_jointID = range(1, 9)
lb = [model.lowerPositionLimit[i] for i in active_jointID]
ub = [model.upperPositionLimit[i] for i in active_jointID]
offset = np.array([offset_factor*(ub[j] - lb[j]) *
                  (random.getrandbits(1)*2 - 1) for j in range(len(lb))])
njoints = 8
offset_0 = np.zeros(njoints)  # zero offsets


# create sample configurations
q_sample = np.empty((param['NbSample'], model.nq))
for i in range(param['NbSample']):
    config = pin.randomConfiguration(model)
    config[8:] = robot.q0[8:]
    q_sample[i, :] = config

# create simulated end effector coordinates measures (PEEm)
PEEm = get_PEE(offset, q_sample, model, data, param, noise=False)

# x,y,z,r,p,y from mocap to base ( 6 parameters for pos and orient)
qBase_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# x,y,z,r,p,y from wrist to end_effector ( 6 parameters for pos and orient)
qEE_0 = np.array([0.0, 0.0, 0.0, 0., 0., 0.])

# variable vector 20 par = 6 base par + 8 joint off + 6 endeffector par
var_0 = np.append(np.append(qBase_0, offset_0), qEE_0)

# # NON-LINEAR model with Levenberg-Marquardt ##############
# """
#     - minimize the difference between measured coordinates of end-effector
#     and its estimated values from DGM by Levenberg-Marquardt
# """
# # ###########################################################


def cost_func(var, q_sample, model, data, param,  PEEm):
    PEEe = get_PEE_var(var, q_sample, model, data, param, noise=False)
    return (PEEm-PEEe)


# calculate oMf up to the previous frame of end effector
param['IDX_TOOL'] = model.getFrameId('arm_7_link')

LM_solve = least_squares(cost_func, var_0,  method='lm',
                         args=(q_sample, model, data, param,  PEEm))
print("predefined offsets: ", np.append(np.zeros([6]), np.append(
    offset, [0.1, 0.1, 0.1, 0., 0., 0.])))
print("solution: ", LM_solve.x)
print("minimized cost function: ", LM_solve.cost)
print("optimality: ", LM_solve.optimality)

# # LINEARIZED model with iterative least square
# """
#     - obtaining base regressor and base geometric parameters (BGP) offsets
#     - update the regressor and predited coordinates errors at every iteration
#     - apply least square to get BGP offsets
#     - repeate the process until coordinates errors < threshold
# """
# # create regressor: modeling code
# R_b, params_baseR, J_b, params_baseJ = Calculate_identifiable_kinematics_model(
#     q, model, data, param)
# # iterative pseudo inverse matrix
# eps = 1e-3  # threshold for delta_X

# PEEe = get_PEE(offset_0, q, model, data, param, noise=False)
# delta_X = PEEm - PEEe
# delta_p = np.dot(np.linalg.pinv(J_b), delta_X)
# # print("iteration 1: ", delta_p)
# iter = 1


# while np.linalg.norm(delta_X) > eps and iter < 10:
#     iter += 1
#     # update delta_X
#     PEEe_update = get_PEE(delta_p, q, model, data, param)
#     delta_X = PEEm - PEEe

#     # update q
#     q_update = np.empty(q.shape)
#     for i in range(param['NbSample']):
#         q_update[i, :8] = q[i, :8] + delta_p

#     # update regressor
#     _, _, J_b_update, _ = Calculate_identifiable_kinematics_model(
#         q_update, model, data, param)

#     delta_p = np.dot(np.linalg.pinv(J_b_update), delta_X)
#     # print("iteration %d: " % iter, delta_p)
#     # print("norm: ", np.linalg.norm(delta_X))

# print("predefined offsets: ", offset)
