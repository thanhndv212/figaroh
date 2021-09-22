from scripts.test_2DOF_calibration_main import Calculate_kinematics_model
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

from tiago_mocap_calib_fun_def import get_geoOffset, get_jointOffset, get_PEE, Calculate_kinematics_model, Calculate_identifiable_kinematics_model


# load robot
robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf"
)
model = robot.model
print(model)
data = robot.data
id = model.getFrameId('torso_lift_joint')
print(data.oMf[id])

joint_names = [name for i, name in enumerate(model.names)]
geo_params = get_geoOffset(joint_names)
joint_offs = get_jointOffset(joint_names)
NbSample = 8

# create values storing dictionary 'param'
param = {
    'q0': np.array(robot.q0),
    'IDX_TOOL': model.getFrameId('ee_marker_joint'),
    'NbSample': NbSample,
    'eps': 1e-3,
    'Ind_joint': np.arange(8),
    'PLOT': 0,
    'calibration_index': 3
}

# create artificial offsets
offset_factor = 0.002
active_jointID = range(1, 9)
lb = [model.lowerPositionLimit[i] for i in active_jointID]
ub = [model.upperPositionLimit[i] for i in active_jointID]
offset = np.array([offset_factor*(ub[j] - lb[j]) *
                  (random.getrandbits(1)*2 - 1) for j in range(len(lb))])
nvars = 8

offset_0 = np.zeros(nvars)  # zero offsets


# create configs and configs w/ offsets (equivalent to joint offsets)
q = np.empty((param['NbSample'], model.nq))
q_offset = np.empty((param['NbSample'], model.nq))
for i in range(param['NbSample']):
    config = pin.randomConfiguration(model)
    config[8:] = robot.q0[8:]
    q[i, :] = config

    noise = np.random.normal(0, 0.001, offset.shape)
    config_offset = config
    config_offset[:8] = config[:8] + offset
    q_offset[i, :] = config_offset


# create simulated end effector coordinates measures (PEEm)
PEEm = get_PEE(offset_0, q_offset, model, data, param, noise=False)


# NON-LINEAR model with Levenberg-Marquardt
"""
    - minimize the difference between measured coordinates of end-effector 
    and its estimated values from DGM by Levenberg-Marquardt
"""
# estimate end effector coordinates from FGM (PEEe)


def cost_func(x, q, model, data, param,  PEEm):
    PEEe = get_PEE(x, q, model, data, param, noise=False)
    return (PEEm-PEEe)


x0 = np.zeros(nvars)
LM_solve = least_squares(cost_func, x0,  method='lm',
                         args=(q, model, data, param,  PEEm))
print("predefined offsets: ", offset)
print("solution: ", LM_solve.x)
print("minimized cost function: ", LM_solve.cost)
print("optimality: ", LM_solve.optimality)


# LINEARIZED model with iterative least square
"""
    - obtaining base regressor and base geometric parameters (BGP) offsets
    - update the regressor and predited coordinates errors at every iteration
    - apply least square to get BGP offsets
    - repeate the process until coordinates errors < threshold 
"""
# create regressor: modeling code
R_b, params_baseR, J_b, params_baseJ = Calculate_identifiable_kinematics_model(
    q, model, data, param)
print(R_b.shape)
# iterative pseudo inverse matrix
eps = 1e-3  # threshold for delta_X

PEEe = get_PEE(offset_0, q, model, data, param, noise=False)
delta_X = PEEm - PEEe
delta_p = np.dot(np.linalg.pinv(J_b), delta_X)
print("iteration 1: ", delta_p)
iter = 1


while np.linalg.norm(delta_X) > eps and iter < 10:
    iter += 1
    # update delta_X
    PEEe_update = get_PEE(delta_p, q, model, data, param)
    delta_X = PEEm - PEEe

    # update q
    q_update = np.empty(q.shape)
    for i in range(param['NbSample']):
        q_update[i, :8] = q[i, :8] + delta_p

    # update regressor
    _, _, J_b_update, _ = Calculate_identifiable_kinematics_model(
        q_update, model, data, param)
    print(data.oMf[id])

    delta_p = np.dot(np.linalg.pinv(J_b_update), delta_X)
    # print("iteration %d: " % iter, delta_p)
    # print("norm: ", np.linalg.norm(delta_X))

print("predefined offsets: ", offset)
