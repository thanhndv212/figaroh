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
import time

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num


from tiago_mocap_calib_fun_def import (
    extract_expData,
    get_param,
    init_var,
    get_PEE_fullvar,
    get_PEE_var,
    get_geoOffset,
    get_jointOffset,
    get_PEE,
    Calculate_kinematics_model,
    Calculate_identifiable_kinematics_model)

# load robot

robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

NbSample = 50
param = get_param(robot, NbSample)

dataSet = 'sample'  # choose data source 'sample' or 'experimental'
################ simulated data ##########################
if dataSet == 'sample':
    # create artificial offsets
    var_sample, nvars_sample = init_var(param, mode=1)
    print(var_sample)
    # create sample configurations
    q_sample = np.empty((param['NbSample'], model.nq))
    for i in range(param['NbSample']):
        config = pin.randomConfiguration(model)
        config[8:] = param['q0'][8:]
        q_sample[i, :] = config

    # create simulated end effector coordinates measures (PEEm)
    PEEm_sample = get_PEE_fullvar(
        var_sample, q_sample, model, data, param, noise=False)

    q_LM = np.copy(q_sample)
    PEEm_LM = np.copy(PEEm_sample)

################ experiment data ##########################
elif dataSet == 'experimental':
    # read csv file
    path = '/home/thanhndv212/Cooking/figaroh/data/exp_data_0924.csv'
    PEEm_exp, q_exp = extract_expData(path, param)

    q_LM = np.copy(q_exp)
    PEEm_LM = np.copy(PEEm_exp)

for k in range(8):
    print('to check if model modified', model.jointPlacements[k].translation)

# # NON-LINEAR model with Levenberg-Marquardt #################
# """
#     - minimize the difference between measured coordinates of end-effector
#     and its estimated values from DGM by Levenberg-Marquardt
# """
#############################################################

param['IDX_TOOL'] = model.getFrameId('arm_7_joint')


def cost_func(var, q, model, data, param,  PEEm):
    PEEe = get_PEE_fullvar(var, q, model, data, param, noise=False)
    res_vect = PEEm - PEEe
    # return res_vect[0:(param['NbSample']*3)]
    return res_vect


# initial guess
var_0, nvars = init_var(param, mode=0)

# calculate oMf up to the previous frame of end effector
LM_solve = least_squares(cost_func, var_0,  method='lm',
                         args=(q_LM, model, data, param,  PEEm_LM))

print("solution: ", LM_solve.x)
print("minimized cost function: ", LM_solve.cost)
print("optimality: ", LM_solve.optimality)

'''
# calculate standard deviation of estimated parameter
sigma_ro_sq = (LM_solve.cost**2) / \
    (param['NbSample']*param['calibration_index'] - nvars)
J = LM_solve.jac
C_param = sigma_ro_sq*np.linalg.pinv(np.dot(J.T, J))
std_dev = []
for i in range(nvars):
    std_dev.append(np.sqrt(C_param[i, i]))
print("standard deviation: ", std_dev)
# plot results
# PEE estimated by solution
PEEe_sol = get_PEE_fullvar(LM_solve.x, q_exp, model, data, param)
# PEE estimated without offset arm 2 to arm 6
offset_26 = np.copy(LM_solve.x)
offset_26[2:7] = np.zeros(5)
PEEe_nonoffs = get_PEE_var(offset_26, q_exp, model, data, param)
'''

'''
plt.figure()
colors = ['b', 'g', 'r']
data_label = ['pos_x', 'pos_y', 'pos_z']
for i in range(3):
    plt.plot(PEEe_sol[i*(param['NbSample']):(i+1) *
             (param['NbSample'])], color=colors[i], label='estimated ' + data_label[i])
    plt.plot(PEEm_exp[i*(param['NbSample']):(i+1) *
             (param['NbSample'])], lineStyle='dashed', marker='o', color=colors[i], label='measured ' + data_label[i])
    plt.plot(PEEe_nonoffs[i*(param['NbSample']):(i+1) *
             (param['NbSample'])], lineStyle='dashdot', marker='o', color=colors[i], label='estimated without offset ' + data_label[i])
    plt.legend(loc='upper left')
plt.xlabel('Number of postures')
plt.ylabel('XYZ coordinates of end effector frame (m) ')
plt.title(
    'Comparison of end effector positions by measurement of MoCap and estimation of calibrated model')
plt.grid()
plt.show()
'''
# # LINEARIZED model with iterative least square ###############
# """
#     - obtaining base regressor and base geometric parameters (BGP) offsets
#     - update the regressor and predited coordinates errors at every iteration
#     - apply least square to get BGP offsets
#     - repeate the process until coordinates errors < threshold
# """
# # ###########################################################
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
