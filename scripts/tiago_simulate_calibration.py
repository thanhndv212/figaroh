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
    extract_expData4Mkr,
    get_param,
    init_var,
    get_PEE_fullvar,
    get_PEE_var,
    get_geoOffset,
    get_jointOffset,
    get_PEE,
    Calculate_kinematics_model,
    Calculate_identifiable_kinematics_model,
    Calculate_base_kinematics_regressor)

# load robot

robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

NbSample = 50
param = get_param(robot, NbSample, TOOL_NAME='ee_marker_joint', NbMarkers=4)

# Base parameters generation
q = []  # testing with random configurations
# calcualte base regressor of kinematic errors model and the base parameters expressions
Rrand_b, R_b, params_base, params_e = Calculate_base_kinematics_regressor(
    q, model, data, param)
# naming for eeframe markers
PEE_names = []
for i in range(param['NbMarkers']):
    PEE_names.extend(['pEEx_%d' % (i+1), 'pEEy_%d' % (i+1), 'pEEz_%d' % (i+1)])
params_name = params_base + PEE_names
print(params_name)

# Data collection
dataSet = 'experimental'  # choose data source 'sample' or 'experimental'
################ simulated data ##########################
if dataSet == 'sample':
    # create artificial offsets
    var_sample, nvars_sample = init_var(param, mode=1)
    print("%d var_sample: " % nvars_sample, var_sample)
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
    # read csv fileT
    path = '/home/thanhndv212/Cooking/figaroh/data/exp_data_nov_64_3011.csv'
    PEEm_exp, q_exp = extract_expData4Mkr(path, param)

    q_LM = np.copy(q_exp)
    PEEm_LM = np.copy(PEEm_exp)
for k in range(param['NbJoint']+1):
    print('to check if model modified',
          model.names[k], ": ", model.jointPlacements[k].translation)
print('updated number of samples: ', param['NbSample'])
# # NON-LINEAR model with Levenberg-Marquardt #################
# """
#     - minimize the difference between measured coordinates of end-effector
#     and its estimated values from DGM by Levenberg-Marquardt
# """
#############################################################

# Data inspecting
PEEm_xyz = PEEm_LM.reshape((param['NbMarkers']*3, param["NbSample"]))
print(PEEm_xyz.shape)


# absolute distance from mocap to markers
PEEm_dist = np.zeros((param['NbMarkers'], param["NbSample"]))
for i in range(param["NbMarkers"]):
    for j in range(param["NbSample"]):
        PEEm_dist[i, j] = np.sqrt(
            PEEm_xyz[i*3, j]**2 + PEEm_xyz[i*3 + 1, j]**2 + PEEm_xyz[i*3 + 2, j]**2)

err_PEE = np.zeros((2, param['NbSample']))
for j in range(param['NbSample']):
    dx_bot = PEEm_xyz[0, j]-PEEm_xyz[3, j]
    dy_bot = PEEm_xyz[1, j]-PEEm_xyz[4, j]
    dz_bot = PEEm_xyz[2, j]-PEEm_xyz[5, j]
    err_PEE[0, j] = np.sqrt(dx_bot**2 + dy_bot**2 + dz_bot**2)
    dx_top = PEEm_xyz[6, j]-PEEm_xyz[9, j]
    dy_top = PEEm_xyz[7, j]-PEEm_xyz[10, j]
    dz_top = PEEm_xyz[8, j]-PEEm_xyz[11, j]
    err_PEE[1, j] = np.sqrt(dx_top**2 + dy_top**2 + dz_top**2)

dist_fig, dist_axs = plt.subplots(2)
dist_fig.suptitle("Relative distances between markers (m) ")
dist_axs[0].plot(err_PEE[0, :], label="error between 2 bottom markers")
dist_axs[1].plot(err_PEE[1, :], label="error between 2 top markers")
dist_axs[0].legend()
dist_axs[1].legend()
dist_axs[0].axhline(np.mean(err_PEE[0, :]), 0,
                    param['NbSample'] - 1, color='r', linestyle='--')
dist_axs[1].axhline(np.mean(err_PEE[1, :]), 0,
                    param['NbSample'] - 1, color='r', linestyle='--')


# Levenberg-Marquardt algorithm
coeff = 1e-3


def cost_func(var, coeff, q, model, data, param,  PEEm):
    PEEe = get_PEE_fullvar(var, q, model, data, param, noise=False)
    res_vect = np.append((PEEm - PEEe), np.sqrt(coeff)
                         * var[6:-param['NbMarkers']*3])
    # res_vect = (PEEm - PEEe)
    return res_vect


# initial guess
# mode = 1: random seed [-0.01, 0.01], mode = 0: init guess = 0
var_0, nvars = init_var(param, mode=1)
print("initial guess: ", var_0)

# solve
LM_solve = least_squares(cost_func, var_0,  method='lm', verbose=1,
                         args=(coeff, q_LM, model, data, param,  PEEm_LM))

# PEE estimated by solution
PEEe_sol = get_PEE_fullvar(LM_solve.x, q_LM, model, data, param, noise=False)

rms = np.sqrt(np.mean((PEEe_sol-PEEm_LM)**2))
print("solution: ", LM_solve.x)
print("minimized cost function: ", rms)
print("optimality: ", LM_solve.optimality)

# result analysis
PEEe_xyz = PEEe_sol.reshape((param['NbMarkers']*3, param["NbSample"]))
PEEe_dist = np.zeros((param['NbMarkers'], param["NbSample"]))
for i in range(param["NbMarkers"]):
    for j in range(param["NbSample"]):
        PEEe_dist[i, j] = np.sqrt(
            PEEe_xyz[i*3, j]**2 + PEEe_xyz[i*3 + 1, j]**2 + PEEe_xyz[i*3 + 2, j]**2)

est_fig, est_axs = plt.subplots(4)
est_fig.suptitle(
    "Relative errors between estimated markers and measured markers in position (m) ")
est_axs[0].bar(np.arange(param['NbSample']), PEEe_dist[0, :] -
               PEEm_dist[0, :], label='bottom left')
est_axs[1].bar(np.arange(param['NbSample']), PEEe_dist[1, :] -
               PEEm_dist[1, :], label='bottom right')
est_axs[2].bar(np.arange(param['NbSample']), PEEe_dist[2, :] -
               PEEm_dist[2, :], label='top left')
est_axs[3].bar(np.arange(param['NbSample']), PEEe_dist[3, :] -
               PEEm_dist[3, :], label='bottom right')
est_axs[0].legend()
est_axs[1].legend()
est_axs[2].legend()
est_axs[3].legend()


# calculate standard deviation of estimated parameter ( Khalil chapter 11)
sigma_ro_sq = (LM_solve.cost**2) / \
    (param['NbSample']*param['calibration_index'] - nvars)
J = LM_solve.jac
C_param = sigma_ro_sq*np.linalg.pinv(np.dot(J.T, J))
std_dev = []
std_pctg = []
for i in range(nvars):
    std_dev.append(np.sqrt(C_param[i, i]))
    std_pctg.append(abs(np.sqrt(C_param[i, i])/LM_solve.x[i]))
path_save_ep = join(
    dirname(dirname(str(abspath(__file__)))),
    f"data/estimation_result.csv")
with open(path_save_ep, "w") as output_file:
    w = csv.writer(output_file)
    for i in range(nvars):
        w.writerow(
            [
                params_name[i],
                LM_solve.x[i],
                std_dev[i],
                std_pctg[i]
            ]
        )
print("standard deviation: ", std_dev)
# plot results

plt.figure(figsize=(7.5, 6))
if dataSet == 'sample':
    plt.barh(params_name, (LM_solve.x - var_sample), align='center')
elif dataSet == 'experimental':
    plt.barh(params_name[0:6], LM_solve.x[0:6], align='center')
    plt.barh(params_name[6:-3*param['NbMarkers']],
             LM_solve.x[6:-3*param['NbMarkers']], align='center')
    plt.barh(params_name[-3*param['NbMarkers']:],
             LM_solve.x[-3*param['NbMarkers']:], align='center')


plt.grid()
plt.show()
# plt.figure(2)

# colors = ['r', 'g', 'b']
# data_label = ['pos_x', 'pos_y', 'pos_z']
# for i in range(3):
#     plt.plot(PEEe_sol[i*(param['NbSample']):(i+1) *
#              (param['NbSample'])], color=colors[i], label='estimated ' + data_label[i])
#     plt.plot(PEEm_LM[i*(param['NbSample']):(i+1) *
#              (param['NbSample'])], lineStyle='dashed', marker='o', color=colors[i], label='measured ' + data_label[i])
#     # plt.plot(PEEe_nonoffs[i*(param['NbSample']):(i+1) *
#     #          (param['NbSample'])], lineStyle='dashdot', marker='o', color=colors[i], label='estimated without offset ' + data_label[i])
#     plt.legend(loc='upper left')
# plt.xlabel('Number of postures')
# plt.ylabel('XYZ coordinates of end effector frame (m) ')
# plt.title(
#     'Comparison of end effector positions by measurement of MoCap and estimation of calibrated model')
# plt.grid()
# plt.show()

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
