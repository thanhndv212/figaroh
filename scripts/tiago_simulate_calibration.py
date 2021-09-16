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

import pandas as pd
import csv

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num


def get_geoOffset(joint_names):
    tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]

    geo_params = []

    for i in range(len(joint_names)):
        for j in tpl_names:
            geo_params.append(j + ("_%d" % i))
    phi_gp = [0] * len(geo_params)

    geo_params = dict(zip(geo_params, phi_gp))

    return geo_params


def get_jointOffset(joint_names):
    joint_off = []
    for i in range(len(joint_names)):
        joint_off.append("off" + "_%d" % i)

    phi_jo = [0] * len(joint_off)
    joint_off = dict(zip(joint_off, phi_jo))
    return joint_off


# load robot
robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf"
)
print(robot.model)
joint_names = [name for i, name in enumerate(robot.model.names)]
geo_params = get_geoOffset(joint_names)
joint_offs = get_jointOffset(joint_names)

# create artificial offsets
offset_factor = 0.01
active_jointID = range(1, 9)
lb = [robot.model.lowerPositionLimit[i] for i in active_jointID]
ub = [robot.model.upperPositionLimit[i] for i in active_jointID]
offset = np.array([offset_factor*(ub[i] - lb[i]) for i in range(len(lb))])

# create configs and configs w/ offsets (equivalent to joint offsets)
nsample = 100

q = np.empty((nsample, robot.model.nq))
q_offset = np.empty((nsample, robot.model.nq))
for i in range(nsample):
    config = pin.randomConfiguration(robot.model)
    config[8:] = robot.q0[8:]
    q[i, :] = config

    noise = np.random.normal(0, 0.001, offset.shape)
    config_offset = config
    config_offset[:8] = config[:8] + offset
    q_offset[i, :] = config_offset


IDX_TOOL = robot.model.getFrameId("ee_marker_joint")

# create simulated end effector coordinates measures (PEEm)
PEEe = np.empty((6, nsample))
for i in range(nsample):
    pin.framesForwardKinematics(robot.model, robot.data, q[i, :])
    pin.updateFramePlacements(robot.model, robot.data)
    PEEe[0:3, i] = robot.data.oMf[IDX_TOOL].translation
    PEEe_rot = robot.data.oMf[IDX_TOOL].rotation
    PEEe[3:6, i] = pin.rpy.matrixToRpy(PEEe_rot)
PEEe = PEEe.flatten('C')
# estimate end effector coordinates from FGM (PEEe)
PEEm = np.empty((6, nsample))
for i in range(nsample):
    pin.framesForwardKinematics(robot.model, robot.data, q_offset[i, :])
    pin.updateFramePlacements(robot.model, robot.data)
    PEEm[0:3, i] = robot.data.oMf[IDX_TOOL].translation
    PEEm_rot = robot.data.oMf[IDX_TOOL].rotation
    PEEm[3:6, i] = pin.rpy.matrixToRpy(PEEm_rot)
PEEm = PEEm.flatten('C')

# create regressor: modeling code
J = np.empty([6*nsample, robot.model.nv])
R = np.empty([6*nsample, 6*robot.model.nv])
for i in range(nsample):
    q_temp = q[i, :]
    pin.framesForwardKinematics(robot.model, robot.data, q_temp)
    pin.updateFramePlacements(robot.model, robot.data)

    fj = pin.computeFrameJacobian(robot.model,
                                  robot.data, q_temp, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)
    kfr = pin.computeFrameKinematicRegressor(robot.model,
                                             robot.data, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)
    for j in range(6):
        J[nsample*j + i, :] = fj[j, :]
        R[nsample*j + i, :] = kfr[j, :]
# select columns correspond to joint offset
joint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
             50, 51, 52, 53]  # all on z axis - checked!!

# regressor matrix on selected paramters
R_sel = R[:, joint_idx]

# a dictionary of selected parameters
gp_listItems = list(geo_params.items())
geo_params_sel = []
for i in joint_idx:
    geo_params_sel.append(gp_listItems[i])
geo_params_sel = dict(geo_params_sel)
# eliminate zero columns
R_e, geo_paramsr = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)
# get base parameters
R_b, params_base = get_baseParams(R_e, geo_paramsr)


# LM code: estimate offset parameters "off_params" with output: PEEm - PEEe, J(q, params)
nparams = len(params_base)
x = np.empty((nparams,))

del_X = PEEm - PEEe


def cost_func(x, R_b, del_X):
    return np.dot(R_b, x) - del_X


x0 = np.zeros(nparams)

LM_solve = least_squares(cost_func, x0,  method='lm', args=(R_b, del_X))
print(offset)
print("solution: ", LM_solve.x)
print("minimized cost function: ", LM_solve.cost)
