from numpy.core.arrayprint import DatetimeFormat
from datetime import datetime
from numpy.core.fromnumeric import shape
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
from pinocchio.utils import *
# from pinocchio.pinocchio_pywrap import rpy

from sys import argv
import os
from os.path import dirname, join, abspath
import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.linalg import norm, solve
from scipy import linalg, signal

import pandas as pd
import json
import csv

from tools.robot import Robot
from tools.regressor import *
from tools.qrdecomposition import *
from tools.randomdata import *
from tools.robotcollisions import *


"""
# load robot

# create a dictionary of geometric parameters errors

# generate data points

# get jacobian + kinematic regressor => rearranging

# eliminate zero columns

# apply qr decomposition to find independent columns and regrouping

# identifiable parameters expressions
"""

# load robot
robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf"
)
print(robot.model)


# create a dictionary of geometric parameters errors

"""
kinematic tree:
1: base -> torso->arm
2: base -> torso -> head
3: base -> wheels
for now, we only care about branch 1
"""

joint_names = [
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
    "head_1_joint",
    "head_2_joint",
    "wheel_left_joint",
    "wheel_right_joint",
    "ee_marker_joint"]

tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]

geo_params = []
joint_off = []

for i in range(len(joint_names)):
    for j in tpl_names:
        geo_params.append(j + ("_%d" % i))
    joint_off.append("off" + "_%d" % i)

# generate data points and regressors
ID_e = robot.model.getFrameId("ee_marker_joint")
nsample = 100

#  jacobian matrix J - only to  offsets ( = robot.model.nv)
#  kinematic regressor R - to  all 6-component parameters

######################################
J = np.empty([6*nsample, robot.model.nv])
R = np.empty([6*nsample, 6*robot.model.nv])
for i in range(nsample):
    q = pin.randomConfiguration(robot.model)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    fj = pin.computeFrameJacobian(robot.model,
                                  robot.data, q, ID_e, pin.LOCAL_WORLD_ALIGNED)
    kfr = pin.computeFrameKinematicRegressor(robot.model,
                                             robot.data, ID_e, pin.LOCAL_WORLD_ALIGNED)
    for j in range(6):
        J[nsample*j + i, :] = fj[j, :]
        R[nsample*j + i, :] = kfr[j, :]

plot1 = plt.figure()
axs = plot1.subplots(6, 1)
ylabel = ["px", "py", "pz", "phix", "phiy", "phiz"]
for j in range(6):
    axs[j].plot(R[nsample*j:nsample*j+nsample, 17])
    axs[j].set_ylabel(ylabel[j])

###########FrameJacobian###########################
# eliminate zero columns
col_norm = np.diag(np.dot(J.T, J))

joint_offe = []
joint_offr = []
for i in range(col_norm.shape[0]):
    if col_norm[i] < 1e-6:
        joint_offe.append(i)
    else:
        joint_offr.append(joint_off[i])
J_e = np.delete(J, joint_offe, 1)

# apply qr decomposition to find independent columns and regrouping
q, r = np.linalg.qr(J_e)

idx_base = []
idx_regroup = []

for i in range(r.shape[1]):
    if abs(np.diag(r)[i]) > 1e-6:
        idx_base.append(i)
    else:
        idx_regroup.append(i)


# identifiable parameters expression


##############FrameKinematicRegressor########################
# select columns correspond to joint offset
joint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
             50, 51, 52, 53]  # all on z axis - checked!!
R_sel = R[:, joint_idx]
geo_params_sel = [geo_params[i] for i in joint_idx]
print(R_sel.shape)
# eliminate zero columns
col_norm = np.diag(np.dot(R_sel.T, R_sel))

geo_paramse = []
geo_paramsr = []
for i in range(col_norm.shape[0]):
    if col_norm[i] < 1e-2:
        geo_paramse.append(i)
    else:
        geo_paramsr.append(geo_params_sel[i])
# R_e = np.delete(R_sel, geo_paramse, 1)
geo_paramsr = geo_params_sel
R_e = R_sel
print(R_e.shape)

# apply qr decomposition to find independent columns and regrouping
q, r = np.linalg.qr(R_e)

idx_base = []
idx_regroup = []

for i in range(r.shape[1]):
    if abs(np.diag(r)[i]) > 1e-6:
        idx_base.append(i)
    else:
        idx_regroup.append(i)
rank_Re = len(idx_base)
print(rank_Re)


# identifiable parameters expression
R1 = np.zeros([R_e.shape[0], len(idx_base)])
R2 = np.zeros([R_e.shape[0], len(idx_regroup)])
params_base = []
params_regroup = []
for i in range(len(idx_base)):
    R1[:, i] = R_e[:, idx_base[i]]
    params_base.append(geo_paramsr[idx_base[i]])
for j in range(len(idx_regroup)):
    R2[:, j] = R_e[:, idx_regroup[j]]
    params_regroup.append(geo_paramsr[idx_regroup[j]])
new_Re = np.c_[R1, R2]
print(new_Re.shape)
# second time qr decomposition

new_q, new_r = np.linalg.qr(new_Re)
new_R1 = new_r[0: rank_Re, 0: rank_Re]
new_Q1 = new_q[:, 0: rank_Re]
new_R2 = new_r[0: rank_Re, rank_Re: new_r.shape[1]]

new_Rbase = np.dot(new_Q1, new_R1)
print("condition number: ", np.linalg.cond(new_Rbase))

beta = np.around(np.dot(np.linalg.inv(new_R1), new_R2), 6)

tol_beta = 1e-10

for i in range(rank_Re):
    for j in range(beta.shape[1]):
        if abs(beta[i, j]) < tol_beta:

            params_base[i] = params_base[i]

        elif beta[i, j] < -tol_beta:

            params_base[i] = (
                params_base[i]
                + " - "
                + str(abs(beta[i, j]))
                + "*"
                + str(params_regroup[j])
            )

        else:

            params_base[i] = (
                params_base[i]
                + " + "
                + str(abs(beta[i, j]))
                + "*"
                + str(params_regroup[j])
            )
print(params_base)
plt.show()


# add a marker at the ee
# name = "ee_marker_frame"
# parent_joint = robot.model.getJointId("arm_7_joint")
# prev_frame = robot.model.getFrameId("arm_7_joint")
# placement = pin.SE3(eye(3), zero(3) + 0.1)
# inertia = pin.Inertia.Zero()
# frame = pin.Frame(name, parent_joint, prev_frame,
#                   placement, pin.FIXED_JOINT, inertia)
# robot.model.addFrame(frame)
# jointName = "ee_marker_joint"
# jointPlacement = pin.SE3(eye(3), zero(3) + 0.1)
# jointId = robot.model.getJointId("arm_7_joint")
# robot.model.addJoint(jointId, pin.JointModelRZ(),
#                      jointPlacement, jointName)
# # inertia = pin.Inertia.Zero()
# # robot.model.appendBodyToJoint(jointId, inertia, pin.SE3.Identity())
# robot.model.lowerPositionLimit[14] = -1.5
# robot.model.upperPositionLimit[14] = 1.5
# robot.model.velocityLimit[12] = 1.5
# robot.model.effortLimit[12] = 1.5
