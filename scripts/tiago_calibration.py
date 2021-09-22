import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *

from sys import argv
import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import csv

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num

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

# create a dictionary of geometric parameters errors
"""
kinematic tree:
1: base -> torso->arm
2: base -> torso -> head
3: base -> wheels
for now, we only care about branch 1
"""

joint_names = [name for i, name in enumerate(robot.model.names)]


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


# # read data from csv file
# folder = dirname(dirname(str(abspath(__file__))))
# file_name = join(folder, 'out_15.csv')
# q_sample = pd.read_csv(file_name).to_numpy()
# nsample = q_sample.shape[0]

geo_params = get_geoOffset(joint_names)
joint_off = get_jointOffset(joint_names)
# generate data points and regressors
ID_e = robot.model.getFrameId("ee_marker_joint")
nsample = 100

######################################
J = np.empty([6*nsample, robot.model.nv])
R = np.empty([6*nsample, 6*robot.model.nv])
for i in range(nsample):
    q = pin.randomConfiguration(robot.model)
    # q = q_sample[i, :]
    q[8:] = robot.q0[8:]
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    fj = pin.computeFrameJacobian(robot.model,
                                  robot.data, q, ID_e, pin.LOCAL_WORLD_ALIGNED)
    kfr = pin.computeFrameKinematicRegressor(robot.model,
                                             robot.data, ID_e, pin.LOCAL_WORLD_ALIGNED)
    for j in range(6):
        J[nsample*j + i, :] = fj[j, :]
        R[nsample*j + i, :] = kfr[j, :]

# plotting
plot1 = plt.figure()
axs = plot1.subplots(6, 1)
ylabel = ["px", "py", "pz", "phix", "phiy", "phiz"]
for j in range(6):
    axs[j].plot(R[nsample*j:nsample*j+nsample, 17])
    axs[j].set_ylabel(ylabel[j])
# plt.show()

###########FrameJacobian###########################
# eliminate zero columns
J_e, params_eJ = eliminate_non_dynaffect(J, joint_off, tol_e=1e-6)

# get base parameters
J_b, params_baseJ = get_baseParams(J_e, params_eJ)

print(params_baseJ)
###########FrameKinematicRegressor#################
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
R_e, params_eR = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)

# get base parameters
R_b, params_baseR = get_baseParams(R_e, params_eR)
print("base parameters: ", (params_baseR))
print("condition number: ", cond_num(R_b))


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
