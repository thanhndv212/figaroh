import cvxpy as cp
import numpy as np
import pandas as pd
import time

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *

from sys import argv
import os
from os.path import dirname, join, abspath


from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num
from meshcat_viewer_wrapper import MeshcatVisualizer

from calibration_tools import (
    get_param,
    get_PEE_fullvar,
    get_PEE_var,
    extract_expData4Mkr,
    get_geoOffset,
    get_jointOffset,
    get_PEE,
    Calculate_kinematics_model,
    Calculate_identifiable_kinematics_model,
    Calculate_base_kinematics_regressor,
    cartesian_to_SE3,
    CIK_problem)

# path = '/home/thanhndv212/Cooking/figaroh/data/talos/talos_feb_arm_02_10_contact.csv'
# df = pd.read_csv(path)
# print(type(df))
# print(type(df[['x1']]))

robot = Robot(
    "talos_data/robots",
    "talos_reduced.urdf",
    # "tiago_description/robots",
    # "tiago_no_hand_mod.urdf",
    # "canopies_description/robots",
    # "canopies_arm.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data
# # 1/ model explore
""" Explore the model's attributes
"""
# print(model)
# for i in range(model.njoints):
#     # print(model.name)
#     print("joint name: ", model.names[i])
#     print("joint id: ", model.joints[i].id)
#     print("joint details: ", model.joints[i])
#     print("joint placement: ",  model.jointPlacements[i])
# for i, frame in enumerate(model.frames):
#     print(frame)
# viz = MeshcatVisualizer(
#     model=robot.model, collision_model=robot.collision_model,
#     visual_model=robot.visual_model, url='classical'
# )
# for i in range(20):
#     q = pin.randomConfiguration(robot.model)
#     viz.display(robot.q0)
#     time.sleep(1)


# target_frameId = model.getFrameId("gripper_left_fingertip_1_link")
# pin.forwardKinematics(model, data, pin.randomConfiguration(model))
# pin.updateFramePlacements(model, data)
# kin_reg = pin.computeFrameKinematicRegressor(
#     model, data, target_frameId, pin.LOCAL)
# print(kin_reg.shape, kin_reg)

# 2/ test param
""" check names, IDs given a sub-tree that supports the tool.
"""
# # given the tool_frame ->
# # find parent joint (tool_joint) ->
# # find root-tool kinematic chain
# # eliminate universe joint,
# # output list of joint idx
# # output list of joint configuration idx
# tool_name = 'gripper_left_base_link'
# tool_joint = model.frames[model.getFrameId(tool_name)].parent
# actJoint_idx = model.supports[tool_joint].tolist()[1:]
# Ind_joint = [model.joints[i].idx_q for i in actJoint_idx]


# 3/ test base parameters calculation
""" check if base parameters calculation is correct
"""
NbSample = 50
param = get_param(robot, NbSample,
                  TOOL_NAME='gripper_left_fingertip_1_link', NbMarkers=1, free_flyer=True)


# def random_freeflyer_config(trans_range, orient_range):
#     "Output a vector of 7 tranlation + quaternion within range"
#     # trans_range = trans_range.sort()
#     # orient_range = orient_range.sort()
#     config_rpy = []
#     for itr in range(3):
#         config_rpy.append(
#             (trans_range[1]-trans_range[0])
#             * np.random.random_sample() + trans_range[0])
#     for ior in range(3):
#         config_rpy.append((orient_range[1]-orient_range[0])
#                           * np.random.random_sample() + orient_range[0])
#     config_SE3 = cartesian_to_SE3(np.array(config_rpy))
#     config_quat = pin.se3ToXYZQUAT(config_SE3)
#     return config_quat


# q = np.empty((NbSample, model.nq))

# for it in range(NbSample):
#     trans_range = [0.1, 0.5]
#     orient_range = [-np.pi/2, np.pi/2]
#     q_i = pin.randomConfiguration(model)
#     q_i[:7] = random_freeflyer_config(trans_range, orient_range)
#     q[it, :] = np.copy(q_i)
# Rrand_b, R_b, Rrand_e, params_base, params_e = Calculate_base_kinematics_regressor(
#     q, model, data, param)
# _, s, _ = np.linalg.svd(Rrand_e)
# for i, pr_e in enumerate(params_e):
#     print(pr_e, s[i])
# print("condition number: ", cond_num(R_b), cond_num(Rrand_b))

# print("%d base parameters: " % len(params_base))
# for enum, pb in enumerate(params_base):
#     print( pb)
# path = '/home/thanhndv212/Cooking/figaroh/data/tiago/tiago_nov_30_64.csv'
# PEEm_exp, q_exp = extract_expData4Mkr(path, param)

# print(PEEm_exp.shape)
# 4/ test extract quaternion data to rpy
""" check order in quaternion convention, i.e. Pinocchio: scalar w last
"""
# x1 = [1., 1., 1., -0.7536391, -0.0753639, -0.1507278, -0.6353185]
# x2 = [2., 2., 2., 0, 0, 0, 1]

# se1 = pin.XYZQUATToSE3(x1)
# se2 = pin.XYZQUATToSE3(x2)
# se12 = pin.SE3.inverse(se1)*se2

# print(se12.translation)


# 5/ concatenating multi-subtree kinematic regressors
""" Pinocchio only provides computation of kinematic regressor for a kinematic sub-tree 
with regard to the root joint. Therefore, for structures like humanoid, in order to compute
a kinematic regressor of 2 or more serial subtrees, we need to concatenate 2 or more kinematic
regressors.
"""
# TODO:
# given a set of configurations:
# 1/ compute kinematic regressor from root_joint to tool_link
# 2/ compute kinematic regressor from root_joint to foot_sole_link
# 3/ re-arrange and concatenate 2 matrix in correct manner
# 4/ remove zero columns
# 5/ perform QR decomp to find identifiable parameters a.k.a base parameters



def get_info_matrix(q, base_index, model, data, param):
    # compute kinematic regressor given a config
    # remove rows
    # remove cols
    # stor in list
    pass
