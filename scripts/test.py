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
    isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data
# # 1/ model explore
print(model)
for i in range(model.njoints):
    # print(model.name)
    print("joint name: ", model.names[i])
    print("joint id: ", model.joints[i].id)
    print("joint details: ", model.joints[i])
    print("joint placement: ",  model.jointPlacements[i])
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


# target_frameId = model.getFrameId("arm_right_7_link")
# pin.forwardKinematics(model, data, robot.q0)
# pin.updateFramePlacements(model, data)
# kin_reg = pin.computeFrameKinematicRegressor(
#     model, data, target_frameId, pin.LOCAL)
# print(kin_reg.shape, kin_reg)

# 2/ test param
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
NbSample = 50
param = get_param(robot, NbSample,
                  TOOL_NAME='right_sole_link', NbMarkers=1, free_flyer=True)


def random_freeflyer_config(trans_range, orient_range):
    "Output a vector of 7 tranlation + quaternion within range"
    # trans_range = trans_range.sort()
    # orient_range = orient_range.sort()
    config_rpy = []
    for itr in range(3):
        config_rpy.append(
            (trans_range[1]-trans_range[0])
            * np.random.random_sample() + trans_range[0])
    for ior in range(3):
        config_rpy.append((orient_range[1]-orient_range[0])
                          * np.random.random_sample() + orient_range[0])
    config_SE3 = cartesian_to_SE3(np.array(config_rpy))
    config_quat = pin.se3ToXYZQUAT(config_SE3)
    return config_quat


q = np.empty((NbSample, model.nq))

for it in range(NbSample):
    trans_range = [0.1, 0.5]
    orient_range = [-np.pi/2, np.pi/2]
    q_i = pin.randomConfiguration(model)
    q_i[:7] = random_freeflyer_config(trans_range, orient_range)
    q[it, :] = np.copy(q_i)
Rrand_b, R_b, Rrand_e, params_base, params_e = Calculate_base_kinematics_regressor(
    q, model, data, param)
_, s, _ = np.linalg.svd(Rrand_e)
for i, pr_e in enumerate(params_e):
    print(pr_e, s[i])
print("condition number: ", cond_num(R_b), cond_num(Rrand_b))

print("%d base parameters: " % len(params_base))
for enum, pb in enumerate(params_base):
    print(enum+1, pb)
# path = '/home/thanhndv212/Cooking/figaroh/data/tiago/tiago_nov_30_64.csv'
# PEEm_exp, q_exp = extract_expData4Mkr(path, param)

# print(PEEm_exp.shape)
# 4/ test extract quaternion data to rpy


# x1 = [1., 1., 1., -0.7536391, -0.0753639, -0.1507278, -0.6353185]
# x2 = [2., 2., 2., 0, 0, 0, 1]

# se1 = pin.XYZQUATToSE3(x1)
# se2 = pin.XYZQUATToSE3(x2)
# se12 = pin.SE3.inverse(se1)*se2

# print(se12.translation)
