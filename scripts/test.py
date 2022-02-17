import numpy as np
import pandas as pd

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *

from sys import argv
import os
from os.path import dirname, join, abspath


from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num

from tiago_mocap_calib_fun_def import (
    # get_param,
    get_PEE_fullvar,
    get_PEE_var,
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
    "talos_reduced.urdf"
    # "tiago_description/robots",
    # "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data
# print(model)
# 1/ model explore
# for i in range(model.njoints):
#     print(model.name)
#     print(model.names[i])
#     print(model.joints[i].id)
#     print(model.joints[i])
#     print(model.jointPlacements[i])

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


def get_param(robot, NbSample, TOOL_NAME='ee_marker_joint', NbMarkers=1,  calib_model='full_params', calib_idx=3):
    tool_FrameId = robot.model.getFrameId(TOOL_NAME)
    parentJoint2Tool_Id = robot.model.frames[tool_FrameId].parent
    # NbJoint = parentJoint2Tool_Id  # joint #0 is  universe
    root_joint = 13
    NbJoint = parentJoint2Tool_Id - root_joint + 1
    print("number of active joint: ", NbJoint)
    print("tool name: ", TOOL_NAME)
    print("parent joint of tool frame: ",
          robot.model.names[parentJoint2Tool_Id])
    print("number of markers: ", NbMarkers)
    print("calibration model: ", calib_model)
    param = {
        'q0': np.array(robot.q0),
        'x_opt_prev': np.zeros([NbJoint]),
        'NbSample': NbSample,
        'IDX_TOOL': tool_FrameId,
        'eps': 1e-3,
        'Ind_joint': np.array(range(root_joint-1, parentJoint2Tool_Id)),
        'PLOT': 0,
        'NbMarkers': NbMarkers,
        'calib_model': calib_model,  # 'joint_offset' / 'full_params'
        'calibration_index': calib_idx,  # 3 / 6
        'NbJoint': NbJoint
    }
    return param


# 3/ test base parameters calculation
# param = get_param(robot, 50, 'arm_left_1_link')
# q = []
# Rrand_b, R_b, params_base, params_e = Calculate_base_kinematics_regressor(
#     q, model, data, param)
# print("condition number: ", cond_num(R_b), cond_num(Rrand_b))

# print("%d base parameters: " % len(params_base), params_base)

# 4/ test extract quaternion data to rpy


x1 = [1., 1., 1., -0.7536391, -0.0753639, -0.1507278, -0.6353185]
x2 = [2., 2., 2., 0, 0, 0, 1]

se1 = pin.XYZQUATToSE3(x1)
se2 = pin.XYZQUATToSE3(x2)
se12 = pin.SE3.inverse(se1)*se2

print(se12.translation)
