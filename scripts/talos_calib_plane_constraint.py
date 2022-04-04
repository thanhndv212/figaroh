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

from tiago_mocap_calib_fun_def import (
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
robot = Robot(
    "talos_data/robots",
    "talos_reduced.urdf"
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

# target frame
tool_name = 'head_2_joint'
target_frameId = model.getFrameId(tool_name)
target_frame = model.frames[target_frameId]

joint_parent = target_frame.parent

q_rand = pin.randomConfiguration(model)

pin.forwardKinematics(model, data, robot.q0)
pin.updateFramePlacements(model, data)
# tranformation from base link to joint placement
oMp = data.oMi[14]*model.jointPlacements[15]

# transformation from base link to target frame
print(data.oMi[target_frame.parent])
print(data.oMf[target_frameId])

R = pin.computeFrameKinematicRegressor(
    model, data, target_frameId, pin.LOCAL)
print(R[:, -48:-23])


# create transformation chain from world frame to the gripper

# data.oMf(): give transformation from base link frame -> target frame

# kinematic chain: world_frame -> feet-baselink -> baselink-wrist -> wrist-gripper
# SE3(world_frame)*inv(data.oMf(leg_left_7))*data.oMf(arm_left_7)*SE3(gripper)

# to justify the observability: derive regressor matrix of above transformation matrix
# how?
