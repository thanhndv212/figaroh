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

print(len(model.lowerPositionLimit.tolist()))
# for i in range(model.njoints):
#     print(model.name)
#     print(model.names[i])
#     print(model.joints[i].id)
#     print(model.joints[i])
#     print(model.jointPlacements[i])
# given the tool_frame ->
# find parent joint (tool_joint) ->
# find root-tool kinematic chain
# eliminate universe joint,
# output list of joint idx
# output list of joint configuration idx

tool_name = 'gripper_left_base_link'
tool_joint = model.frames[model.getFrameId(tool_name)].parent
actJoint_idx = model.supports[tool_joint].tolist()[1:]
Ind_joint = [model.joints[i].idx_q for i in actJoint_idx]
