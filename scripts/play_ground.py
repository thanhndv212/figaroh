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

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import cyipopt

import pandas as pd
import csv
import json
import time

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num

from tiago_mocap_calib_fun_def import (
    get_param,
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
from tiago_simplified import check_tiago_autocollision
from meshcat_viewer_wrapper import MeshcatVisualizer


# 1/ Load robot model and call a dictionary containing reserved constants
robot = Robot(
    # "talos_data/robots",
    # "talos_reduced.urdf"
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data
print(model)
NbSample = 50
# param = get_param(
#     robot, NbSample, TOOL_NAME='gripper_left_joint', NbMarkers=1)
param = get_param(
    robot, NbSample, TOOL_NAME='ee_marker_joint', NbMarkers=1)

# 2/ Base parameters calculation
q_rand = []
Rrand_b, R_b, params_base, params_e = Calculate_base_kinematics_regressor(
    q_rand, model, data, param)
print("condition number: ", cond_num(R_b), cond_num(Rrand_b))

print("reduced parameters: ", params_e)

print("%d base parameters: " % len(params_base), params_base)


# # config = pin.randomConfiguration(model)
# config = robot.q0

# pin.framesForwardKinematics(model, data, config)
# pin.updateFramePlacements(model, data)

# # calculate oMf from the 1st join tto last joint (wrist)
# lastJoint_name = model.names[param['NbJoint']]
# lastJoint_frameId = model.getFrameId(lastJoint_name)
# inter_placements = data.oMf[lastJoint_frameId]

# last_frame = np.array([0, 0, 0, np.pi/4, np.pi/4, 0])

# last_placement = cartesian_to_SE3(last_frame)
# new_placement = inter_placements*last_placement
# print(new_placement)

# inter_placements.translation += last_frame[0:3]
# new_rpy = pin.rpy.matrixToRpy(inter_placements.rotation) + last_frame[3:6]
# inter_placements.rotation = pin.rpy.rpyToMatrix(new_rpy)
# print(inter_placements)

# new_inter = data.oMf[lastJoint_frameId]
# new_inter.translation += last_placement.translation
# new_inter.rotation += last_placement.rotation
# print(new_inter)
# print(model.getJointId('ee_marker_joint'))
# print(model)


# calcualte base regressor of kinematic errors model and the base parameters expressions
# q = []

# Rrand_b, R_b, params_base = Calculate_base_kinematics_regressor(
#     q, model, data, param)
# condition number
# print("condition number: ", cond_num(R_b), cond_num(Rrand_b))
# print(params_base)

# text_file = join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"data/talos_full_calib_BP.txt")
# with open(text_file, 'w') as out:
#     for n in params_base:
#         out.write(n + '\n')
