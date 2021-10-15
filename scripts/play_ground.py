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
    CIK_problem)
from tiago_simplified import check_tiago_autocollision
from meshcat_viewer_wrapper import MeshcatVisualizer

robot = Robot(
    # "tiago_description/robots",
    # "tiago_no_hand_mod.urdf"
    "ur_description/urdf",
    "ur10_robot.urdf"
)

data = robot.model.createData()
model = robot.model

IDX_TOOL = model.getFrameId("ee_marker_joint")
NbSample = 100
param = get_param(robot, NbSample)
param['IDX_TOOL'] = model.getFrameId('ee_link')


# calcualte base regressor of kinematic errors model and the base parameters expressions
q = []
Rrand_b, R_b, params_base = Calculate_base_kinematics_regressor(
    q, model, data, param)
# condition number
print("condition number: ", cond_num(R_b), cond_num(Rrand_b))
print(params_base)

text_file = join(
    dirname(dirname(str(abspath(__file__)))),
    f"data/tiago_full_calib_BP.txt")
with open(text_file, 'w') as out:
    for n in params_base:
        out.write(n + '\n')
print(model)
