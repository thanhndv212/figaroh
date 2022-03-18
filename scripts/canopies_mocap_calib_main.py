from numpy.core.arrayprint import DatetimeFormat
from datetime import datetime
from numpy.core.fromnumeric import shape
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
# from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
from pinocchio.utils import *
# from pinocchio.pinocchio_pywrap import rpy
from meshcat_viewer_wrapper import MeshcatVisualizer
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

from tiago_simplified import (check_tiago_autocollision)

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
# from tiago_simplified import check_tiago_autocollision
# from meshcat_viewer_wrapper import MeshcatVisualizer


# 1/ Load robot model and call a dictionary containing reserved constants
robot = Robot(
    "canopies_description/robots",
    "canopies_arm.urdf"
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data

NbSample = 60
param = get_param(
    robot, NbSample, TOOL_NAME='arm_right_7_link', NbMarkers=1)

# 2/ Generate cartesian poses


# 3/ Solve ipopt IK problem to generate joint configuration from given poses

# 2/ Base parameters calculation
q_rand = np.empty((NbSample, robot.model.nq))
for i in range(NbSample):
    q_rand[i, :] = pin.randomConfiguration(robot.model)

# abnormal setup
joint_abm = 'arm_right_2_joint'
joint_abm_idx = model.joints[model.getJointId(joint_abm)].idx_q
q_rand[:, joint_abm_idx] = -1 * q_rand[:, joint_abm_idx]

collided_idx = check_tiago_autocollision(robot, q_rand)
q = np.delete(q_rand, collided_idx, 0)

param['NbSample'] = q.shape[0]


Rrand_b, R_b, params_base, params_e = Calculate_base_kinematics_regressor(
    q, model, data, param)
print("condition number: ", cond_num(R_b), cond_num(Rrand_b))
_, s1, _ = np.linalg.svd(Rrand_b)
print(pow(np.prod(s1), 1/38), s1)
_, s2, _ = np.linalg.svd(R_b)
print(pow(np.prod(s2), 1/38), s2)
print("reduced parameters: ", params_e)

print("%d base parameters: " % len(params_base))
for i in params_base:
    print(i)

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111, projection='3d')
lb = ub = []
for j in param['Ind_joint']:
    # model.names does not accept index type of numpy int64
    # and model.lowerPositionLimit index lag to model.names by 1
    lb = np.append(lb, model.lowerPositionLimit[j])
    ub = np.append(ub, model.upperPositionLimit[j])

q_actJoint = q[:, param['Ind_joint']]
sample_range = np.arange(param['NbSample'])

print(sample_range.shape)
print(len(param['actJoint_idx']))

for i in range(len(param['actJoint_idx'])):
    ax4.scatter3D(q_actJoint[:, i], sample_range, i)
for i in range(len(param['actJoint_idx'])):
    ax4.plot([lb[i], ub[i]], [sample_range[0],
             sample_range[0]], [i, i])

ax4.set_xlabel('Angle (rad)')
ax4.set_ylabel('Sample')
ax4.set_zlabel('Joint')

# plt.show()


print("You have to start 'meshcat-server' in a terminal ...")
time.sleep(3)

# display few configurations
viz = MeshcatVisualizer(
    model=robot.model, collision_model=robot.collision_model,
    visual_model=robot.visual_model, url='classical'
)
time.sleep(1)
# for i in range(param['NbSample']):
#     viz.display(q[i, :])
#     time.sleep(1)
q[0, joint_abm_idx] = -1.57
viz.display(q[0, :])
time.sleep(10)

# write generated configs to text file

# text_file = join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"data/canopies/canopies_full_calib_BP.txt")
# with open(text_file, 'w') as out:
#     for n in params_base:
#         out.write(n + '\n')
# q_write = np.around(q[:, param['Ind_joint']], 4).tolist()
# dt = datetime.now()
# current_time = dt.strftime("%d_%b_%Y_%H%M")
# text_file = join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"data/canopies/canopies_calib_right_arm_{current_time}.txt")
# json.dump(q_write, open(text_file, "w"))
