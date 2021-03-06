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

from collision_model_simplified import (check_tiago_autocollision)

from calibration_tools import (
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
    "talos_data/robots",
    "talos_reduced.urdf"
    # "tiago_description/robots",
    # "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data
print(model)

NbGrid = 3
NbSample = pow(NbGrid, 3)
# op_arm_joint = 'arm_right_2_joint'
# op_value = -1.5
# rand_joints = [model.joints[model.getJointId('arm_left_5_joint')].idx_q,
#                model.joints[model.getJointId('arm_left_6_joint')].idx_q,
#                model.joints[model.getJointId('arm_left_7_joint')].idx_q]
# param = get_param(
#     robot, NbSample, TOOL_NAME='gripper_left_base_link', NbMarkers=1)
op_arm_joint = 'arm_left_2_joint'
op_value = 1.5
rand_joints = [model.joints[model.getJointId('arm_right_5_joint')].idx_q,
               model.joints[model.getJointId('arm_right_6_joint')].idx_q,
               model.joints[model.getJointId('arm_right_7_joint')].idx_q]
param = get_param(
    robot, NbSample, TOOL_NAME='arm_right_7_joint', NbMarkers=1)

print(param)
IDX_TOOL = param['IDX_TOOL']
# 2/ Generate cartesian poses
# cube for gripper base link
# cube_pose = [0.55, 0.0, 0.2]  # position of the cube
# cube_pose[len(cube_pose):] = [0, 0, 0, 1]  # orientation of the cube
# cube_dim = [0.5, 0.6, 0.6]

cube_pose = [0.55, 0.0, 0.2]  # position of the cube
cube_pose[len(cube_pose):] = [0, 0, 0, 1]  # orientation of the cube
cube_dim = [0.4, 0.4, 0.4]

PEEd_x = np.linspace(cube_pose[0] - cube_dim[0]/2,
                     cube_pose[0] + cube_dim[0]/2, NbGrid)
PEEd_y = np.linspace(cube_pose[1] - cube_dim[1]/2,
                     cube_pose[1] + cube_dim[1]/2, NbGrid)
PEEd_z = np.linspace(cube_pose[2] - cube_dim[2]/2,
                     cube_pose[2] + cube_dim[2]/2, NbGrid)

PEEd_2d = np.empty((3, NbSample))
for i in range(PEEd_x.shape[0]):
    for j in range(PEEd_y.shape[0]):
        for k in range(PEEd_z.shape[0]):
            idx = NbGrid*NbGrid*i + NbGrid*j + k
            PEEd_2d[:, idx] = np.array([PEEd_x[i], PEEd_y[j], PEEd_z[k]])

# uniformly distributed random posture


# 3/ Solve ipopt IK problem to generate joint configuration from given poses

    # desired poses
PEEd = PEEd_2d.flatten('C')

param['PEEd'] = PEEd
param['eps_gradient'] = 1e-6


# set initial conditions and joint limits
lb = ub = x0 = []
for j in param['Ind_joint']:
    # model.names does not accept index type of numpy int64
    # and model.lowerPositionLimit index lag to model.names by 1
    lb = np.append(lb, model.lowerPositionLimit[j])
    ub = np.append(ub, model.upperPositionLimit[j])
    x0 = np.append(
        x0, (model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)


q = []
q_list = []

print(robot.q0.shape)


for iter in range(NbSample):

    param['iter'] = iter+1

# constraint bounds -> c == 0 needed -> both bounds = 0
    cl = []
    cu = []

    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=CIK_problem(data, model, param),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    nlp.add_option(b'hessian_approximation', b'limited-memory')
    nlp.add_option('max_iter', 2000)
    # Tolerance on teh end-effector 3D position
    nlp.add_option('tol', 1e-6)
    # Tolerance on teh end-effector 3D position
    nlp.add_option('print_level', 1)

    # starttime = time.time()
    x_opt, info = nlp.solve(x0)
    # print('That took {} seconds'.format(time.time() - starttime))

    # save joint configuration to maximise distance with the next one
    param['x_opt_prev'] = x_opt

    q_opt = np.array(robot.q0)  # np.zeros(shape=(12, 1))

    PEEe = []
    for j in range(1):
        q_opt[param['Ind_joint']] = x_opt

        pin.forwardKinematics(model,  data, q_opt)
        pin.updateFramePlacements(model,  data)

        # Calculate_kinematics_model(q_opt, model, data, IDX_TOOL)
        PEEe = np.array(data.oMf[IDX_TOOL].translation)

    PEEd_iter = PEEd[[param['iter']-1, NbSample +
                      param['iter']-1, 2*NbSample+param['iter']-1]]

    J = np.sum(np.sqrt(np.square(PEEd_iter-PEEe)/3))
    if J <= 1e-3:
        print("Iter {} success ".format(iter+1))
        q = np.append(q, q_opt)
        q_list.append(np.around(x_opt, 4).tolist())
    else:
        print("Iter {} Desired end-effector position: {} ".format(iter+1, PEEd_iter))
        print("Iter {} Achieved end-effector position: {} ".format(iter+1, PEEe))


# 2/ Base parameters calculation
# q = []
q = np.reshape(q, (NbSample, model.nq), order='C')

# raise opposite arm to avoid collision in simulator
op_arm2 = model.joints[model.getJointId(op_arm_joint)].idx_q
q[:, op_arm2] = np.full(NbSample, op_value)
# print(q)


# randomize arm 5, 6, 7
for i in range(q.shape[0]):
    for j in rand_joints:
        limit_range = np.abs(
            model.upperPositionLimit[j] - model.lowerPositionLimit[j])
        print(limit_range)
        q[i, j] = np.random.uniform(
            model.lowerPositionLimit[j] + 0.05*limit_range, model.upperPositionLimit[j]-0.05*limit_range)

# compute base parameters
Rrand_b, R_b, params_base, params_e = Calculate_base_kinematics_regressor(
    q, model, data, param)
print("condition number: ", cond_num(R_b), cond_num(Rrand_b))
_, s1, _ = np.linalg.svd(Rrand_b)
print(pow(np.prod(s1), 1/38), s1)
_, s2, _ = np.linalg.svd(R_b)
print(pow(np.prod(s2), 1/38), s2)
# print("reduced parameters: ", params_e)

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

plt.show()


print("You have to start 'meshcat-server' in a terminal ...")
time.sleep(3)

# Tiago no hand
# urdf_dr = "tiago_description/robots"
# urdf_file = "tiago_no_hand_mod.urdf"
# srdf_dr = "/tiago_description/srdf/"
# srdf_file = "tiago.srdf"

# Talos reduced
urdf_dr = "talos_data/robots"
urdf_file = "talos_reduced.urdf"
srdf_dr = "/talos_data/srdf/"
srdf_file = "talos.srdf"

# robot = Robot(urdf_dr, urdf_file)

# q = np.empty((20, robot.q0.shape[0]))
# for i in range(20):
#     q[i, :] = pin.randomConfiguration(robot.model)
check_tiago_autocollision(robot, q, srdf_dr, srdf_file)

# display few configurations
viz = MeshcatVisualizer(
    model=robot.model, collision_model=robot.collision_model,
    visual_model=robot.visual_model, url='classical'
)
time.sleep(1)
for i in range(NbSample):
    viz.display(q[i, :])
    time.sleep(1)


# text_file = join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"data/talos/talos_full_calib_BP.txt")
# with open(text_file, 'w') as out:
#     for n in params_base:
#         out.write(n + '\n')

# # save designed configs to txt file
# dt = datetime.now()
# current_time = dt.strftime("%d_%b_%Y_%H%M")
# text_file = join(
#     dirname(dirname(str(abspath(__file__)))),
#     f"data/talos/talos_calib_right_exp_{current_time}.txt")
# json.dump(q_list, open(text_file, "w"))
