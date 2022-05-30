import time
prev_time = time.time()
import sys

sys.path.append('/home/thanhndv212/miniconda3/lib/python3.8/site-packages')
sys.path.remove('/opt/openrobots/lib/python3.6/site-packages')

from os.path import abspath, dirname, join
from ast import Sub
import time
import os
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from calibration_tools import (Calculate_base_kinematics_regressor,
                               Calculate_identifiable_kinematics_model,
                               Calculate_kinematics_model, CIK_problem,
                               cartesian_to_SE3, extract_expData4Mkr,
                               get_geoOffset, get_jointOffset, get_param,
                               get_PEE, get_PEE_fullvar, get_PEE_var)
from meshcat_viewer_wrapper import MeshcatVisualizer
from tools.qrdecomposition import cond_num, get_baseParams
from tools.regressor import eliminate_non_dynaffect
from tools.robot import Robot


# robot = Robot(
#     "talos_data/robots",
#     "talos_reduced.urdf",
#     # "tiago_description/robots",
#     # "tiago_no_hand_mod.urdf",
#     # "canopies_description/robots",
#     # "canopies_arm.urdf",
#     isFext=True  # add free-flyer joint at base
# )
# model = robot.model
# data = robot.data
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
# """ check if base parameters calculation is correct
# """
# NbSample = 50
# param = get_param(robot, NbSample,
#                   TOOL_NAME='gripper_left_fingertip_1_link', NbMarkers=1, free_flyer=True)


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
    config_quat = pin.SE3ToXYZQUAT(config_SE3)
    return config_quat


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
# """ check order in quaternion convention, i.e. Pinocchio: scalar w last
# """
# x1 = [1., 1., 1., -0.7536391, -0.0753639, -0.1507278, -0.6353185]
# x2 = [2., 2., 2., 0, 0, 0, 1]

# se1 = pin.XYZQUATToSE3(x1)
# se2 = pin.XYZQUATToSE3(x2)
# se12 = pin.SE3.inverse(se1)*se2
# se12_actinv = se1.actInv(se2)
# print(se12.action)
# print(se12.toDualActionMatrix())


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


# param_1 = get_param(robot, NbSample,
#                     TOOL_NAME='gripper_left_fingertip_1_link', NbMarkers=1, free_flyer=True)
# Rrand_b_1, R_b_1, R_e_1,  params_base_1, params_e_1 = Calculate_base_kinematics_regressor(
#     q, model, data, param_1)
# for i in range(6):
#     print(params_base[i])
#     print(params_base_1[i])
# print(np.array_equal(R_b[:, 0:3], R_b_1[:, 0:3]))

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


def get_random_reg(robot, NbSample):
    param = get_param(robot, NbSample,
                    TOOL_NAME='gripper_left_fingertip_1_link', NbMarkers=1, free_flyer=False)

    # create random data with or without freeflyer root_joint
    q = np.empty((NbSample, robot.q0.shape[0]))
    # range for free flyer joint
    trans_range = [0.1, 0.5]
    orient_range = [-np.pi/2, np.pi/2]
    for it in range(NbSample):
        q_i = pin.randomConfiguration(model)
        if param['free_flyer']:
            q_i[:7] = random_freeflyer_config(trans_range, orient_range)
            q[it, :] = np.copy(q_i)
        else:
            q[it, :] = np.copy(q_i)

    Rrand_b, R_b, R_e, params_base, params_e = Calculate_base_kinematics_regressor(
        q, model, data, param)
    return  R_b, NbSample

# rearrange base regressor


def rearrange_rb(R_b, NbSample):
    Rb_rearr = np.empty_like(R_b)
    for i in range(3):
        for j in range(NbSample):
            Rb_rearr[j*3 + i, :] = R_b[i*NbSample + j]
    return Rb_rearr


def filter_matrix(var, NbSample, NbChosen):
    """ var is 1D array contain 0  or 1, sum = number of chosen samples
        this function transform var into a filter matrix that filter out the
        unchosen samples in information matrix
    """
    M = np.zeros((3*NbChosen, 3*NbSample))
    idx = 0
    idx_list = []
    for b in var:
        if b:
            idx_list.append(idx)
        idx += 1
    for it in range(NbChosen):
        start_row = 3*it
        start_col = idx_list[it]*3
        M[start_row:(start_row+3), start_col:(start_col+3)] = np.eye(3)
    return M


def sub_info_matrix(R, NbSample):
    "Returns a list of sub regressors which corresponds to each data point"
    subX_list = []
    for it in range(NbSample):
        subX = np.matmul(R[it*3:(it*3+3), :].T, R[it*3:(it*3+3), :])
        subX_list.append(subX)
    return subX_list 


def chosen_info_matrix(R, var):
    # assert cp.sum(var) == NbChosen, "Invalid variable!"
    full_X = np.zeros_like(np.matmul(R.T, R))
    count = 0
    eps = 1e-2
    for it in range(var.size):
        if var[it] != 0:
            count += 1
            full_X = full_X + 1 * \
                np.matmul(R[it*3:(it*3+3), :].T, R[it*3:(it*3+3), :])
        else:
            full_X = full_X
    print(count)
    return full_X

# investigate log_det vs NbSample

# NbSample_list = 20*np.array(list(range(1,250 )))
# print(NbSample_list)
# log_det = []

# for nb in NbSample_list:
#     R_b, NbSample = get_random_reg(robot, nb)

#     R_rearr = rearrange_rb(R_b, NbSample)
#     _, u, _ = np.linalg.svd(R_rearr)
#     # print("singular values: ", u)
#     # information matrix
#     subX_list = sub_info_matrix(R_rearr, NbSample)
#     # optimization with scipy 

#     # a = np.random.choice([0,1], size=(NbSample,), p=[1/2, 1/2])
#     a = np.full((NbSample, ), 1)
#     aggrX = 0
#     for i in range(NbSample):
#         aggrX += a[i]*subX_list[i]

#     log_detX = np.log(np.linalg.det(aggrX))
#     log_det.append(log_detX)
# plt.plot(NbSample_list, log_det)
# plt.show()

# picos optimization ( A-optimality, C-optimality, D-optimality)

import cvxopt as cvx
import picos as pc 

NbSample = 30
R_b, NbSample = get_random_reg(robot, NbSample)
R_rearr = rearrange_rb(R_b, NbSample)
subX_list = sub_info_matrix(R_rearr, NbSample)

# # transform to cvx matrix-type data
# M = []
# for i in range(len(subX_list)):
#     M.append(cvx.matrix(subX_list[i].tolist()))

# create problem
D_MAXDET = pc.Problem()
# decision problem
w = pc.RealVariable('w', NbSample, lower=0)
t = pc.RealVariable('t', 1)

# constraints
Mw = pc.sum(w[i]*subX_list[i] for i in range(NbSample))
wgt_cons = D_MAXDET.add_constraint(1|w <= 1)
det_root_cons = D_MAXDET.add_constraint(t <= pc.DetRootN(Mw))

# objective
D_MAXDET.set_objective('max', t)
print(D_MAXDET)
# solution
solution = D_MAXDET.solve(solver='cvxopt')
print(solution.problemStatus)
print(solution.info)
solve_time = time.time() - prev_time
print('solve time: ', solve_time)

# solution re-arrangement
# list 
w_list =[]
for i in range(w.dim):
    w_list.append(float(w.value[i]))
# dict
w_dict = {}
for i, w_i in enumerate(w_list): 
    w_dict[i] = w_i
# sort dict
w_dict_sort = dict(reversed(sorted(w_dict.items(), key=lambda item: item[1])))

# eps = 1e-5
# for i, k in enumerate(w_dict_sort): 
#     print(k)
#     if w_dict_sort[k] < eps:
#         max_NbChosen = i
#         break
max_NbChosen = NbSample
min_NbChosen = 25
if max_NbChosen < min_NbChosen:
    print("Infeasible design")
# evaluate NbChosen given a certain NbSample 
det_root_list = []
n_key_list = []
for nbc in range(min_NbChosen, NbSample+1):
    n_key = list(w_dict_sort.keys())[0:nbc]
    n_key_list.append(n_key)
    M_i = pc.sum(w_dict_sort[i]*subX_list[i] for i in n_key)
    det_root_list.append(pc.DetRootN(M_i))

# calculate detroot by a window of min_NbChosen elements on decreasing order
# det_root_list = []
# n_key_list = []
# for nbc in range(min_NbChosen, NbSample+1):
#     n_key = list(w_dict_sort.keys())[(nbc-min_NbChosen):nbc]
#     n_key_list.append(n_key)
#     M_i = pc.sum(subX_list[i] for i in n_key)
#     det_root_list.append(pc.DetRootN(M_i))

# calculate corresponding condition number 
print(R_rearr.shape)
cond_whole = np.linalg.cond(R_rearr)
M_whole = np.matmul(R_rearr.T, R_rearr)
det_root_whole = pc.DetRootN(M_whole)
# cond_list = []
# for n_key in n_key_list:
#     R_temp = np.copy(R_rearr)
#     for i in range(NbSample):
#         if i not in n_key:
#             R_temp = np.delete(R_temp, range(i*3,(i*3+3)), axis=0)
#     cond_list.append(np.linalg.cond(R_temp))
# print(cond_whole, cond_list)

# # select combination length min_NbChosen on NbSample from itertools 
# det_root_combi = []
# n_key_combi = []
# print('pass here')
# from itertools import combinations as combi 
# for nb in combi(list(w_dict_sort.keys()), min_NbChosen): 
#     n_key_combi.append(list(nb))
# for n_key in n_key_combi: 
#     M_i = pc.sum(subX_list[i] for i in n_key)
#     det_root_combi.append(pc.DetRootN(M_i))

# def find_index_sublist(list, sub_list):
#     idx_list = []
#     for idx, l_i in enumerate(list):
#         if l_i in sub_list:
#             idx_list.append(idx)

#     return idx_list
# idx_subList = find_index_sublist(n_key_combi, n_key_list)
idx_subList = range(len(det_root_list))
# NbSample_1 = 25  
# R_b_1, NbSample_1 = get_random_reg(robot, NbSample_1)
# R_rearr_1 = rearrange_rb(R_b_1, NbSample_1)
# M_whole_1 = np.matmul(R_rearr_1.T, R_rearr_1)
# det_root_whole_1 = pc.DetRootN(M_whole_1)
# plot
fig, ax = plt.subplots(2)
ratio = det_root_whole/det_root_list[-1]
# # evolution of detroot along decending order
color = 'tab:red'
ax[0].set_xlabel('Data point index')
ax[0].set_ylabel('m-th root of determinant of (mxm) information matrix', color=color)
ax[0].tick_params(axis='y', labelcolor=color)
ax[0].scatter(idx_subList, ratio*np.array(det_root_list), color=color)
ax[0].hlines(det_root_whole, min(idx_subList), max(idx_subList))
# ax[0].hlines(det_root_whole_1, min(idx_subList), max(idx_subList))

# # plot combinatrionarial det root 
# ax_1 = ax[0].twinx()
# color = 'tab:blue'
# ax_1.set_ylabel('m-th root of determinant of (mxm) information matrix', color=color)
# ax_1.tick_params(axis='y', labelcolor=color)
# ax_1.plot(range(len(det_root_combi)), det_root_combi, color=color)

# quality of estimation
color = 'tab:blue'
ax[1].set_ylabel('Quality of estimation per data point', color=color)  # we already handled the x-label with ax[0]
ax[1].tick_params(axis='y', labelcolor=color)
w_list.sort(reverse=True)
ax[1].plot(range(NbSample), w_list, color=color)


plt.show()
# cvxpy optimization problem formulization


# class MyProblem(param, NbChosen, R_rearr):
#     def __init__(self) -> None:
#         super().__init__()
#         self._var = cp.Variable(shape=(param['NbSample'],), integer=True)   
#         self._obj_fnc = 0
#         self._constraints = []
#         self._problem = cp.Problem(cp.Maximize(obj), constraints)


#     def solve(self):
#         self._problem.solve()
#         return self._problem.value, self._var.value


# var = []
# par= []
# print(subX_list[0].shape)
# for ii in range(NbSample):
#     var.append(cp.Variable(integer=True))
#     p = cp.Parameter((44, 44))
#     p.value = subX_list[ii]
#     par.append(p)

# constraints = []
# for i in range(len(var)):
#     constraints.append(var[i] >= 0)
#     constraints.append(var[i] <= 1)
# constraints.append(cp.sum(var)-NbChosen == 0)
# t = cp.Variable(1)

# aggr_X = var[0]*par[0]
# for i in range(1, NbSample):
#     sub_matmul = var[i]*par[i]
#     aggr_X += sub_matmul
# constraints.append(t <= cp.log_det(aggr_X))
# obj = cp.Maximize(t)
# prob = cp.Problem(obj, constraints)
# prob.solve(solver=cp.SCIP, ignore_dpp =True, verbose=True)
# print(prob.status)
# print(prob.value)
# sol = []
# for i in var:
#     sol.append(i.value)
#     print(i.value)
# print(len(var))
# print(sol)


# pyscipopt optimization problem formulization 
# from pyscipopt import Model
# model = Model()
# x = model.addVar("x", vtype ="I")
# model.addCons(x>=0)
# model.addCons(x<=1)
# var = []
# for i in range(0, NbSample):
#     var.append(model.addVar(vtype="I"))
#     model.addCons(var[i]<=1)
#     model.addCons(var[i]>=0)
# model.addCons(np.sum(var)-NbChosen ==0)

# def get_cost(var, subX_list):
#     aggr_X = var[0]*subX_list[0]
#     for i in range(1, NbSample):
#         sub_matmul = var[i]*subX_list[i]
#         aggr_X += sub_matmul

#     _, cost = np.linalg.slogdet(aggr_X)
#     return cost
# model.setObjective(-np.log(np.linalg.det(x*np.array([[1,1],[2,2]]))), 'miminize')


# model.optimize()
# sol = model.getBestSol()
# print("x: {}".format(sol[x]))
# print("y: {}".format(sol[y]))

