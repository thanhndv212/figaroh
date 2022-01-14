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

import pandas as pd
import csv
import json

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num

from tiago_mocap_calib_fun_def_VC import *

"""
# load robot

# create a dictionary of geometric parameters errors

# generate data points

# get jacobian + kinematic regressor => rearranging

# eliminate zero columns

# apply qr decomposition to find independent columns and regrouping

# identifiable parameters expressions
"""


def main():

    NbGrid = 3
    NbSample = pow(NbGrid, 3)
    # NbSample=2
    Nq = 8  # number of joints to be optimized

    # load robot
    robot = Robot(
        "tiago_description/robots",
        "tiago_no_hand_mod.urdf"
    )

    data = robot.model.createData()
    model = robot.model

    # create a dictionary of geometric parameters errors
    joint_names = []
    for i, name in enumerate(model.names):
        joint_names += [name]

    joint_names = joint_names[0:-4]  # remove the head and wheels joint

    tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]

    geo_params = joint_off = []
    for i in range(len(joint_names)):
        for j in tpl_names:
            geo_params.append(j + ("_%d" % i))
        joint_off.append("off" + "_%d" % i)
    phi_jo = [0] * len(joint_off)
    phi_gp = [0] * len(geo_params)

    joint_off = dict(zip(joint_off, phi_jo))
    geo_params = dict(zip(geo_params, phi_gp))

    IDX_TOOL = model.getFrameId("ee_marker_joint")

    # Use a dictionnary to store most of the non model related parameters
    # eps is the tolerance on the desired end effector 3D position accuracy
    # Ind_joint index of the joint that sould be modified for/by the calibration process
    param = {
        'q0': np.array(robot.q0),
        'x_opt_prev': np.zeros([8]),
        'IDX_TOOL': model.getFrameId('ee_marker_joint'),
        'NbSample': NbSample,
        'eps': 1e-3,
        'Ind_joint': np.arange(8),
        'PLOT': 0,
        'calibration_index': 6,
        'optim_IK': 1,
        'R_prev': []
    }

    # Get the index of the base regressor matrix
    param['NbSample'] = 100

    R_b, paramsrand_base, idx_base = Calculate_base_kinematics_regressor(
        [], model, data, param)

    param['NbSample'] = NbSample
    param['idx_base'] = idx_base
    print("This is index base printedsadkjsdbfjksdbfjksdbfjksbdjkf", idx_base)
    # Generate feasible joint configuration

    cube_pose = [0.55, 0.0, 0.5]  # position of the cube
    cube_pose[len(cube_pose):] = [0, 0, 0, 1]  # orientation of the cube
    cube_dim = [0.45, 0.5, 0.5]

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

    PEEd = PEEd_2d.flatten('C')
    param['PEEd'] = PEEd
    param['eps_gradient'] = 1e-6

    # First we will generate feasible solutions to serve as initial guess to the OCP generation process
    # set initial conditions and joint limits
    lb = ub = x0 = []
    for j in range(len(param['Ind_joint'])):
        for i in range(1):
            lb = np.append(lb, model.lowerPositionLimit[j])
            ub = np.append(ub, model.upperPositionLimit[j])
            x0 = np.append(
                x0, (model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)
    starttime = time.time()
    q = []
    q_list = x_opt_all = []
    for iter in range(NbSample):

        param['iter'] = iter+1

    # constraint bounds -> c == 0 needed -> both bounds = 0
        cl = []
        cu = []

        x_opt = np.zeros([8])

        nlp = cyipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=CIK_problem(data, model, param),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        nlp.addOption(b'hessian_approximation', b'limited-memory')
        nlp.addOption('max_iter', 2000)
        nlp.addOption('tol', 1e-6)  # Tolerance on the end-effector 3D position
        nlp.addOption('print_level', 1)

        # starttime = time.time()
        x_opt, info = nlp.solve(x0)
        # print('That took {} seconds'.format(time.time() - starttime))

        # save joint configuration to maximise distance with the next one
        param['x_opt_prev'] = x_opt

        q_opt = np.array(robot.q0)  # np.zeros(shape=(12, 1))

        PEEe = []
        for j in range(1):
            q_opt[0:8] = x_opt

            pin.forwardKinematics(model,  data, q_opt)
            pin.updateFramePlacements(model,  data)

            # Calculate_kinematics_model(q_opt, model, data, IDX_TOOL)
            PEEe = np.array(data.oMf[IDX_TOOL].translation)

        PEEd_iter = PEEd[[param['iter']-1, NbSample +
                          param['iter']-1, 2*NbSample+param['iter']-1]]

        J = np.sum(np.sqrt(np.square(PEEd_iter-PEEe)))/3
        if J <= 1e-3:
            print("Iter {} success ".format(iter+1))
            q = np.append(q, q_opt)
            x_opt_all = np.concatenate([x_opt_all, x_opt])

            q_list.append(np.around(x_opt, 4).tolist())
        else:
            print("Iter {} Desired end-effector position: {} ".format(iter+1, PEEd_iter))
            print("Iter {} Achieved end-effector position: {} ".format(iter+1, PEEe))
    R_b, paramsrand_base, idx_base = Calculate_base_kinematics_regressor(
        q, model, data, param)
    # condition number
    print("condition number of normal configs: ", cond_num(R_b))

    # OCP Generation process

    # set initial conditions and joint limits
    lb = ub = x0 = []
    for j in range(len(param['Ind_joint'])):
        for i in range(1):
            lb = np.append(lb, model.lowerPositionLimit[j])
            ub = np.append(ub, model.upperPositionLimit[j])
            x0 = np.append(
                x0, (model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)
    starttime = time.time()

    q = []
    q_list = x_opt_OCP = []

    # loop over a set of samples
    for iter in range(NbSample):

        param['iter'] = iter+1

    # use the IK feasible solution as initial guess
        x0 = x_opt_all[0+(param['iter']-1)*8:8+(param['iter']-1)*8]

    # constraint bounds -> c == 0 needed -> both bounds = 0
        cl = []
        cu = []

        # constraints on the end effector desired position
        cl = -1e-7*np.ones([3])
        cu = 1e-7*np.ones([3])

        x_opt = np.zeros([8])

        nlp = cyipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=OCP_determination_problem(data, model, param),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        nlp.addOption(b'hessian_approximation', b'limited-memory')
        nlp.addOption('max_iter', 2000)
        nlp.addOption('tol', 1e-6)  # Tolerance on the end-effector 3D position
        nlp.addOption('print_level', 1)

        # starttime = time.time()
        x_opt, info = nlp.solve(x0)
        # x_opt=x0
        # print('That took {} seconds'.format(time.time() - starttime))

        x_opt_OCP = np.concatenate([x_opt_OCP, x_opt])
        # x_opt[0+(param['iter']-1)*8:8+(param['iter']-1)*8]
        print(x_opt_OCP.shape)

        q_opt = np.array(robot.q0)  # np.zeros(shape=(12, 1))

        PEEe = []
        for j in range(1):
            q_opt[0:8] = x_opt

            model, data, R, J = Calculate_kinematics_model(
                q_opt, model, data, param['IDX_TOOL'])

            # dirty nothing to do here
            actJoint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
                            50, 51, 52, 53]  # all on z axis - checked!!

            # select columns corresponding to joint_idx
            R = R[:, actJoint_idx]
            # select form the list of columns given by the QR decomposition
            R = R[:, param['idx_base']]

            if param['iter'] > 1:
                param['R_prev'] = np.concatenate([param['R_prev'], R])
            else:
                param['R_prev'] = R

            # here we need to determine the worst posture we get to reoptimize it
            J_ind_max = ind_max = 1

            ind_max_prev = 1e3
            if param['iter'] > 3:  # we need at least 3 postures for this to make sense
                same_ind = True
                loop = 0
                while same_ind == True:

                    # find a posture that has maximum condition number of regressor
                    for ind in range(param['iter']):

                        lines_R = np.array(range(3*param['iter']))
                        ind_rm_R = (range(ind+2*(ind), 3+3*(ind)))
                        lines_rm = np.delete(lines_R, ind_rm_R)

                        R_ind = param['R_prev'][ind_rm_R, :]
                        J_ind = la.cond(R_ind)

                        if J_ind >= J_ind_max:
                            J_ind_max = J_ind
                            ind_max = ind

                            ind_max_prev = ind_max
                            #print("new index max:",ind_max)
                            print("new cond max:", J_ind)
                        if ind_max == ind_max_prev or loop > 10:
                            print("same index already optimized")
                            same_ind = False

                    # here we remove the lines of the regressor and optimal config (x_opt) vector corresponding to the worst posture
                    lines_R = np.array(range(3*param['iter']))
                    lines_q = np.array(range(8*param['iter']))

                    ind_rm_R = (range(ind_max+2*(ind_max), 3+3*(ind_max)))
                    ind_rm_q = (range(ind_max+7*(ind_max), 8+8*(ind_max)))

                    lines_rm_R = np.delete(lines_R, ind_rm_R)
                    lines_rm_q = np.delete(lines_q, ind_rm_q)

                    param['R_prev'] = param['R_prev'][lines_rm_R, :]

                    x_opt_OCP = x_opt_OCP[lines_rm_q]

                    # we reoptimise this posture
                    x_opt, info = nlp.solve(x0)
                    #x_opt= x0
                    x_opt_OCP = np.concatenate([x_opt_OCP, x_opt])

                    for j in range(1):
                        q_opt[0:8] = x_opt

                        model, data, R, J = Calculate_kinematics_model(
                            q_opt, model, data, param['IDX_TOOL'])

                        # dirty nothing to do here
                        actJoint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
                                        50, 51, 52, 53]  # all on z axis - checked!!

                        # select columns corresponding to joint_idx
                        R = R[:, actJoint_idx]
                        # select form the list of columns given by the QR decomposition
                        R = R[:, param['idx_base']]

                        if param['iter'] > 1:
                            param['R_prev'] = np.concatenate(
                                [param['R_prev'], R])
                        else:
                            param['R_prev'] = R
                loop = loop+1
            # Calculate_kinematics_model(q_opt, model, data, IDX_TOOL)
            PEEe = np.array(data.oMf[IDX_TOOL].translation)

        PEEd_iter = PEEd[[param['iter']-1, NbSample +
                          param['iter']-1, 2*NbSample+param['iter']-1]]

        J = np.sum(np.sqrt(np.square(PEEd_iter-PEEe)))/3
        if J <= 1e-3:
            print("Iter {} success ".format(iter+1))
            q = np.append(q, q_opt)
            q_list.append(np.around(x_opt, 4).tolist())
        else:
            print("Iter {} Desired end-effector position: {} ".format(iter+1, PEEd_iter))
            print("Iter {} Achieved end-effector position: {} ".format(iter+1, PEEe))

        J_cond = la.cond(param['R_prev'])
        print("Final regressor shape is:", np.array(param['R_prev']).shape)
        print('Final condition number is:', J_cond)


if __name__ == "__main__":
    main()
