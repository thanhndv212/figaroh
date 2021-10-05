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

from tiago_mocap_calib_fun_def import *
from tiago_simplified import check_tiago_autocollision
from meshcat_viewer_wrapper import MeshcatVisualizer

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
    Nq = 8  # number of joints to be optimized

    # load robot
    robot = Robot(
        "tiago_description/robots",
        "tiago_no_hand_mod.urdf"
    )

    data = robot.model.createData()
    model = robot.model

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
    }

    # Generate feasible joint configuration
    cube_pose = [0.65, 0.0, 0.7]  # position of the cube
    cube_pose[len(cube_pose):] = [0, 0, 0, 1]  # orientation of the cube
    cube_dim = [0.4, 0.5, 0.4]

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

    # # visualize
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.scatter3D(PEEd_2d[0, :], PEEd_2d[1, :], PEEd_2d[2, :], color="green")
    plt.title("simple 3D scatter plot")
    # plt.show()

    PEEd = PEEd_2d.flatten('C')
    param['PEEd'] = PEEd
    param['eps_gradient'] = 1e-6

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
    q_list = []
    for iter in range(NbSample):

        param['iter'] = iter+1

    # constraint bounds -> c == 0 needed -> both bounds = 0
        cl = []
        cu = []

        x_opt = np.zeros([8])

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
            q_list.append(np.around(x_opt, 4).tolist())
        else:
            print("Iter {} Desired end-effector position: {} ".format(iter+1, PEEd_iter))
            print("Iter {} Achieved end-effector position: {} ".format(iter+1, PEEe))


# save designed configs to txt file
    # dt = datetime.now()
    # current_time = dt.strftime("%d_%b_%Y_%H%M")
    # text_file = join(
    #     dirname(dirname(str(abspath(__file__)))),
    #     f"data/tiago_calib_exp_{current_time}.txt")
    # json.dump(q_list, open(text_file, "w"))

# PLEASE THANH find a way to put this in a function

    q = np.reshape(q, (NbSample, model.nq), order='C')
    # robot.initViewer(loadModel=True)
    # gui = robot.viewer.gui

    # for iter in range(NbSample):

    #     #pin.forwardKinematics(model,  data, q[iter,:])
    #     #pin.updateFramePlacements(model,  data)

    #     display(robot,model, q[iter,:])

    #     gui.addBox("world/box_1",cube_dim[0], cube_dim[1],cube_dim[2],[0, 1, 0, 0.4])
    #     corner_cube=[cube_pose[0],cube_pose[1],cube_pose[2]]
    #     corner_cube[len(corner_cube):] = [0, 0, 0, 1]
    #     gui.applyConfiguration("world/box_1",cube_pose)

    #     param['iter'] = iter+1
    #     PEEd_iter = PEEd[[param['iter']-1, NbSample +
    #                       param['iter']-1, 2*NbSample+param['iter']-1]]

    #     gui.addSphere("world/sph_1", 0.02, [1., 0., 0., 0.75])
    #     gui.applyConfiguration("world/sph_1", [PEEd_iter[0],PEEd_iter[1],PEEd_iter[2]] + [0, 0, 0, 1])

    #     #print(iter)
    #     #print(PEEd_iter)

    #     gui.refresh()
    #     programPause = input("Press the <ENTER> to continue to next posture...")


##############
    # calcualte base regressor of kinematic errors model and the base parameters expressions
    R_b, params_base = Calculate_base_kinematics_regressor(
        q, model, data, param)
    # condition number
    cond_R = cond_num(R_b)
    print("condition number: ", cond_R)
    print(params_base)

# check autocollision and display
    check_tiago_autocollision(robot, q)
    # display few configurations
    viz = MeshcatVisualizer(
        model=robot.model, collision_model=robot.collision_model, visual_model=robot.visual_model, url='classical'
    )
    time.sleep(3)
    for i in range(20):
        viz.display(q[i, :])
        time.sleep(2)


if __name__ == "__main__":
    main()
