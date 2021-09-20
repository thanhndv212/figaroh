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

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num

from tiago_mocap_calib_fun_def import *

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

    NbGrid = 2
    NbSample = pow(NbGrid, 3)
    #NbSample=2
    Nq=8 #number of joints to be optimized


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
            joint_names+=[name]
    
    joint_names=joint_names[0:-4] # remove the head and wheels joint 

    tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]

    geo_params =joint_off= []
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
        'IDX_TOOL': model.getFrameId('ee_marker_joint'),
        'NbSample': NbSample,
        'eps': 1e-3,
        'Ind_joint': np.arange(8),
        'PLOT': 0,
    }
    '''
    
    # # read data from csv file
    # folder = dirname(dirname(str(abspath(__file__))))
    # file_name = join(folder, 'out_15.csv') 
    # q_sample = pd.read_csv(file_name).to_numpy()
    # NbSample = q_sample.shape[0]

    # generate data points and regressors
    
    
    R,J=Calculate_identifiable_kinematics_model([], model, data, param)


    ###########FrameKinematicRegressor#################
    # select columns correspond to joint offset
    joint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
             50, 51, 52, 53]  # all on z axis - checked!!

    # regressor matrix on selected paramters
    R_sel = R[:, joint_idx]

    # a dictionary of selected parameters
    gp_listItems = list(geo_params.items())
    geo_params_sel = []
    for i in joint_idx:
        geo_params_sel.append(gp_listItems[i])
    geo_params_sel = dict(geo_params_sel)

    # eliminate zero columns
    R_e, geo_paramsr = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)

    # get base parameters
    R_b, params_base = get_baseParams(R_e, geo_paramsr)
    print("base parameters: ", (params_base))
    print("condition number: ", cond_num(R_b))
    '''

    
    # Generate feasible joint configuration
    '''
    cube_pose=[0.5, 0.1, 0.5]# position of the cube
    cube_pose[len(cube_pose):] = [0, 0, 0, 1]# orientation of the cube
    cube_dim=[0.2, 0.2, 0.2]
    
    PEEd = np.linspace(cube_pose[0], cube_pose[0], NbSample)
    PEEd = np.append(PEEd, np.linspace(cube_pose[1], cube_pose[1], NbSample))
    PEEd = np.append(PEEd, np.linspace(
        cube_pose[2]+cube_dim[2]/2, cube_pose[2]+cube_dim[2]/2, NbSample))
 
    '''
   
    


   

    cube_pose = [0.5, 0.1, 0.5]  # position of the cube
    cube_pose[len(cube_pose):] = [0, 0, 0, 1]  # orientation of the cube
    cube_dim = [0.2, 0.2, 0.2]

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
    #plt.show()

    PEEd = PEEd_2d.flatten('C')
    print(PEEd)
    param['PEEd']= PEEd
    param['eps_gradient']=1e-6

    
    # set initial conditions and joint limits
    lb = ub = x0 = []
    for j in range(len(param['Ind_joint'])):
        for i in range(1):
            lb = np.append(lb, model.lowerPositionLimit[j])
            ub = np.append(ub, model.upperPositionLimit[j])
            x0 = np.append(
                x0, (model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)
    starttime = time.time()

    for iter in range(NbSample):

        param['iter'] = iter+1

    # constraint bounds -> c == 0 needed -> both bounds = 0
        cl = []
        cu = []

        x_opt = np.zeros([8])

        nlp = ipopt.problem(
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
        nlp.addOption('tol', 1e-6)  # Tolerance on teh end-effector 3D position
        # Tolerance on teh end-effector 3D position
        nlp.addOption('print_level', 1)

        #starttime = time.time()
        x_opt, info = nlp.solve(x0)
        #print('That took {} seconds'.format(time.time() - starttime))
         
        q_opt = np.array(robot.q0)  # np.zeros(shape=(12, 1))

        PEEe = []
        for j in range(1):
            q_opt[0:8] = x_opt 
             
            pin.forwardKinematics( model,  data, q_opt)
            pin.updateFramePlacements( model,  data)
            
            #Calculate_kinematics_model(q_opt, model, data, IDX_TOOL)
            PEEe = np.array(data.oMf[IDX_TOOL].translation)
             

        PEEd_iter = PEEd[[param['iter']-1, NbSample +
                          param['iter']-1, 2*NbSample+param['iter']-1]]
            
        J = np.sum(np.sqrt(np.square(PEEd_iter-PEEe)))/3
        if J <= 1e-3:
            print("Iter {} success ".format(iter+1))
        else:
            print("Iter {} Desired end-effector position: {} ".format(iter+1, PEEd_iter))
            print("Iter {} Achieved end-effector position: {} ".format(iter+1, PEEe))

    # PLEASE THANH COMPLETE THE CODE HERE TO calulate the condition number of the base regressor matrix using x_opt

    '''
    R,J=Calculate_identifiable_kinematics_model([], model, data, param)


    # select columns correspond to joint offset
    joint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
             50, 51, 52, 53]  # all on z axis - checked!!

    # regressor matrix on selected paramters
    R_sel = R[:, joint_idx]

    # a dictionary of selected parameters
    gp_listItems = list(geo_params.items())
    geo_params_sel = []
    for i in joint_idx:
        geo_params_sel.append(gp_listItems[i])
    geo_params_sel = dict(geo_params_sel)

    # eliminate zero columns
    R_e, geo_paramsr = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)

    # get base parameters
    R_b, params_base = get_baseParams(R_e, geo_paramsr)
    print("base parameters: ", (params_base))
    print("condition number: ", cond_num(R_b))
'''

     

if __name__ == "__main__":
    main()



# add a marker at the ee
# name = "ee_marker_frame"
# parent_joint = model.getJointId("arm_7_joint")
# prev_frame = model.getFrameId("arm_7_joint")
# placement = pin.SE3(eye(3), zero(3) + 0.1)
# inertia = pin.Inertia.Zero()
# frame = pin.Frame(name, parent_joint, prev_frame,
#                   placement, pin.FIXED_JOINT, inertia)
# model.addFrame(frame)
# jointName = "ee_marker_joint"
# jointPlacement = pin.SE3(eye(3), zero(3) + 0.1)
# jointId = model.getJointId("arm_7_joint")
# model.addJoint(jointId, pin.JointModelRZ(),
#                      jointPlacement, jointName)
# # inertia = pin.Inertia.Zero()
# # model.appendBodyToJoint(jointId, inertia, pin.SE3.Identity())
# model.lowerPositionLimit[14] = -1.5
# model.upperPositionLimit[14] = 1.5
# model.velocityLimit[12] = 1.5
# model.effortLimit[12] = 1.5
