from sys import argv
import os
from os.path import dirname, join, abspath

import math
import time
import numpy as np

import numdifftools as nd


import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

#from pinocchio.pinocchio_pywrap import rpy

#from qrdecomposition import *
#from regressor import *

from Tiago_cube_calibration_def import *

#import time
#import multiprocessing as mp

#from itertools import repeat


def main():

    robot = Import_model()

    data = robot.model.createData()
    model = robot.model

    # robot.initViewer()
    # robot.loadViewerModel()

    # for i, name in enumerate(model.names):
    #    print(name)

    IDX_TOOL = model.getFrameId('wrist_ft_tool_link')

    ######################################################

    # NbSample=3
    # Nq=8 #number of joints to be optimized

    # cube_pose=[0.5, 0.1, 0.5]# position of the cube
    # cube_pose[len(cube_pose):] = [0, 0, 0, 1]# orientation of the cube
    # cube_dim=[0.2, 0.2, 0.2]

    # #oMdes = pin.SE3(np.eye(3), np.array([cube_pose[0], cube_pose[1], cube_pose[2]+cube_dim[2]/2]))
    # #PEEd = np.array([cube_pose[0], cube_pose[1], cube_pose[2]+cube_dim[2]/2])#np.array(data.oMf[IDX_TOOL].translation)
    # #print(PEEd)
    # #PEEd=np.zeros(3*NbSample)
    # #print("Shape of the array PEEd = ",np.shape(PEEd))

    # PEEd=np.linspace(cube_pose[0],cube_pose[0],NbSample )
    # PEEd=np.append(PEEd,np.linspace(cube_pose[1],cube_pose[1],NbSample ))
    # PEEd=np.append(PEEd,np.linspace(cube_pose[2]+cube_dim[2]/2,cube_pose[2]+cube_dim[2]/2,NbSample ))

    ######################################################
    """ I modified this part for a set of configs in a cube in the taskspace"""
    NbGrid = 4
    NbSample = pow(NbGrid, 3)
    Nq = 8  # number of joints to be optimized

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
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(PEEd_2d[0, :], PEEd_2d[1, :], PEEd_2d[2, :], color="green")
    # plt.title("simple 3D scatter plot")
    # plt.show()

    PEEd = PEEd_2d.flatten('C')
    ######################################################

    # Use a dictionnary to store optimisation parameters
    # eps is the tolerance on the desired end effector 3D position accuracy
    # Ind_joint index of the joint that sould be modified for/by the calibration process
    param = {
        'q0': np.array(robot.q0),
        'PEEd': PEEd,
        'IDX_TOOL': model.getFrameId('wrist_ft_tool_link'),
        'NbSample': NbSample,
        'eps': 1e-3,
        'eps_gradient': 1e-6,
        'Ind_joint': np.arange(8),
        'iter': 1,  # here the iter goes from 1 to NbSample
    }

    # set initial conditions and joint limits
    lb = ub = x0 = []
    for j in range(len(param['Ind_joint'])):
        for i in range(1):
            lb = np.append(lb, model.lowerPositionLimit[j])
            ub = np.append(ub, model.upperPositionLimit[j])
            # np.append(x0,(model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)#np.append(x0,j)#
            x0 = np.append(
                x0, (model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)
    starttime = time.time()

    for iter in range(3):

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
        nlp.addOption('max_iter', 1000)
        nlp.addOption('tol', 1e-6)  # Tolerance on teh end-effector 3D position
        # Tolerance on teh end-effector 3D position
        nlp.addOption('print_level', 1)

        #starttime = time.time()
        x_opt, info = nlp.solve(x0)
        #print('That took {} seconds'.format(time.time() - starttime))

        q_opt = np.array(robot.q0)  # np.zeros(shape=(12, 1))

        PEEe = []
        for j in range(1):
            q_opt[0:8] = x_opt  # [j::NbSample]
            Calculate_kinematics_model(q_opt, model, data, IDX_TOOL)
            PEEe = np.append(PEEe, np.array(data.oMf[IDX_TOOL].translation))

        PEEd_iter = PEEd[[param['iter']-1, param['NbSample'] +
                          param['iter']-1, 2*param['NbSample']+param['iter']-1]]

        J = np.sum(np.square(PEEd_iter-PEEe))
        if J <= 1e-6:
            print("Iter {} success ".format(iter+1))
        else:
            print("Iter {} Desired end-effector position: {} ".format(iter+1, PEEd_iter))
            print("Iter {} Achieved end-effector position: {} ".format(iter+1, PEEe))

    '''  
    ## Plotting
    #plt.ion() # use this to have non bocking windows
    plt.plot(PEEd,'k')       
    plt.plot(PEEe,'--r')      
    plt.xlabel('Samples')
    plt.ylabel('End-effector position')# 
    plt.show()                   # Display the plot
'''

    '''
    print('\nresult: %s' % q.flatten().tolist())
    print('\nfinal error: %s' % err.T)
    print(i)
    robot.display(q)

    
    M_ee = data.oMf[IDX_TOOL].translation
    # Contact placement
    #M_ct = M_ee.copy() 
    #M_ct.translation+=[.2, 0., 0.] # 20cm ahead     
    

    robot.viewer.gui.addBox("world/box_1",cube_dim[0], cube_dim[1], cube_dim[2],[0, 1, 0, 0.4])     
    robot.viewer.gui.applyConfiguration("world/box_1",cube_pose)
    #robot.viewer.gui.addFloor('world/floor')
    robot.viewer.gui.addLandmark('world/pinocchio/visuals/contact_0', 10)
    robot.viewer.gui.applyConfiguration('world/pinocchio/visuals/contact_0', [0.,0.,0.,0.,0.,0.,1.0])

    robot.viewer.gui.refresh()
    #tf_ref = pin.utils.se3ToXYZQUAT(M_ct)
    #robot.viewer.gui.addLandmark('world/ref_wrench', .5)
    #robot.viewer.gui.addLandmark('world/ref_wrench', tf_ref)
    '''


if __name__ == "__main__":
    main()


'''
#delta_prev=np.zeros(shape=(12, 1), dtype=float)

 



#rank=matrix_rank(Rtest)
 


#
#params_list={'PX1':0.0,'PY1':0.0,'PZ1':0.0,'RX1':0.0,'RY1':0.0,'RZ1':0.0,'PX2':0.0,'PY2':0.0,'PZ2':0.0,'RX2':0.0,'RY2':0.0,'RZ2':0.0}
 
#R_b, idx_b, params_rb=eliminate_non_dynaffect(Rtest, params_list, 0.0001)

#rint(params_rb)
#print(idx_b)

#print(params_r)

 
doMi[0,0:NbSample-1]=np.gradient(oMf[0,0:NbSample-1])/Ts
doMi[1,0:NbSample-1]=np.gradient(oMf[1,0:NbSample-1])/Ts
doMi[2,0:NbSample-1]=np.gradient(oMf[2,0:NbSample-1])/Ts

s1=dX[0:3,:]
s1_R=dXR#[0:3,:]


#a.plot(x1, y1, 'g^', x2, y2, 'g-')

#fig, [ax1] = plt#.plots(1, 1, sharex=True)
#fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)

#ax1.plot(oMi.T,'k--',Xvb.T,'g')

#ax1=plt.plots(1, 1, sharex=True)
'''
