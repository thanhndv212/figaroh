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

    NbSample = 3
    Nq = 8  # number of joints to be optimized

    cube_pose = [0.5, 0.1, 0.5]  # position of the cube
    cube_pose[len(cube_pose):] = [0, 0, 0, 1]  # orientation of the cube
    cube_dim = [0.2, 0.2, 0.2]

    #oMdes = pin.SE3(np.eye(3), np.array([cube_pose[0], cube_pose[1], cube_pose[2]+cube_dim[2]/2]))
    # PEEd = np.array([cube_pose[0], cube_pose[1], cube_pose[2]+cube_dim[2]/2])#np.array(data.oMf[IDX_TOOL].translation)
    # print(PEEd)
    # PEEd=np.zeros(3*NbSample)
    #print("Shape of the array PEEd = ",np.shape(PEEd))

    PEEd = np.linspace(cube_pose[0], cube_pose[0], NbSample)
    PEEd = np.append(PEEd, np.linspace(cube_pose[1], cube_pose[1], NbSample))
    PEEd = np.append(PEEd, np.linspace(
        cube_pose[2]+cube_dim[2]/2, cube_pose[2]+cube_dim[2]/2, NbSample))

    #PEEd=np.linspace(cube_pose[0]-cube_dim[0]/2,cube_pose[0]+cube_dim[0]/2,NbSample )
    #PEEd=np.append(PEEd,np.linspace(cube_pose[1]-cube_dim[1]/2,cube_pose[1]+cube_dim[1]/2,NbSample ))
    #PEEd=np.append(PEEd,np.linspace(cube_pose[2]+cube_dim[2]/2,cube_pose[2]+cube_dim[2]/2,NbSample ))

    q0 = np.array(robot.q0)
    # q[0]=q[0]#+0.1

    #Calculate_kinematics_model(q0, model,data,IDX_TOOL)
    #PEEd = np.array(data.oMf[IDX_TOOL].translation)

    # Use a dictionnary to store optimisation parameters
    # eps is the tolerance on the desired end effector 3D position accuracy
    # Ind_joint index of the joint that sould be modified for/by the calibration process
    param = {
        'q0': np.array(robot.q0),
        'PEEd': PEEd,
        'IDX_TOOL': model.getFrameId('wrist_ft_tool_link'),
        'NbSample': NbSample,
        'eps': 1e-3,
        'Ind_joint': np.arange(8),
    }

    # set initial conditions and joint limits
    lb = ub = []
    for j in range(len(param['Ind_joint'])):
        for i in range(NbSample):
            # np.append(lb,j-10)#
            lb = np.append(lb, model.lowerPositionLimit[j])
            # np.append(ub,j+10)#
            ub = np.append(ub, model.upperPositionLimit[j])
            #x0 = np.append(x0,q0[j])#np.append(x0,(model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)##np.append(x0,(model.lowerPositionLimit[j]+model.upperPositionLimit[j])/2)#np.append(x0,j)#
    starttime = time.time()
    q0 = x0 = []
    for i in range(NbSample):
        # print(PEEd[i::NbSample])
        q = Inverse_kinematics_joint_limits_avoidance(
            PEEd[i::NbSample], model, data, param)
        q0 = np.append(q0, q[0:len(param['Ind_joint'])])

    for j in range(len(param['Ind_joint'])):
        x0 = np.append(x0, q0[j::len(param['Ind_joint'])])
    #print('Initial condition determination took {} seconds'.format(time.time() - starttime))

    plt.plot(x0, '--r')       # Plot the sine of each x point
    plt.plot(ub, 'k')
    plt.plot(lb, 'k')
    # plt.xlabel('Samples')
    #plt.ylabel('joint angles')
    plt.show()


''' 
# constraint bounds -> c == 0 needed -> both bounds = 0
    cl = [ ]
    cu = [ ]

    x_opt=np.zeros([8,NbSample])

    nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=fitting_problem( data, model,param),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
            )
     
    # IMPORTANT: need to use limited-memory / lbfgs here as we didn't give a valid hessian-callback
    nlp.addOption(b'hessian_approximation', b'limited-memory')
    nlp.addOption('max_iter', 1000)

    starttime = time.time()
    #x_opt, info = nlp.solve(x0)
    print('That took {} seconds'.format(time.time() - starttime))
    
    q_opt = np.array(robot.q0)#np.zeros(shape=(12, 1)) 
    
    PEEe=[] 
    for j in range(NbSample): 
        q_opt[0:8]=x_opt[j::NbSample]    
        Calculate_kinematics_model(q_opt, model,data,IDX_TOOL)
        PEEe =np.append(PEEe, np.array(data.oMf[ IDX_TOOL].translation))

    print('Desired end-effector position:',PEEd) 
    print('Achieved end-effector position:',PEEe) 
     
    ## Plotting
    #plt.ion() # use this to have non bocking windows
    plt.plot(PEEd,'k')       
    plt.plot(PEEe,'--r')      
    plt.xlabel('Samples')
    plt.ylabel('End-effector position')# 
    plt.show()                   # Display the plot
'''

#
#data = []
# for i in range(1000):
#    data.append(model.createData())

# Calculate_kinematics_model_nodata=partial(Calculate_kinematics_model,q,model,IDX_TOOL)

#data = [[model.createData()] for _ in range(1000)]
#starttime = time.time()

#POOL_SIZE = 4
#data = [model.createData() for _ in range(POOL_SIZE)]
# with mp.Pool() as pool:
#    pool.starmap(Calculate_kinematics_model_nodata, data)
#print('That took {} seconds'.format(time.time() - starttime))

#starttime = time.time()
#list(map(Calculate_kinematics_model_nodata, data))
#print('That took {} seconds'.format(time.time() - starttime))

#print("Number of cpu : ", mp.cpu_count())

#starttime = time.time()

# with mp.Pool() as pool:
#    pool.starmap(Calculate_kinematics_model_nodata, data )

#result = pool.map(Calculate_kinematics_model, )
#result_set_2 = pool.map(my_func, [4,6,5,4,6,3,23,4,6])

# print(data[0].oMi)
# print(result_set_2)

#print('That took {} seconds'.format(time.time() - starttime))

#starttime = time.time()

# for i in range(10):
#    p = multiprocessing.Process(target=Calculate_kinematics_model, args=(q, model,data,IDX_TOOL))
#    #print(p)
#    processes.append(p)
#    p.start()
#print('That took {} seconds'.format(time.time() - starttime))


'''
    for i in range(0,NbSample):
        print(i)

'''
# IMPORTANT: need to use limited-memory / lbfgs here as we didn't give a valid hessian-callback
#nlp.addOption(b'hessian_approximation', b'limited-memory')
#    nlp.addOption('print_level',0)
#    x[:,i], info = nlp.solve(x0)

# %%


# %% Plotting
# plt.ion() # use this to have non bocking windows
# plt.plot(x[0,:],'k')       # Plot the sine of each x point
# plt.plot(q[:,0],'--r')       # Plot the sine of each x point

# plt.xlabel('Samples')
# plt.ylabel('joint angles')#'Mks positions [m]')
# plt.show()                   # Display the plot

# plt.pause(1)
#input("Press [enter] to continue.")


# code starts about identification of parameters
# NbSample=100

#off_P=np.array([0.1, 0.0, 0.1, 0.1, 0.0, 0])
#off_R=np.array([0.0, 0.3, 0.0, 0.0, 0, 0.0])

# idx_meas=np.array([0, 2])#np.arange(3)# x and z axis position
#idx_param=np.array([0, 1, 2, 4, 6, 7, 8,10])

#NQ = model.nq
#NV = model.nv
#IDX_TOOL = model.getFrameId('wrist_ft_tool_link')
#IDX_TOOL= model.getJointId('arm_7_joint')
# print(IDX_TOOL)
# pre-allocate memory
#q = np.zeros(shape=(NQ, NbSample+1))
#dq = np.zeros(shape=(NQ, NbSample+1))

#oMf_meas=np.zeros(shape=(3, NbSample+1), dtype=float)
#oMf_est=np.zeros(shape=(3, NbSample+1), dtype=float)

#doMi=np.empty(shape=(3, NbSample), dtype=float)

#R_all=np.empty(shape=(len(idx_meas),len(idx_param), NbSample+1), dtype=float)
#R_sub=np.zeros(shape=(len(idx_meas),len(idx_param), NbSample+1), dtype=float)

'''
    J_all=np.empty(shape=(len(idx_meas),2, NbSample+1), dtype=float)
    
    


    q      = robot.q0#pin.neutral(model)
    print(q)
    eps    = 1e-4
    IT_MAX = 2000
    DT     = 1e-1
    damp   = 1e-12
 
    i=0
    while True:
        pin.forwardKinematics(model,data,q)
        #dMi = oMdes.actInv(data.oMi[IDX_TOOL])
        #err = pin.log(dMi).vector

        dMf = oMdes.actInv(data.oMf[IDX_TOOL])
        err = pin.log(dMf).vector

        if norm(err) < eps:
             success = True
             break
        if i >= IT_MAX:
             success = False
             break
        #J = pin.computeJointJacobian(model,data,q,IDX_TOOL)
        J = pin.computeFrameJacobian(model,data,q,IDX_TOOL,pin.LOCAL_WORLD_ALIGNED)

        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model,q,v*DT)
        #if not i % 10:
        #     print('%d: error = %s' % (i, err.T))
        i += 1
    #

    if success:
         print("Convergence achieved!")
    else:
         print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
    
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
    

    w=2*1*np.pi
    Ts=0.1
    t=np.array(0.0)
    q0 = np.array([0.0, 0.0])

    

    # Modify model with offset in joint placement n°1
   

    #model, data, R, J=Calculate_kinematics_model(q0, model,data,IDX_TOOL)
    #print("end-effector placement = ", data.oMf[IDX_TOOL])
    #print(IDX_TOOL)
    # simulate data with an offset 
    p=model.jointPlacements[1].translation
    model.jointPlacements[1].translation=p + off_P[0:3]

    p=model.jointPlacements[2].translation
    model.jointPlacements[2].translation=p + off_P[3:6]
    print(model.jointPlacements[2])

    Rot = rpy.rpyToMatrix(off_R[0:3])
    model.jointPlacements[1].rotation = Rot 
    
    Rot = rpy.rpyToMatrix(off_R[3:6])
    model.jointPlacements[2].rotation = Rot

    model, data, R, J=Calculate_kinematics_model(q0, model,data,IDX_TOOL)
    print("New end-effector placement = ", data.oMf[IDX_TOOL])
 

  


    # simulate data with an offset 
    p=model.jointPlacements[1].translation
    model.jointPlacements[1].translation=p + off_P[0:3]

    p=model.jointPlacements[2].translation
    model.jointPlacements[2].translation=p + off_P[3:6]
 
    Rot = rpy.rpyToMatrix(off_R[0:3])
    model.jointPlacements[1].rotation = Rot 
    
    Rot = rpy.rpyToMatrix(off_R[3:6])
    model.jointPlacements[2].rotation = Rot

    for i in range(NbSample+1):

        toto=np.random.rand(1,1)
        tata=np.random.rand(1,1)

        q[0,i] =0.2*np.cos(w*t)+t*t#+0.2
        q[1,i] =0.3*np.cos(2*w*t)+t#+np.random.rand(1,1)

        #dq[0,i] =-0.1*(w)*np.sin(w*t)
        #dq[1,i] =-0.2*(2*w)*np.sin(w*t)
        #print(model.jointPlacements[1].translation)
        model, data, R, J=Calculate_kinematics_model(q[:,i], model,data,IDX_TOOL)

        oMf_meas[:,i]=np.array(data.oMf[IDX_TOOL].translation )

        t=t+Ts 

    oMf_meas_vect=np.reshape(oMf_meas[[idx_meas],:],(len(idx_meas)*(NbSample+1), 1))
    #print("New end-effector placement = ", data.oMf[IDX_TOOL])
     
 
    # Calculate the identification model with measured joint data
    # remove the offset to re-identify it

    p=model.jointPlacements[1].translation
    model.jointPlacements[1].translation=p - off_P[0:3]
    p01=model.jointPlacements[1].translation

    p=model.jointPlacements[2].translation
    model.jointPlacements[2].translation=p - off_P[3:6]
    p02=model.jointPlacements[2].translation


    #delta=np.array([0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.])#np.array([0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.])
    delta=np.array([0., 0., 0., 0., 0., 0.,0., 0.]) 
    #print(delta.shape)
 
    
    for iter in range(2): 

    #update model parameters
        print(iter)

        p=model.jointPlacements[1].translation
        model.jointPlacements[1].translation=p + np.array([delta[0],delta[1],delta[2]])

        p=model.jointPlacements[2].translation
        model.jointPlacements[2].translation=p + np.array([delta[4],delta[5],delta[6]])

        Rot = rpy.rpyToMatrix(np.array([0.,delta[3],0.]))#delta[6:9])
        model.jointPlacements[1].rotation = Rot 
    
        Rot = rpy.rpyToMatrix(np.array([0.,delta[7],0.]))
        model.jointPlacements[2].rotation = Rot
        
        #R=rpy.rpyToMatrix(0.,*delta[0], 0.)
        #model.jointPlacements[1].rotation=R
       

        for i in range(NbSample+1):
            

            model, data, R, J=Calculate_kinematics_model(q[:,i], model,data,IDX_TOOL)
            oMf_est[:,i]=np.array(data.oMf[IDX_TOOL].translation )
             
            
            J_all[:,:,i]=J[idx_meas,:]
            
            for j in range(len(idx_meas)):
                R_all[[j],:,[i]]=R[idx_meas[j],idx_param]

         
        print(R_all.shape)
        #if (np.sum(np.abs(delta-delta_prev)))>10:
        #print("end-effector placement = ", data.oMf[IDX_TOOL])

        oMf_est_vect=np.reshape(oMf_est[[idx_meas],:],(len(idx_meas)*(NbSample+1), 1))
        J_reg=np.reshape(J_all.transpose(0,2,1),(len(idx_meas)*(NbSample+1), 2))

        #for j, ind in enumerate(idx_meas):
        #    R_sub[j,:,:]=np.squeeze(R_all[ind,:,:])

    
        Regressor=np.reshape(R_all.transpose(0,2,1),(len(idx_meas)*(NbSample+1), len(idx_param)))
        #print(Regressor.shape)(0,2,1)

        # scipy has QR pivoting using Householder reflection
        Q, R = np.linalg.qr(Regressor)#, pivoting=True)

        indep = np.where(np.abs(R.diagonal()) >  1e-6)[0]
         
        print("Independent columns are: {}".format(indep))

        Regressor_base=Regressor[:, indep]
        #print(Regressor[:, indep])
       
        R_pinv=np.linalg.pinv(Regressor_base)

        delta_b=np.matmul(R_pinv,oMf_meas_vect-oMf_est_vect)
        for ind_b,value in enumerate(indep):
            print(value)
            print(delta[value])
            print(delta_b[ind_b])
            delta[value]= delta_b[ind_b]

        print(Regressor_base.shape)#delta=delta+delta_new
        print(delta.shape)
        oMf_id_vect=np.matmul(Regressor,delta)

        res=np.sqrt(np.sum(np.square(oMf_meas_vect-oMf_est_vect))/NbSample)
     
        
        print(delta) 
        print(res) 

        plt.plot(oMf_meas_vect,'k',oMf_est_vect,'g--')#,oMf_est_vect,'--r')
        #plt.plot(q.T,'k',q_new.T,'g--')
        ind=4
        #print(ind+6)
        #plt.plot(Regressor[:,ind],'k',Regressor[:,ind+6],'r--',J_reg,'g--')
        plt.show() 

        #Q,R=np.linalg.qr(Regressor.T)

     
        condition_number=np.linalg.cond(Regressor)

    #rank=np.linalg.matrix_rank(Regressor)
        #print("condition number is: {}".format(condition_number))
    #print("rank is: {}".format(rank))

    #plt.plot(Regressor) 
    #plt.show()
    

    #J=np.array(data.J)
    #JR=np.squeeze(R_all[0:3,[4, 10],[i]])# Only for x and Z position
     
    #dXR[:,[i]]=np.matmul(JR,dq)

    #dX[:,[i]]=np.matmul(J,dq)#J.dot(dq)#np.matmul(J,dq)
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
