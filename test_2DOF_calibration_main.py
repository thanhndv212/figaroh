from sys import argv
import os
from os.path import dirname, join, abspath

import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

import math
import time
import numpy as np

import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

from pinocchio.pinocchio_pywrap import rpy


from qrdecomposition import *
from regressor import *

def Import_model():

    filename="robots/planar_2DOF/urdf/planar_2DOF_TCP.urdf"
    mesh_dir="robots/planar_2DOF/meshes/"

    robot = pin.RobotWrapper.BuildFromURDF(filename, mesh_dir)

    data     = robot.model.createData()
    model = robot.model
    return model, data






def Calculate_kinematics_model(q_i, model,data,IDX_TOOL):
    

    #pin.updateGlobalPlacements(model , data)
    pin.forwardKinematics(model, data, q_i)

    pin.updateFramePlacements(model , data)

    J=pin.computeFrameJacobian(model,data,q_i,IDX_TOOL,pin.LOCAL_WORLD_ALIGNED)
    R=pin.computeFrameKinematicRegressor(model,data,IDX_TOOL,pin.LOCAL_WORLD_ALIGNED)
    

    return model, data, R, J

def main():
    
    model, data=Import_model()
    NbSample=100
    
    off_P=np.array([0.1, 0.0, 0.1, 0.1, 0.0, 0])
    off_R=np.array([0.0, 0.3, 0.0, 0.0, 0, 0.0])

    idx_meas=np.array([0, 2])#np.arange(3)# x and z axis position
    idx_param=np.array([0, 1, 2, 4, 6, 7, 8,10])

    NQ = model.nq
    NV = model.nv
    IDX_TOOL = model.getFrameId('tcp')

    # pre-allocate memory
    q = np.zeros(shape=(2, NbSample+1))    
    dq = np.zeros(shape=(2, NbSample+1))
     
    oMf_meas=np.zeros(shape=(3, NbSample+1), dtype=float)
    oMf_est=np.zeros(shape=(3, NbSample+1), dtype=float)

    doMi=np.empty(shape=(3, NbSample), dtype=float)

    R_all=np.empty(shape=(len(idx_meas),len(idx_param), NbSample+1), dtype=float)
    R_sub=np.zeros(shape=(len(idx_meas),len(idx_param), NbSample+1), dtype=float)


    J_all=np.empty(shape=(len(idx_meas),2, NbSample+1), dtype=float)


    w=2*1*np.pi
    Ts=0.1
    t=np.array(0.0)
    q0 = np.array([0.0, 0.0])

    

    # Modify model with offset in joint placement n°1
   
    '''
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
    '''


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
'''

#a.plot(x1, y1, 'g^', x2, y2, 'g-')

#fig, [ax1] = plt#.plots(1, 1, sharex=True)
#fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)

#ax1.plot(oMi.T,'k--',Xvb.T,'g')

#ax1=plt.plots(1, 1, sharex=True)






