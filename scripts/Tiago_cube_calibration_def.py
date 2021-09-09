import numpy as np
from scipy.optimize import approx_fprime
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import time
import cyipopt
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import numdifftools as nd
import quadprog as qp


#from scipy.optimize import approx_fprime

def Import_model():

    filename = "models/robots/tiago_description/robots/tiago_no_hand.urdf"
    mesh_dir = "models/robots/tiago_description/meshes/"

    robot = pin.RobotWrapper.BuildFromURDF(filename, mesh_dir)

    return robot


def display(robot, q):

    robot.display(q)
    # for name, oMi in zip(model.names[1:], robot.viz.data.oMi[1:]):
    #    robot.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(oMi))
    # gui.refresh()


def Calculate_kinematics_model(q_i, model, data, IDX_TOOL):

    # print(mp.current_process())
    #pin.updateGlobalPlacements(model , data)
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)

    J = pin.computeFrameJacobian(
        model, data, q_i, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)
    R = pin.computeFrameKinematicRegressor(
        model, data, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)

    return model, data, R, J


def Inverse_kinematics_joint_limits_avoidance(PEEd, model, data, param):

    oMdes = PEEd  # pin.SE3(np.eye(3), PEEd)

    q = param['q0']
    q_IK = q[param['Ind_joint']]
    q_opt = []

    delta_q = model.upperPositionLimit[param['Ind_joint']
                                       ]-model.lowerPositionLimit[param['Ind_joint']]
    q_avg = (model.upperPositionLimit[param['Ind_joint']] +
             model.lowerPositionLimit[param['Ind_joint']])/2
    eps = param['eps']
    IT_MAX = 1000
    DT = 1e-1
    damp = 1e-10

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    i = 0
    while True:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # oMdes.actInv(data.oMf[param['IDX_TOOL']])
        dMf = oMdes - data.oMf[param['IDX_TOOL']].translation
        err = dMf  # pin.log(dMf).vector

        if np.linalg.norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        #J = pin.computeJointJacobian(model,data,q,IDX_TOOL)
        J = pin.computeFrameJacobian(
            model, data, q, param['IDX_TOOL'], pin.LOCAL_WORLD_ALIGNED)
        J = J[0:3, param['Ind_joint']]
        # print(J.shape)
        #v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))

        nabla_phi = 2*(q_IK-q_avg)/delta_q

        Z = -10e0 * \
            np.matmul(np.identity(
                len(param['Ind_joint']))-np.matmul(np.linalg.pinv(J), J), nabla_phi)

        # - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        v = np.matmul(np.linalg.pinv(J), err)+Z

        q_IK = q_IK+v*DT
        q[param['Ind_joint']] = q_IK

        i += 1

    if success:
        print("Convergence achieved, error is:", err)
        q_opt = q
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the  desired precision")
        q_opt = []

    #print('\nresult: %s' % q.flatten().tolist())
    #print('\nfinal error: %s' % err.T)
    # print(i)

    # Contact placement
    return q_opt


# %% IK process
class CIK_problem(object):

    def __init__(self,  data, model, param):

        self.param = param
        self.model = model
        self.data = data

    def objective(self, x):
        # callback for objective

        q = np.array(self.param['q0'])

        PEEe = []
        for j in range(1):  # range(self.NbSample):

            q[self.param['Ind_joint']] = x  # [j::self.param['NbSample']]

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            PEEe = np.append(PEEe, np.array(
                self.data.oMf[self.param['IDX_TOOL']].translation))

        PEEd_all = self.param['PEEd']
        PEEd = PEEd_all[[self.param['iter']-1, self.param['NbSample'] +
                         self.param['iter']-1, 2*self.param['NbSample']+self.param['iter']-1]]

        # regularisation term noneed to use now +1e-4*np.sum( np.square(x) )
        J = np.sum(np.square(PEEd-PEEe))

        return J

    # def constraint_0(self, x):
     #   return np.array([x[0]*x[1]])

    # def constraint_1(self, x):
     #   return np.array([x[0]**2 * x[3] - x[2]])

    # def constraint_2(self, x):
    #    return np.array([x[3]**2 - x[1]])

    # def constraints(self, x):
        # callback for constraints
     #   return np.concatenate([self.constraint_0(x)])

    def gradient(self, x):
        # callback for gradient

        G = approx_fprime(x, self.objective, self.param['eps_gradient'])

        return G
    # def jacobian(self, x):
        # callback for jacobian
    #    return np.concatenate([
    #        approx_fprime(x, self.constraint_0, self.num_diff_eps)])

    # def hessian(self, x, lagrange, obj_factor):
    #    return False  # we will use quasi-newton approaches to use hessian-info

    # progress callback
    '''def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
'''
