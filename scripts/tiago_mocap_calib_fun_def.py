import numpy as np
from scipy.optimize import approx_fprime
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import time
import ipopt
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import numdifftools as nd
import quadprog as qp

from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num


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


def get_jointOffset(joint_names):
    joint_off = []

    for i in range(len(joint_names)):
        joint_off.append("off" + "_%d" % i)

    phi_jo = [0] * len(joint_off)
    joint_off = dict(zip(joint_off, phi_jo))
    return joint_off


def get_geoOffset(joint_names):
    tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
    geo_params = []

    for i in range(len(joint_names)):
        for j in tpl_names:
            geo_params.append(j + ("_%d" % i))

    phi_gp = [0] * len(geo_params)
    geo_params = dict(zip(geo_params, phi_gp))
    return geo_params


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


def Calculate_identifiable_kinematics_model(q, model, data, param):
    # Note if no q id given then use random generation of q to determine the minimal kinematics model
    if np.any(q):
        MIN_MODEL = 0
    else:
        MIN_MODEL = 1

    # obtain aggreated Jacobian matrix J and kinematic regressor R
    J = np.empty([6*param['NbSample'], model.nv])
    R = np.empty([6*param['NbSample'], 6*model.nv])
    for i in range(param['NbSample']):
        if MIN_MODEL == 1:
            q_i = pin.randomConfiguration(model)
        elif MIN_MODEL == 0:
            q_i = q[i, :]

        q_i[8:] = param['q0'][8:]

        model, data, Ri, Ji = Calculate_kinematics_model(
            q_i, model, data, param['IDX_TOOL'])
        for j in range(6):
            J[param['NbSample']*j + i, :] = Ji[j, :]
            R[param['NbSample']*j + i, :] = Ri[j, :]

    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names)]

    # calculate base Jacobian matrix J_b
    joint_off = get_jointOffset(joint_names)
    J_e, params_eJ = eliminate_non_dynaffect(J, joint_off, tol_e=1e-6)
    J_b, params_baseJ = get_baseParams(J_e, params_eJ)

    # select columns correspond to joint offset
    geo_params = get_geoOffset(joint_names)
    joint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
                 50, 51, 52, 53]  # all on z axis - checked!!
    R_sel = R[:, joint_idx]

    # a dictionary of selected parameters
    gp_listItems = list(geo_params.items())
    geo_params_sel = []
    for i in joint_idx:
        geo_params_sel.append(gp_listItems[i])
    geo_params_sel = dict(geo_params_sel)

    # calculate base kinematic regressor R_b
    R_e, params_eR = eliminate_non_dynaffect(R_sel, geo_params_sel, tol_e=1e-6)
    R_b, params_baseR = get_baseParams(R_e, params_eR)

    if param['PLOT'] == 1:
        # plotting
        plot1 = plt.figure()
        axs = plot1.subplots(6, 1)
        ylabel = ["px", "py", "pz", "phix", "phiy", "phiz"]
        for j in range(6):
            axs[j].plot(R[param['NbSample']*j:param['NbSample']
                        * j+param['NbSample'], 17])
            axs[j].set_ylabel(ylabel[j])
        # plt.show()

    return R_b, params_baseR, J_b, params_baseJ

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

    def gradient(self, x):
        # callback for gradient

        G = approx_fprime(x, self.objective, self.param['eps_gradient'])

        return G
