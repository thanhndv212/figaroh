import numpy as np
from scipy.optimize import approx_fprime
from scipy.optimize._numdiff import approx_derivative
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import time
import cyipopt
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import numdifftools as nd
import quadprog as qp

from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, get_baseIndex, build_baseRegressor, cond_num

from numpy import linalg as la

#from scipy.optimize import approx_fprime


def Import_model():

    filename = "models/robots/tiago_description/robots/tiago_no_hand.urdf"
    mesh_dir = "models/robots/tiago_description/meshes/"

    robot = pin.RobotWrapper.BuildFromURDF(filename, mesh_dir)

    return robot


def display(robot, model,  q):

    robot.display(q)
    for name, oMi in zip(model.names[1:], robot.viz.data.oMi[1:]):
        robot.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(oMi))
    robot.viewer.gui.refresh()


def get_jointOffset(joint_names):
    """ This function give a dictionary of joint offset parameters.
            Input:  joint_names: a list of joint names (from model.names)
            Output: joint_off: a dictionary of joint offsets.
    """
    joint_off = []

    for i in range(len(joint_names)):
        joint_off.append("off" + "_%d" % i)

    phi_jo = [0] * len(joint_off)  # default zero values
    joint_off = dict(zip(joint_off, phi_jo))
    return joint_off


def get_geoOffset(joint_names):
    """ This function give a dictionary of variations (offset) of kinematics parameters.
            Input:  joint_names: a list of joint names (from model.names)
            Output: geo_params: a dictionary of variations of kinematics parameters.
    """
    tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
    geo_params = []

    for i in range(len(joint_names)):
        for j in tpl_names:
            geo_params.append(j + ("_%d" % i))

    phi_gp = [0] * len(geo_params)  # default zero values
    geo_params = dict(zip(geo_params, phi_gp))
    return geo_params


def add_eemarker_frame(frame_name, p, rpy, model, data):
    """ Adds a frame at the end_effector.
    """
    p = np.array([0.1, 0.1, 0.1])
    R = pin.rpy.rpyToMatrix(rpy)
    frame_placement = pin.SE3(R, p)

    parent_jointId = model.getJointId("arm_7_joint")
    prev_frameId = model.getFrameId("arm_7_joint")
    ee_frame_id = model.addFrame(
        pin.Frame(frame_name, parent_jointId, prev_frameId, frame_placement, pin.FrameType(0), pin.Inertia.Zero()), False)
    return ee_frame_id


def get_PEE_var(var, q, model, data, param, noise=False):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
    """
    # calibration index = 3 or 6, indicating whether or not orientation incld.
    nrow = param['calibration_index']
    ncol = param['NbSample']
    PEE = np.empty((nrow, ncol))
    q_temp = np.copy(q)
    for i in range(ncol):
        config = q_temp[i, :]
        # frame trasformation matrix from mocap to base
        p_base = var[0:3]
        rpy_base = var[3:6]
        R_base = pin.rpy.rpyToMatrix(rpy_base)
        base_placement = pin.SE3(R_base, p_base)

        # adding offset (8 for 8 joints)
        config[0:8] = config[0:8] + var[6:14]

        # adding zero mean additive noise to simulated measured coordinates
        if noise:
            noise = np.random.normal(0, 0.001, var[0:8].shape)
            config[0:8] = config[0:8] + noise

        pin.framesForwardKinematics(model, data, config)
        pin.updateFramePlacements(model, data)

        # calculate oMf from wrist to the last frame
        p_ee = var[14:17]
        rpy_ee = var[17:20]
        R_ee = pin.rpy.rpyToMatrix(rpy_ee)
        last_placement = pin.SE3(R_ee, p_ee)

        base_oMf = base_placement * \
            data.oMf[param['IDX_TOOL']]  # from mocap to wirst
        new_oMf = base_oMf*last_placement  # from wrist to end effector

        # create a matrix containing coordinates of end_effector
        PEE[0:3, i] = new_oMf.translation
        if nrow == 6:
            PEE_rot = new_oMf.rotation
            PEE[3:6, i] = pin.rpy.matrixToRpy(PEE_rot)

    PEE = PEE.flatten('C')
    return PEE


def get_PEE(offset_var, q, model, data, param, noise=False):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
    """
    # calibration index = 3 or 6, indicating whether or not orientation incld.
    nrow = param['calibration_index']
    ncol = param['NbSample']
    PEE = np.empty((nrow, ncol))
    q_temp = np.copy(q)
    for i in range(ncol):
        config = q_temp[i, :]
        config[0:8] = config[0:8] + offset_var
        # adding zero mean additive noise to simulated measured coordinates
        if noise:
            noise = np.random.normal(0, 0.001, offset_var.shape)
            config[0:8] = config[0:8] + noise

        pin.framesForwardKinematics(model, data, config)
        pin.updateFramePlacements(model, data)

        PEE[0:3, i] = data.oMf[param['IDX_TOOL']].translation
        if nrow == 6:
            PEE_rot = data.oMf[param['IDX_TOOL']].rotation
            PEE[3:6, i] = pin.rpy.matrixToRpy(PEE_rot)
    PEE = PEE.flatten('C')
    return PEE


def Calculate_kinematics_model(q_i, model, data, IDX_TOOL):
    """ Calculate jacobian matrix and kinematic regressor given ONE configuration.
    """
    # print(mp.current_process())
    #pin.updateGlobalPlacements(model , data)
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)

    J = pin.computeFrameJacobian(
        model, data, q_i, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)
    R = pin.computeFrameKinematicRegressor(
        model, data, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)

    return model, data, R, J


# def Calculate_identifiable_jointOffset_model(q, model, data, param):

#     # Note if no q id given then use random generation of q to determine the minimal kinematics model
#     if np.any(q):
#         MIN_MODEL = 0
#     else:
#         MIN_MODEL = 1

#     # obtain aggreated Jacobian matrix J and kinematic regressor R
#     calib_idx = param['calibration_index']
#     J = np.empty([calib_idx*param['NbSample'], model.nv])
#     for i in range(param['NbSample']):
#         q_i = pin.randomConfiguration(model)
#         q_i[8:] = param['q0'][8:]
#         model, data, Ri, Ji = Calculate_kinematics_model(
#             q_i, model, data, param['IDX_TOOL'])
#         for j in range(calib_idx):
#             J[param['NbSample']*j + i, :] = Ji[j, :]

#     # obtain joint names
#     joint_names = [name for i, name in enumerate(model.names)]

#     # calculate base Jacobian matrix J_b
#     joint_off = get_jointOffset(joint_names)
#     J_e, params_eJ = eliminate_non_dynaffect(J, joint_off, tol_e=1e-6)
#     J_b, params_baseJ = get_baseParams(J_e, params_eJ)
#     return J_b, params_baseJ


def Calculate_identifiable_kinematics_model(q, model, data, param):
    """ Calculate jacobian matrix and kinematic regressor and aggreating into one matrix,
        given a set of configurations or random configurations if not given.
    """
    q_temp = q
    # Note if no q id given then use random generation of q to determine the minimal kinematics model
    if np.any(q):
        MIN_MODEL = 0
    else:
        MIN_MODEL = 1

    # obtain aggreated Jacobian matrix J and kinematic regressor R
    calib_idx = param['calibration_index']
    R = np.empty([calib_idx*param['NbSample'], 6*model.nv])
    for i in range(param['NbSample']):

        if MIN_MODEL == 1:
            q_i = pin.randomConfiguration(model)
        else:
            q_i = q_temp[i, :]
        q_i[8:] = param['q0'][8:]
        model, data, Ri, Ji = Calculate_kinematics_model(
            q_i, model, data, param['IDX_TOOL'])
        for j in range(calib_idx):
            R[param['NbSample']*j + i, :] = Ri[j, :]
    return R


def Calculate_base_kinematics_regressor(q, model, data, param):

    if np.any(q):
        MIN_MODEL = 0
    else:
        MIN_MODEL = 1

    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names)]
    geo_params = get_geoOffset(joint_names)

    # particularly select columns/parameters corresponding to joint and 6 last parameters
    actJoint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
                    50, 51, 52, 53]  # all on z axis - checked!!

    # a dictionary of selected parameters
    gp_listItems = list(geo_params.items())
    geo_params_sel = []
    for i in actJoint_idx:
        geo_params_sel.append(gp_listItems[i])
    geo_params_sel = dict(geo_params_sel)

    # calculate kinematic regressor with random configs
    Rrand = Calculate_identifiable_kinematics_model([], model, data, param)

    # select columns corresponding to joint_idx
    Rrand_sel = Rrand[:, actJoint_idx]

    # obtain a list of column after apply QR decomposition
    Rrand_e, paramsrand_e = eliminate_non_dynaffect(
        Rrand_sel, geo_params_sel, tol_e=1e-6)
    print(Rrand_e.shape)
    idx_base = get_baseIndex(Rrand_e, paramsrand_e)
    _, paramsrand_base = get_baseParams(Rrand_e, paramsrand_e)

    if MIN_MODEL == 0:
        # select columns corresponding to joint_idx
        R_sel = Rrand[:, actJoint_idx]

    else:
        # calculate kinematic regressor with random configs
        R = Calculate_identifiable_kinematics_model(q, model, data, param)
        # select columns corresponding to joint_idx
        R_sel = R[:, actJoint_idx]

    # obtain a list of column after apply QR decomposition
    R_e, params_e = eliminate_non_dynaffect(
        R_sel, geo_params_sel, tol_e=1e-6)

    R_b = build_baseRegressor(R_e, idx_base)

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

    return R_b, paramsrand_base, idx_base

# %% IK process


class CIK_problem(object):

    def __init__(self,  data, model, param):

        self.param = param
        self.model = model
        self.data = data

    def objective(self, x):
        # callback for objective

        config = np.array(self.param['q0'])

        PEEe = []
        for j in range(1):  # range(self.NbSample):

            config[self.param['Ind_joint']] = x  # [j::self.param['NbSample']]

            pin.forwardKinematics(self.model, self.data, config)
            pin.updateFramePlacements(self.model, self.data)
            PEEe = np.append(PEEe, np.array(
                self.data.oMf[self.param['IDX_TOOL']].translation))

        PEEd_all = self.param['PEEd']
        PEEd = PEEd_all[[self.param['iter']-1, self.param['NbSample'] +
                         self.param['iter']-1, 2*self.param['NbSample']+self.param['iter']-1]]

        # regularisation term noneed to use now +1e-4*np.sum( np.square(x) )
        # +1e-1/np.sum(np.square(x-self.param['x_opt_prev'])+100)
        J = np.sum(np.square(PEEd-PEEe))

        return J

    def gradient(self, x):
        # callback for gradient

        G = approx_fprime(x, self.objective, self.param['eps_gradient'])

        return G


class OCP_determination_problem(object):

    def __init__(self,  data, model, param):

        self.param = param
        self.model = model
        self.data = data
        self.const_ind = 0

    def objective(self, x):
        # callback for objective

        # check with Thanh is those are the index correspoding to the base parameters
        actJoint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
                        50, 51, 52, 53]  # all on z axis - checked!!

        config = np.array(self.param['q0'])

        PEEe = []
        for j in range(1):  # range(self.NbSample):

            config[self.param['Ind_joint']] = x  # [j::self.param['NbSample']]
            model, data, R, J = Calculate_kinematics_model(
                config, self.model, self.data, self.param['IDX_TOOL'])

            # select columns corresponding to joint_idx
            R = R[:, actJoint_idx]
            # select form the list of columns given by the QR decomposition
            R = R[:, self.param['idx_base']]
            # print(self.param['R_prev'])

            if self.param['iter'] > 1:

                R = np.concatenate([self.param['R_prev'], R])

            # PEEe = np.append(PEEe, np.array(
            #    self.data.oMf[self.param['IDX_TOOL']].translation))

        #PEEd_all = self.param['PEEd']
        # PEEd = PEEd_all[[self.param['iter']-1, self.param['NbSample'] +
        #                self.param['iter']-1, 2*self.param['NbSample']+self.param['iter']-1]]

        J = la.cond(R)  # np.sum(np.square(PEEd-PEEe))

        return J

    def constraint_0(self, x):

        config = np.array(self.param['q0'])

        PEEe = []
        for j in range(1):  # range(self.NbSample):

            config[self.param['Ind_joint']] = x  # [j::self.param['NbSample']]

            pin.forwardKinematics(self.model, self.data, config)
            pin.updateFramePlacements(self.model, self.data)
            PEEe = np.append(PEEe, np.array(
                self.data.oMf[self.param['IDX_TOOL']].translation))

        PEEd_all = self.param['PEEd']
        PEEd = PEEd_all[[self.param['iter']-1, self.param['NbSample'] +
                         self.param['iter']-1, 2*self.param['NbSample']+self.param['iter']-1]]

        c = np.array([np.square(PEEd[0]-PEEe[0])])

        return c

    def constraint_1(self, x):

        config = np.array(self.param['q0'])

        PEEe = []
        for j in range(1):  # range(self.NbSample):

            config[self.param['Ind_joint']] = x  # [j::self.param['NbSample']]

            pin.forwardKinematics(self.model, self.data, config)
            pin.updateFramePlacements(self.model, self.data)
            PEEe = np.append(PEEe, np.array(
                self.data.oMf[self.param['IDX_TOOL']].translation))

        PEEd_all = self.param['PEEd']
        PEEd = PEEd_all[[self.param['iter']-1, self.param['NbSample'] +
                         self.param['iter']-1, 2*self.param['NbSample']+self.param['iter']-1]]

        c = np.array([np.square(PEEd[1]-PEEe[1])])

        return c

    def constraint_2(self, x):

        config = np.array(self.param['q0'])

        PEEe = []
        for j in range(1):  # range(self.NbSample):

            config[self.param['Ind_joint']] = x  # [j::self.param['NbSample']]

            pin.forwardKinematics(self.model, self.data, config)
            pin.updateFramePlacements(self.model, self.data)
            PEEe = np.append(PEEe, np.array(
                self.data.oMf[self.param['IDX_TOOL']].translation))

        PEEd_all = self.param['PEEd']
        PEEd = PEEd_all[[self.param['iter']-1, self.param['NbSample'] +
                         self.param['iter']-1, 2*self.param['NbSample']+self.param['iter']-1]]

        c = np.array([np.square(PEEd[2]-PEEe[2])])

        return c

    def constraints(self, x):
        """Returns the constraints."""
        # callback for constraints

        return np.concatenate([self.constraint_0(x),
                               self.constraint_1(x),
                               self.constraint_2(x)])

        '''
        if  self.const_ind==0: 

            cts= J[self.const_ind] 
        
        if  self.const_ind==1: 

            cts= J[self.const_ind] 
        '''

    def jacobian(self, x):
        # Returns the Jacobian of the constraints with respect to x
        #
        # self.const_ind=0
        #J0 = approx_fprime(x, self.constraints, self.param['eps_gradient'])

        return np.concatenate([
            approx_fprime(x, self.constraint_0, self.param['eps_gradient']),
            approx_fprime(x, self.constraint_1, self.param['eps_gradient']),
            approx_fprime(x, self.constraint_2, self.param['eps_gradient'])])

        # print(J0)
        # J=nd.Jacobian(self.constraints)(x)

        return J0  # np.array(J)#np.concatenate([J0,J0])

    def gradient(self, x):
        # callback for gradient
        G = approx_fprime(x, self.objective, self.param['eps_gradient'])

        return G
