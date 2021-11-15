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
import pandas as pd

from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import (
    get_baseParams,
    get_baseIndex,
    build_baseRegressor,
    cond_num)


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
    # PLEASE THANH find a way to put this in a function
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


def get_param(robot, NbSample, TOOL_NAME='ee_marker_joint', NbMarkers=2,  calib_model='full_params', calib_idx=3):
    tool_FrameId = robot.model.getFrameId(TOOL_NAME)
    parentJoint2Tool_Id = robot.model.frames[tool_FrameId].parent
    NbJoint = parentJoint2Tool_Id  # joint #0 is  universe
    print("number of active joint: ", NbJoint)
    print("tool name: ", TOOL_NAME)
    print("parent joint of tool frame: ", robot.model.names[NbJoint])

    param = {
        'q0': np.array(robot.q0),
        'x_opt_prev': np.zeros([NbJoint]),
        'NbSample': NbSample,
        'IDX_TOOL': tool_FrameId,
        'eps': 1e-3,
        'Ind_joint': np.arange(NbJoint),
        'PLOT': 0,
        'NbMarkers': NbMarkers,
        'calib_model': calib_model,  # 'joint_offset',  #
        'calibration_index': calib_idx,
        'NbJoint': NbJoint
    }
    return param


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


def extract_expData(path_to_file, param):
    # first 7 cols: coordinates xyzquat of end effector by mocap
    xyz_rotQuat = pd.read_csv(
        path_to_file, usecols=list(range(0, 7))).to_numpy()

    # next 8 cols: joints position
    q_act = pd.read_csv(path_to_file, usecols=list(range(7, 15))).to_numpy()

    # delete the 4th data sample/outlier
    xyz_rotQuat = np.delete(xyz_rotQuat, 4, 0)
    q_act = np.delete(q_act, 4, 0)

    param['NbSample'] = q_act.shape[0]

    # extract measured end effector coordinates
    PEEm_exp = np.empty((param['calibration_index'], param['NbSample']))
    for i in range(param['NbSample']):
        PEE_se3 = pin.XYZQUATToSE3(xyz_rotQuat[i, :])
        PEEm_exp[0:3, i] = PEE_se3.translation
        if param['calibration_index'] == 6:
            PEEm_exp[3:6, i] = pin.rpy.matrixToRpy(PEE_se3.rotation)
    PEEm_exp = PEEm_exp.flatten('C')

    # extract measured joint configs
    q_exp = np.empty((param['NbSample'], param['q0'].shape[0]))
    for i in range(param['NbSample']):
        q_exp[i, 0:8] = q_act[i, :]
        # ATTENTION: need to check robot.q0 vs TIAGo.q0
        q_exp[i, 8:] = param['q0'][8:]
    return PEEm_exp, q_exp


def extract_expData4Mkr(path_to_file, param):
    # first 12 cols: xyz positions of 4 markers
    xyz_4Mkr = pd.read_csv(
        path_to_file, usecols=list(range(0, param['NbMarkers']*3))).to_numpy()

    # next 8 cols: joints position
    q_act = pd.read_csv(path_to_file, usecols=list(range(12, 20))).to_numpy()
    param['NbSample'] = q_act.shape[0]

    # extract measured end effector coordinates
    PEEm_exp = xyz_4Mkr.T
    PEEm_exp = PEEm_exp.flatten('C')
    # extract measured joint configs
    q_exp = np.empty((param['NbSample'], param['q0'].shape[0]))
    for i in range(param['NbSample']):
        q_exp[i, 0:8] = q_act[i, :]
        # ATTENTION: need to check robot.q0 vs TIAGo.q0
        q_exp[i, 8:] = param['q0'][8:]
    return PEEm_exp, q_exp
# create values storing dictionary 'param'


def cartesian_to_SE3(X):
    ''' Convert (6,) cartesian coordinates to SE3
        input: 1D (6,) numpy array
        out put: SE3 type placement
    '''
    X = X.flatten('C')
    translation = X[0:3]
    rot_matrix = pin.rpy.rpyToMatrix(X[3:6])
    placement = pin.SE3(rot_matrix, translation)
    return placement

######################## LM least squares functions ########################################


def init_var(param, mode=0, base_model=True):
    ''' Creates variable vector, mode = 0/1 for zero values/nonzero-random values
    '''
    # x,y,z,r,p,y from mocap to base ( 6 parameters for pos and orient)
    # 3D base frame
    # qBase_0 = np.array([0.1, 0.1, 0.1])

    # 6D base frame
    qBase_0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # parameter variation at joints
    if mode == 0:
        if param['calib_model'] == 'joint_ offset':
            offset_0 = np.zeros(param['NbJoint'])
        elif param['calib_model'] == 'full_params':
            offset_0 = np.zeros(param['NbJoint']*6)
    elif mode == 1:
        if param['calib_model'] == 'joint_ offset':
            offset_0 = np.random.uniform(-0.02, 0.02, (param['NbJoint'],))
        elif param['calib_model'] == 'full_params':
            offset_0 = np.random.uniform(-0.02, 0.02, (param['NbJoint']*6,))

    # list of parameters to be set as zero
    # torso_list = [0, 1, 2, 3, 4, 5]
    # arm1_list = [6, 7, 8, 11]
    # arm2_list = [13, 14, 16]
    # arm3_list = [19, 22]
    # arm4_list = [24, 27]
    # arm5_list = [30, 33]
    # arm6_list = [36, 39]
    # arm7_list = [43, 44, 46]  # include phiz7
    torso_list = [0, 1, 2, 3, 4, 5]
    arm1_list = [6, 7, 8, 11]
    arm2_list = [13, 16]
    arm3_list = [19, 22]
    arm4_list = [24, 27]
    arm5_list = [30, 33]
    arm6_list = [36, 39]
    arm7_list = [43, 46]  # include phiz7
    total_list = [torso_list, arm1_list, arm2_list, arm3_list, arm4_list, arm5_list,
                  arm6_list, arm7_list]

    zero_list = []
    for i in range(param['NbJoint']):
        zero_list = [*zero_list, *total_list[i]]
    print("list of elements to be set zero: ", zero_list)
    if base_model == True:
        offset_0 = np.delete(offset_0, zero_list, None)
        # x,y,z,r,p,y from wrist to end_effector ( 6 parameters for pos and orient)
    qEE_0 = np.full((param['NbMarkers']*3,), 0.1)

    # variable vector 20 par = 6 base par + 8 joint off + 6 endeffector par
    var = np.append(np.append(qBase_0, offset_0), qEE_0)
    nvars = var.shape[0]
    return var, nvars


def get_PEE_fullvar(var, q, model, data, param, noise=False, base_model=True):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
        var: consist of full 6 params for each joint (6+6xNbJoints+6xNbMarkers,) since scipy.optimize 
        only takes 1D array variables.
        Reshape to ((1 + NbJoints + NbMarkers, 6))
        Use jointplacement to add offset to 6 axes of joint       

    """
    PEE = []

    # calibration index = 3 or 6, indicating whether or not orientation incld.
    nrow = param['calibration_index']
    ncol = param['NbSample']
    q_temp = np.copy(q)

    # reshape variable vector to vectors of 6
    NbFrames = 1 + param['NbJoint'] + param['NbMarkers']
    if base_model == False:
        var_rs = np.reshape(var, (NbFrames, 6))
    elif base_model == True:
        var_rs = np.zeros((NbFrames, 6))
        # base frame
        # 3D base frame
        # var_rs[0, 0:3] = var[0:3]

        # 6D base frame
        var_rs[0, 0:6] = var[0:6]

        if param['NbJoint'] > 0:
            # torso
            var_rs[1, :] = np.zeros(6)
            if param['NbJoint'] > 1:
                # arm 1
                var_rs[2, 3] = var[6]
                var_rs[2, 4] = var[7]
            if param['NbJoint'] > 2:
                # arm 2
                var_rs[3, 0] = var[8]
                var_rs[3, 2] = var[9]
                var_rs[3, 3] = var[10]
                var_rs[3, 5] = var[11]
                if param['NbJoint'] > 3:
                    # arm3
                    var_rs[4, 0] = var[12]
                    var_rs[4, 2] = var[13]
                    var_rs[4, 3] = var[14]
                    var_rs[4, 5] = var[15]
                    if param['NbJoint'] > 4:
                        # arm4
                        var_rs[5, 1] = var[16]
                        var_rs[5, 2] = var[17]
                        var_rs[5, 4] = var[18]
                        var_rs[5, 5] = var[19]
                        if param['NbJoint'] > 5:
                            # arm5
                            var_rs[6, 1] = var[20]
                            var_rs[6, 2] = var[21]
                            var_rs[6, 4] = var[22]
                            var_rs[6, 5] = var[23]
                            if param['NbJoint'] > 6:
                                # arm6
                                var_rs[7, 1] = var[24]
                                var_rs[7, 2] = var[25]
                                var_rs[7, 4] = var[26]
                                var_rs[7, 5] = var[27]
                                if param['NbJoint'] > 7:
                                    # arm7
                                    var_rs[8, 0] = var[28]
                                    var_rs[8, 2] = var[29]
                                    var_rs[8, 3] = var[30]
                                    var_rs[8, 5] = var[31]

            # # torso
            # var_rs[1, 0:6] = var[6:12]
            # if param['NbJoint'] > 1:
            #     # arm 1
            #     var_rs[2, 3] = var[12]
            #     var_rs[2, 4] = var[13]
            # if param['NbJoint'] > 2:
            #     # arm 2
            #     var_rs[3, 0] = var[13]
            #     var_rs[3, 2] = var[14]
            #     var_rs[3, 3] = var[15]
            #     var_rs[3, 5] = var[16]
            #     if param['NbJoint'] > 3:
            #         # arm3
            #         var_rs[4, 0] = var[17]
            #         var_rs[4, 2] = var[18]
            #         var_rs[4, 3] = var[19]
            #         var_rs[4, 5] = var[20]
            #         if param['NbJoint'] > 4:
            #             # arm4
            #             var_rs[5, 1] = var[21]
            #             var_rs[5, 2] = var[22]
            #             var_rs[5, 4] = var[23]
            #             var_rs[5, 5] = var[24]
            #             if param['NbJoint'] > 5:
            #                 # arm5
            #                 var_rs[6, 1] = var[25]
            #                 var_rs[6, 2] = var[26]
            #                 var_rs[6, 4] = var[27]
            #                 var_rs[6, 5] = var[28]
            #                 if param['NbJoint'] > 6:
            #                     # arm6
            #                     var_rs[7, 1] = var[29]
            #                     var_rs[7, 2] = var[30]
            #                     var_rs[7, 4] = var[31]
            #                     var_rs[7, 5] = var[32]
            #                     if param['NbJoint'] > 7:
            #                         # arm7
            #                         var_rs[8, 0] = var[32]
            #                         var_rs[8, 2] = var[33]
            #                         var_rs[8, 3] = var[34]
            #                         var_rs[8, 5] = var[35]
        # markers

    # # frame trasformation matrix from mocap to base
    base_placement = cartesian_to_SE3(var_rs[0, 0:6])

    for k in range(param['NbMarkers']):
        markerId = 1 + param['NbJoint'] + k
        curr_varId = var.shape[0] - \
            param['NbMarkers']*param['calibration_index']
        var_rs[markerId, 0:param['calibration_index']] = var[(
            curr_varId+k*param['calibration_index']):(curr_varId+(k+1)*param['calibration_index'])]

        PEE_marker = np.empty((nrow, ncol))
        for i in range(ncol):
            config = q_temp[i, :]
            '''
            some fckps here in exceptional case wherer no parameters of the joint are to be variables
            even the joint is active 
            '''
            # update 8 joints
            for j in range(param['NbJoint']):
                joint_placement = cartesian_to_SE3(var_rs[j+1, :])
                model.jointPlacements[j].translation += joint_placement.translation
                model.jointPlacements[j].rotation += joint_placement.rotation

            pin.framesForwardKinematics(model, data, config)
            pin.updateFramePlacements(model, data)

            # calculate oMf from the 1st join tto last joint (wrist)
            lastJoint_name = model.names[param['NbJoint']]
            lastJoint_frameId = model.getFrameId(lastJoint_name)
            inter_placements = data.oMf[lastJoint_frameId]

            # # calculate oMf from wrist to the last frame
            # print("marker row: ", markerId)
            last_placement = cartesian_to_SE3(var_rs[markerId, :])

            new_oMf = base_placement * \
                inter_placements * \
                last_placement  # from wrist to end effector

            # create a matrix containing coordinates of end_effector
            PEE_marker[0:3, i] = new_oMf.translation
            if nrow == 6:
                PEE_rot = new_oMf.rotation
                PEE_marker[3:6, i] = pin.rpy.matrixToRpy(PEE_rot)

            # reset 8 joints
            # update 8 joints
            for j in range(param['NbJoint']):
                joint_placement = cartesian_to_SE3(var_rs[j+1, :])
                model.jointPlacements[j].translation -= joint_placement.translation
                model.jointPlacements[j].rotation -= joint_placement.rotation

        PEE_marker = PEE_marker.flatten('C')
        PEE = np.append(PEE, PEE_marker)
    return PEE


def get_PEE_var(var, q, model, data, param, noise=False):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
        var: consist of geometric parameters from base->1st joint, joint offsets, wirst-> end effector
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
        offset_var: consist of only joint offsets
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


######################## base parameters functions ########################################


def Calculate_kinematics_model(q_i, model, data, IDX_TOOL):
    """ Calculate jacobian matrix and kinematic regressor given ONE configuration.
    """
    # print(mp.current_process())
    #pin.updateGlobalPlacements(model , data)
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)

    J = pin.computeFrameJacobian(
        model, data, q_i, IDX_TOOL, pin.LOCAL)
    R = pin.computeFrameKinematicRegressor(
        model, data, IDX_TOOL, pin.LOCAL)
    return model, data, R, J


def Calculate_identifiable_kinematics_model(q, model, data, param):
    """ Calculate jacobian matrix and kinematic regressor and aggreating into one matrix,
        given a set of configurations or random configurations if not given.
    """
    q_temp = np.copy(q)
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
        q_i[param['NbJoint']:] = param['q0'][param['NbJoint']:]
        # q_i = param['q0']

        model, data, Ri, Ji = Calculate_kinematics_model(
            q_i, model, data, param['IDX_TOOL'])
        for j in range(calib_idx):
            R[param['NbSample']*j + i, :] = Ri[j, :]
    return R


def Calculate_base_kinematics_regressor(q, model, data, param):
    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names)]
    geo_params = get_geoOffset(joint_names)
    # calculate kinematic regressor with random configs
    Rrand = Calculate_identifiable_kinematics_model([], model, data, param)
    # calculate kinematic regressor with input configs
    R = Calculate_identifiable_kinematics_model(q, model, data, param)

    ############## only joint offset parameters ########
    if param['calib_model'] == 'joint_offset':
        # particularly select columns/parameters corresponding to joint and 6 last parameters
        actJoint_idx = [2, 11, 17, 23, 29, 35, 41, 47, 48, 49,
                        50, 51, 52, 53]  # all on z axis - checked!!

        # a dictionary of selected parameters
        gp_listItems = list(geo_params.items())
        geo_params_sel = []
        for i in actJoint_idx:
            geo_params_sel.append(gp_listItems[i])
        geo_params_sel = dict(geo_params_sel)

        # select columns corresponding to joint_idx
        Rrand_sel = Rrand[:, actJoint_idx]

        # select columns corresponding to joint_idx
        R_sel = R[:, actJoint_idx]

    ############## full 6 parameters ###################
    elif param['calib_model'] == 'full_params':
        geo_params_sel = geo_params
        Rrand_sel = Rrand
        R_sel = R

    # obtain a list of column after apply QR decomposition
    Rrand_e, paramsrand_e = eliminate_non_dynaffect(
        Rrand_sel, geo_params_sel, tol_e=1e-6)
    idx_base = get_baseIndex(Rrand_e, paramsrand_e)
    Rrand_b, paramsrand_base = get_baseParams(Rrand_e, paramsrand_e)

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
    _, s, _ = np.linalg.svd(Rrand_b)
    print('shape of observation matrix',
          Rrand.shape, Rrand_e.shape, Rrand_b.shape)
    return Rrand_b, R_b, paramsrand_base, paramsrand_e

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
        J = np.sum(np.square(PEEd-PEEe))+1e-1*1 / \
            np.sum(np.square(x-self.param['x_opt_prev'])+100)

        return J

    def gradient(self, x):
        # callback for gradient

        G = approx_fprime(x, self.objective, self.param['eps_gradient'])

        return G
