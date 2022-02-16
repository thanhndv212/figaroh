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

# create values storing dictionary 'param'

# TODO: define in config file, parse from there


def get_param(robot, NbSample, TOOL_NAME='ee_marker_joint', NbMarkers=1,  calib_model='full_params', calib_idx=3):
    tool_FrameId = robot.model.getFrameId(TOOL_NAME)
    parentJoint2Tool_Id = robot.model.frames[tool_FrameId].parent
    NbJoint = parentJoint2Tool_Id  # joint #0 is  universe
    print("number of active joint: ", NbJoint)
    print("tool name: ", TOOL_NAME)
    print("parent joint of tool frame: ", robot.model.names[NbJoint])
    print("number of markers: ", NbMarkers)
    print("calibration model: ", calib_model)
    param = {
        'q0': np.array(robot.q0),
        'x_opt_prev': np.zeros([NbJoint]),
        'NbSample': NbSample,
        'IDX_TOOL': tool_FrameId,
        'eps': 1e-3,
        'Ind_joint': np.arange(NbJoint),
        'PLOT': 0,
        'NbMarkers': NbMarkers,
        'calib_model': calib_model,  # 'joint_offset' / 'full_params'
        'calibration_index': calib_idx,  # 3 / 6
        'NbJoint': NbJoint
    }
    return param

# TODO: to add to a class


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

# TODO: to add to a class


def get_geoOffset(joint_names):
    """ This function give a dictionary of variations (offset) of kinematics parameters.
            Input:  joint_names: a list of joint names (from model.names)
            Output: geo_params: a dictionary of variations of kinematics parameters.
    """
    tpl_names = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
    geo_params = []

    for i in range(len(joint_names)):
        for j in tpl_names:
            # geo_params.append(j + ("_%d" % i))
            geo_params.append(j + "_" + joint_names[i])

    phi_gp = [0] * len(geo_params)  # default zero values
    geo_params = dict(zip(geo_params, phi_gp))
    return geo_params

# TODO: to add to tools


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

# TODO: to add to a data class (extracting, inspecting, plotting)


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


def extract_expData4Mkr(path_to_file, param, del_list=[]):
    """ Read a csv file into dataframe by pandas, then transform to the form
    of full joint configuration and markers' position/location.
    NOTE: indices matter! Pay attention.
        Input:  path_to_file: str, path to csv
                param: Param, a class contain neccesary constant info.
        Output: np.ndarray, joint configs
                1D np.ndarray, markers' position/location 
        Csv headers: 
                i-th marker position: xi, yi, zi
                i-th marker orientation: phixi, phiyi, phizi (not used atm)
                active joint angles: 
                    Tiago: torso, arm1, arm2, arm3, arm4, arm5, arm6, arm7
                    Talos: torso1, torso2, armL1, armL2, armL3, armL4, armL5, armL6, armL7
    """
    # list of "bad" data samples of Tiago exp data
    # del_list = [4, 8, -3, -1] # calib_data_oct
    # del_list = [2, 26, 39]  # calib_nov_64
    # del_list = [1, 2, 4, 5, 10, 13, 19, 22,
    #             23, 24, 28, 33, -3, -2]  # clean Nov 30
    # del_list = [13, 28]  # no clean Nov 30

    # list of "bad" data samples of Talos exp data
    del_list = [0, 4, 34, 46, 50, 55, 56, 57, 58]
    # # first 12 cols: xyz positions of 4 markers
    # xyz_4Mkr = np.delete(pd.read_csv(
    #     path_to_file, usecols=list(range(0, param['NbMarkers']*3))).to_numpy(), del_list, axis=0)
    # print('csv read', xyz_4Mkr.shape)

    # # extract measured end effector coordinates
    # PEEm_exp = xyz_4Mkr.T
    # PEEm_exp = PEEm_exp.flatten('C')

    # # next 8 cols: joints position
    # q_act = np.delete(pd.read_csv(path_to_file, usecols=list(
    #     range(12, 20))).to_numpy(), del_list, axis=0)
    # param['NbSample'] = q_act.shape[0]

    # # extract measured joint configs
    # q_exp = np.empty((param['NbSample'], param['q0'].shape[0]))
    # for i in range(param['NbSample']):
    #     q_exp[i, 0:8] = q_act[i, :]
    #     # ATTENTION: need to check robot.q0 vs TIAGo.q0
    #     q_exp[i, 8:] = param['q0'][8:]

    # new fixes:
    # read_csv
    df = pd.read_csv(path_to_file)

    # create headers for marker position
    PEE_headers = []
    if param['calibration_index'] == 3:
        for i in range(param['NbMarkers']):
            PEE_headers.append('x%s' % str(i+1))
            PEE_headers.append('y%s' % str(i+1))
            PEE_headers.append('z%s' % str(i+1))

    joint_headers = []
    # create headers for joint configurations
    if param['robot_name'] == "Tiago":
        joint_headers = ['torso', 'arm1', 'arm2', 'arm3', 'arm4',
                         'arm5', 'arm6', 'arm7']
    elif param['robot_name'] == "Talos":
        joint_headers = ['torso1', 'torso2', 'armL1', 'armL2', 'armL3',
                         'armL4', 'armL5', 'armL6', 'armL7']
    # check if all created headers present in csv file
    csv_headers = list(df.columns)

    for header in (PEE_headers + joint_headers):
        if header not in csv_headers:
            print("Headers for extracting data is wrongly defined!")
            break

    # Extract marker position/location
    xyz_4Mkr = df[PEE_headers].to_numpy()

    # Extract joint configurations
    q_act = df[joint_headers].to_numpy()

    # remove bad data
    if del_list:
        xyz_4Mkr = np.delete(xyz_4Mkr, del_list, axis=0)
        q_act = np.delete(q_act, del_list, axis=0)
    # update number of data points
    param['NbSample'] = q_act.shape[0]

    PEEm_exp = xyz_4Mkr.T
    PEEm_exp = PEEm_exp.flatten('C')

    q_exp = np.empty((param['NbSample'], param['q0'].shape[0]))
    if param['robot_name'] == 'Tiago':
        pass
    elif param['robot_name'] == 'Talos':
        for i in range(param['NbSample']):
            config = param['q0']
            config[param['Ind_joint']] = q_act[i, :]
            q_exp[i, :] = config
    return PEEm_exp, q_exp

# TODO: to add to tools


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

# TODO: to crete a linear regression class (parent), LM  as a child class

######################## LM least squares functions ########################################


def init_var(param, mode=0, base_model=True):
    ''' Creates variable vector, mode = 0: initial guess, mode = 1: predefined values(randomized)
    '''
    # x,y,z,r,p,y from mocap to base ( 6 parameters for pos and orient)
    # create artificial offsets
    if mode == 0:
        # 6D base frame
        qBase_0 = np.array([0.0, 0., 0., 0., 0., 0.])
        # parameter variation at joints
        if param['calib_model'] == 'joint_ offset':
            offset_0 = np.zeros(param['NbJoint'])
        elif param['calib_model'] == 'full_params':
            offset_0 = np.zeros(param['NbJoint']*6)
        # markers variables
        qEE_0 = np.full((param['NbMarkers']*param['calibration_index'],), 0.0)

    elif mode == 1:
        # 6D base frame
        qBase_0 = np.array([-0.16, 0.047, 0.16, 0., 0., 0.])
        # parameter variation at joints
        if param['calib_model'] == 'joint_ offset':
            offset_0 = np.random.uniform(-0.005, 0.005, (param['NbJoint'],))
        elif param['calib_model'] == 'full_params':
            offset_0 = np.random.uniform(-0.005, 0.005, (param['NbJoint']*6,))
        # markers variables
        qEE_0 = np.full((param['NbMarkers']*param['calibration_index'],), 0)
    # robot_name = "Tiago"
    # robot_name = "Talos"
    if param['robot_name'] == 'Tiago':
        # create list of parameters to be set as zero for Tiago, respect the order
        # TODO: to be imported from a config file
        torso_list = [0, 1, 2, 3, 4, 5]
        arm1_list = [6, 7, 8, 11]
        arm2_list = [13, 16]
        arm3_list = [19, 22]
        arm4_list = [24, 27]
        arm5_list = [30, 33]
        arm6_list = [36, 39]
        arm7_list = [43, 46]  # include phiz7
        total_list = [torso_list, arm1_list, arm2_list, arm3_list, arm4_list,
                      arm5_list, arm6_list, arm7_list]
    elif param['robot_name'] == 'Talos':
        # create list of parameters to be set as zero for Tiago, respect the order
        # TODO: to be imported from a config file
        torso1_list = [0, 1, 2, 3, 4, 5]
        torso2_list = [2, 5]
        arm1_list = [1, 4]
        arm2_list = [2, 5]
        arm3_list = [0, 3]
        arm4_list = [2, 5]
        arm5_list = [1, 4]
        arm6_list = [2, 5]
        arm7_list = [0, 3]  # include phiz7
        total_list = [torso1_list, torso2_list, arm1_list, arm2_list,
                      arm3_list, arm4_list, arm5_list, arm6_list, arm7_list]
        for i in range(len(total_list)):
            total_list[i] = np.array(total_list[i]) + i*6
    zero_list = np.concatenate(total_list)
    print("list of elements to be set zero: ", zero_list)

    # remove parameters are set to zero (dependent params)
    if base_model == True:
        offset_0 = np.delete(offset_0, zero_list, None)
        # x,y,z,r,p,y from wrist to end_effector ( 6 parameters for pos and orient)

    var = np.append(np.append(qBase_0, offset_0), qEE_0)
    nvars = var.shape[0]
    return var, nvars


def get_PEE_fullvar(var, q, model, data, param, noise=False, base_model=True):
    """ Calculates corresponding cordinates of end_effector, given a set of joint configurations
        var: an 1D array containing estimated offset parameters since scipy.optimize
        only takes 1D array variables. Reshape to ((1 + NbJoints + NbMarkers, 6)), replacing zeros 
        those missing.
        base_model: bool, choice to choose using base parameters to estimate ee's coordinates.
        Use jointplacement to add offset to 6 axes of joint
    """
    PEE = []

    # calibration index = 3 or 6, indicating whether or not orientation incld.
    nrow = param['calibration_index']
    ncol = param['NbSample']
    q_temp = np.copy(q)

    # reshape variable vector to vectors of 6
    NbFrames = 1 + param['NbJoint'] + param['NbMarkers']

    if param['robot_name'] == 'Tiago':
        if not base_model:
            var_rs = np.reshape(var, (NbFrames, 6))
        elif base_model:
            var_rs = np.zeros((NbFrames, 6))

            # first 6 params of base frame var_rs[0:6,:]
            var_rs[0, 0: 6] = var[0: 6]

            # offset parameters var_rs[1:(1+Nbjoint),:]
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
            # 32 base parameters for tiago, first 6 assigned to base frame

    elif param['robot_name'] == 'Talos':
        if not base_model:
            var_rs = np.reshape(var, (NbFrames, 6))
        elif base_model:
            var_rs = np.zeros((NbFrames, 6))

            # 6D base frame
            var_rs[0, 0: 6] = var[0: 6]

            # offset parameters var_rs[1:(1+Nbjoint),:]
            if param['NbJoint'] > 0:
                # torso_1
                var_rs[1, :] = np.zeros(6)
                if param['NbJoint'] > 1:
                    #  torso_2
                    var_rs[2, 0] = var[6]
                    var_rs[2, 1] = var[7]
                    var_rs[2, 3] = var[8]
                    var_rs[2, 4] = var[9]
                    if param['NbJoint'] > 2:
                        # arm1
                        var_rs[3, 0] = var[10]
                        var_rs[3, 2] = var[11]
                        var_rs[3, 3] = var[12]
                        var_rs[3, 5] = var[13]
                        if param['NbJoint'] > 3:
                            # arm2
                            var_rs[4, 0] = var[14]
                            var_rs[4, 1] = var[15]
                            var_rs[4, 3] = var[16]
                            var_rs[4, 4] = var[17]
                            if param['NbJoint'] > 4:
                                # arm3
                                var_rs[5, 1] = var[18]
                                var_rs[5, 2] = var[19]
                                var_rs[5, 4] = var[20]
                                var_rs[5, 5] = var[21]
                                if param['NbJoint'] > 5:
                                    # arm4
                                    var_rs[6, 0] = var[22]
                                    var_rs[6, 1] = var[23]
                                    var_rs[6, 3] = var[24]
                                    var_rs[6, 4] = var[25]
                                    if param['NbJoint'] > 6:
                                        # arm5
                                        var_rs[7, 0] = var[26]
                                        var_rs[7, 2] = var[27]
                                        var_rs[7, 3] = var[28]
                                        var_rs[7, 5] = var[29]
                                        if param['NbJoint'] > 7:
                                            # arm6
                                            var_rs[8, 0] = var[30]
                                            var_rs[8, 1] = var[31]
                                            var_rs[8, 3] = var[32]
                                            var_rs[8, 4] = var[33]
                                            if param['NbJoint'] > 8:
                                                # arm6
                                                var_rs[9, 1] = var[34]
                                                var_rs[9, 2] = var[35]
                                                var_rs[9, 4] = var[36]
                                                var_rs[9, 5] = var[37]

            # 38 base parameters for talos torso-arm, first 6 assigned to base frame

    # frame trasformation matrix from mocap to base
    base_placement = cartesian_to_SE3(var_rs[0, 0: 6])

    for k in range(param['NbMarkers']):
        """ The last calibration_index(3/6)*NbMarkers of var array to be assigned to 
            the last NbMarkers rows of var_rs
        """
        # kth marker's index in var_rs
        markerId = 1 + param['NbJoint'] + k

        # beginning index belongs to markers in var
        curr_varId = var.shape[0] - \
            param['NbMarkers']*param['calibration_index']

        # kth marker frame var_rs[1+NbJoint+k]
        var_rs[markerId, 0: param['calibration_index']] = var[(
            curr_varId+k*param['calibration_index']): (curr_varId+(k+1)*param['calibration_index'])]

        PEE_marker = np.empty((nrow, ncol))
        for i in range(ncol):
            config = q_temp[i, :]
            '''
            some fckps here in exceptional case wherer no parameters of the joint are to be variables
            even the joint is active
            '''
            # update joint geometric parameters with values stored in var_rs
            # NOTE: jointPlacements modify the model of robot, revert the
            # updates after done calculatioin
            for j, joint_idx in enumerate(param['actJoint_idx']):
                joint_placement = cartesian_to_SE3(var_rs[j+1, :])
                # model.jointPlacements[j].translation += joint_placement.translation
                # model.jointPlacements[j].rotation += joint_placement.rotation (matrix addition => possibly wrong)
                temp_translation = model.jointPlacements[joint_idx].translation
                model.jointPlacements[joint_idx].translation += var_rs[j+1, 0: 3]
                new_rpy = pin.rpy.matrixToRpy(
                    model.jointPlacements[joint_idx].rotation) + var_rs[j+1, 3:6]
                model.jointPlacements[joint_idx].rotation = pin.rpy.rpyToMatrix(
                    new_rpy)
                after_translation = model.jointPlacements[joint_idx].translation
            pin.framesForwardKinematics(model, data, config)
            pin.updateFramePlacements(model, data)

            # calculate oMf from the 1st join tto last joint (wrist)
            lastJoint_name = model.names[param['tool_joint']]
            lastJoint_frameId = model.getFrameId(lastJoint_name)
            inter_placements = data.oMf[lastJoint_frameId]

            # # calculate oMf from wrist to the last frame
            # print("marker row: ", markerId)
            last_placement = cartesian_to_SE3(var_rs[markerId, :])

            new_oMf = base_placement * \
                inter_placements * \
                last_placement  # from baseframe -> joints -> last frame

            # create a 2D array containing coordinates of end_effector
            # calibration_index = 3
            PEE_marker[0:3, i] = new_oMf.translation
            # calibrtion_index = 6
            if nrow == 6:
                PEE_rot = new_oMf.rotation
                PEE_marker[3:6, i] = pin.rpy.matrixToRpy(PEE_rot)

            # revert the updates above/restore original state of robot.model
            for j, joint_idx in enumerate(param['actJoint_idx']):
                joint_placement = cartesian_to_SE3(var_rs[j+1, :])
                # model.jointPlacements[j].translation -= joint_placement.translation
                # model.jointPlacements[j].rotation -= joint_placement.rotation (matrix addition => possibly wrong)
                model.jointPlacements[joint_idx].translation -= var_rs[j+1, 0:3]
                update_rpy = pin.rpy.matrixToRpy(
                    model.jointPlacements[joint_idx].rotation) - var_rs[j+1, 3:6]
                model.jointPlacements[joint_idx].rotation = pin.rpy.rpyToMatrix(
                    update_rpy)

            # TODO: to add changement of joint placements after reverting back to original version
            # stop process, send warning message!!

        # flatten ee's coordinates to 1D array
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
    # pin.updateGlobalPlacements(model , data)
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
            q_rand = pin.randomConfiguration(model)
            q_i = param['q0']
            q_i[param['Ind_joint']] = q_rand[param['Ind_joint']]
        else:
            q_i = q_temp[i, :]
        # q_i[param['NbJoint']:] = param['q0'][param['NbJoint']:]
        # q_i = param['q0']

        model, data, Ri, Ji = Calculate_kinematics_model(
            q_i, model, data, param['IDX_TOOL'])
        for j in range(calib_idx):
            R[param['NbSample']*j + i, :] = Ri[j, :]
    return R


def Calculate_base_kinematics_regressor(q, model, data, param):
    # obtain joint names
    joint_names = [name for i, name in enumerate(model.names[1:])]
    geo_params = get_geoOffset(joint_names)
    # print("joint_names: ", joint_names)
    # print("geo_params: ", geo_params)
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
    print("remained parameters: ", paramsrand_e)
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

        # PEEd_all = self.param['PEEd']
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
        # J0 = approx_fprime(x, self.constraints, self.param['eps_gradient'])

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
