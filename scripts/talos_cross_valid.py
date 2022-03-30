import csv
from re import split
import pandas as pd
import numpy as np
import rospy
# import dask.dataframe as dd

from sys import argv
import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pinocchio as pin


def extract_t_list(path_to_tf):
    """ Extracts list of timestampts where samples were recorded
        """
    pass


def test_readCSV(path_to_values, path_to_names):
    # read names and values from csv to dataframe
    dt_names = pd.read_csv(path_to_names)
    dt_values = pd.read_csv(path_to_values)

    dt_values_val = dt_values.loc[:, 'values'].values

    test_msg = dt_values_val[1]
    first_row = test_msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')


def extract_tf(path_to_tf, frame_names):
    """ Extract Qualysis data from tf bag of PAL robots,
        Input:  path_to_tf: path to csv file
                frame_name: list of str, frame defined in Qualisys streamed and recorded in rosbag
        Output: a dictionary
                keys: frame_names /values: 7xn array of [time, xyzquaternion]
    """
    tf_dict = {}

    # create data frame
    df = pd.read_csv(path_to_tf)

    # get collumns names
    frame_col = "child_frame_id"

    # translation
    x_col = "x"
    y_col = "y"
    z_col = "z"

    # orientation
    ux_col = "ux"
    uy_col = "uy"
    uz_col = "uz"
    w_col = "w"

    # time
    sec_col = "secs"
    nsec_col = "nsecs"

    # TODO: check if all names are correctly presented in headers of csv file

    # read values
    frame_val = df.loc[:, frame_col].values

    x_val = df.loc[:, x_col].values
    y_val = df.loc[:, y_col].values
    z_val = df.loc[:, z_col].values

    ux_val = df.loc[:, ux_col].values
    uy_val = df.loc[:, uy_col].values
    uz_val = df.loc[:, uz_col].values
    w_val = df.loc[:, w_col].values

    sec_val = df.loc[:, sec_col].values
    nsec_val = df.loc[:, nsec_col].values

    # t_val (list): extract and covert rostime to second
    t_val = []
    # starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec() # mark up t0
    starting_t = 0
    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)

    # tf_dict (dict): return a dict contain key/item = frame_name(str)/numpy array
    for frame_name in frame_names:
        t = []
        x = []
        y = []
        z = []
        ux = []
        uy = []
        uz = []
        w = []
        for i in range(frame_val.shape[0]):
            if frame_val[i] == frame_name:
                t.append(t_val[i])
                x.append(x_val[i])
                y.append(y_val[i])
                z.append(z_val[i])
                ux.append(ux_val[i])
                uy.append(uy_val[i])
                uz.append(uz_val[i])
                w.append(w_val[i])
        tf_dict[frame_name] = np.transpose(
            np.array([t, x, y, z, ux, uy, uz, w]))
    return tf_dict


def extract_instrospection(path_to_values, path_to_names, value_names=[], t_list=[]):
    """ Extracts joint angles from Introspection Msg data from rosbag -> csv
        value_names: names of values to be extracted
        t_list: selected extracting timestamps
    """
    joint_dict = {}
    # read names and values from csv to dataframe
    dt_names = pd.read_csv(path_to_names)
    dt_values = pd.read_csv(path_to_values)

    # t_val (list): extract and convert rostime to second
    sec_col = "secs"
    nsec_col = "nsecs"
    sec_val = dt_values.loc[:, sec_col].values
    nsec_val = dt_values.loc[:, nsec_col].values
    t_val = []

    # starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec() # mark up t0
    starting_t = 0
    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)

    # t_idx (list): get list of instants where data samples are picked up based on t_list
    # if t_list = [], extract the whole collumn
    if not t_list:
        t_list = t_val

    t_idx = []
    eps = 0.01
    for t in t_list:
        t_min = min(t_val, key=lambda x: abs(x-t))
        if abs(t-t_min) < eps:
            t_idx.append(t_val.index(t_min))

    # names (list): slice names in datanames corressponding to "values" column in datavalues
    names = []
    if dt_names.columns[-1] == "names_version":
        last_col = "names_version"
        if dt_names.columns[7] == "names":
            first_col = "names"
            first_idx = dt_names.columns.get_loc(first_col)
            last_idx = dt_names.columns.get_loc(last_col)
            names = list(dt_names.columns[range(first_idx+1, last_idx)])

    print("total number of columns in data_names: ", len(names))

    # joint_idx (list): get indices of corresponding to active joints
    # if value_names = [], extract all available values
    if not value_names:
        value_names = names

    joint_idx = []
    for element in value_names:
        if element in names:
            joint_idx.append(names.index(element))
        else:
            print(element, "Mentioned joint is not present in the names list.")
            break
    print("Joint indices corresponding to active joints: ", joint_idx)

    # joint_val (np.darray): split data in "values" column (str) to numpy array
    # extracted_val (np.darray):extract only values of interest from joint_val
    extracted_val = np.empty((len(t_idx), len(joint_idx)))
    dt_values_val = dt_values.loc[:, 'values'].values

    test_msg = dt_values_val[1]
    first_row = test_msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')

    if not len(split_data) == len(names):
        print("Names and value collumns did not match!")
    else:
        joint_val = []
        # slicing along axis 0 given t_idx
        for i in t_idx:
            # each msg is A STRING, it needs to be splitted and group into a list of float
            msg = dt_values_val[i]
            first_row = msg.replace('[', '')
            first_row = first_row.replace(']', '')
            row_data = first_row.split(',')
            joint_val.append(row_data)
        joint_val = np.asarray(joint_val, dtype=np.float64)

        # slicing along axis 1 given value_idx
        for i in range(len(joint_idx)):
            extracted_val[:, i] = joint_val[:, joint_idx[i]]
    return extracted_val


def get_data_sample(pos, t_list, eps=0.1):
    """ Extracts data samples give a list of specific instants
    """
    pos_idx = []
    count = 0
    for t in t_list:
        count += 1
        t_min = min(list(pos[:, 0]), key=lambda x: abs(x-t))
        print("deviation of time step: ", abs(t-t_min))

        if abs(t-t_min) < eps:
            curr_idx = list(pos[:, 0]).index(t_min)
            pos_idx.append(curr_idx)
        else:
            print("Missing data at %f" % t)
            print(count)
            break

    pos_sample = np.empty((len(pos_idx), pos.shape[1]))
    for i in range(len(pos_idx)):
        pos_sample[i, :] = pos[pos_idx[i], :]
    return pos_sample

#     # project prj_frame onto ref_frame


def project_frame(prj_frame, ref_frame):
    projected_pos = np.empty((prj_frame.shape[0], 3))
    if prj_frame.shape != ref_frame.shape:
        print("projecting two frames have different sizes! Projected positions are empty!")
    else:
        for i in range(prj_frame.shape[0]):
            ref_se3 = pin.XYZQUATToSE3(ref_frame[i, 1:])
            prj_se3 = pin.XYZQUATToSE3(prj_frame[i, 1:])
            projected_se3 = pin.SE3.inverse(ref_se3)*prj_se3
            projected_pos[i, :] = projected_se3.translation
    return projected_pos


def plot_position(frame, frame_pick,  fig=[]):
    if not fig:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(frame[:, 1], frame[:, 2], frame[:, 3], s=15, color='blue')
    ax.scatter(frame_pick[:, 1], frame_pick[:, 2],
               frame_pick[:, 3], s=80, color='red')


def cal_diff(delta_PEE, param):
    PEE_xyz = delta_PEE.reshape((param['NbMarkers']*3, param["NbSample"]))
    PEE_dist = np.zeros((param['NbMarkers'], param["NbSample"]))
    for i in range(param["NbMarkers"]):
        for j in range(param["NbSample"]):
            PEE_dist[i, j] = np.sqrt(
                PEE_xyz[i*3, j]**2 + PEE_xyz[i*3 + 1, j]**2 + PEE_xyz[i*3 + 2, j]**2)
    return PEE_dist


def save_csv(t, xyz, q, path_to_save, side=''):
    if not side:
        print("Error! Take side!")

    # # talos left arm
    elif side == 'left':
        path_save_ep = join(
            dirname(dirname(str(abspath(__file__)))),
            path_to_save)
        headers = [
            "x1", "y1", "z1",
            "torso1", "torso2", "armL1", "armL2", "armL3", "armL4", "armL5", "armL6", "armL7"]

    # # talos right arm
    elif side == 'right':
        path_save_ep = join(
            dirname(dirname(str(abspath(__file__)))),
            path_to_save)
        headers = [
            "x1", "y1", "z1",
            "torso1", "torso2", "armR1", "armR2", "armR3", "armR4", "armR5", "armR6", "armR7"]

    with open(path_save_ep, "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(headers)
        for i in range(len(t)):
            row = list(np.concatenate((xyz[i, :],
                                      q[i, :])))
            w.writerow(row)


def main():
    # list of instants where data samples are picked up
    # NOTE: cycle is not periodic!!!
    # sample test

    t_pick = []

    t_pick = [52.15, 67.94, 81.95,
              91.166, 113.213, 127.908, 142.4, 157.3]

    t_pick.sort()
    print(t_pick)

    frame_names = ['"waist_frame"', '"endeffector_frame"']

    # extract mocap data
    # Talos

    path_to_tf = '/home/dvtnguyen/calibration/raw_data/talos_mars/2022_03_24/square_rightArm_noOffset_ground_2022-03-24-13-18-34/tf_throttle.csv'
    path_to_values = '/home/dvtnguyen/calibration/raw_data/talos_mars/2022_03_24/square_rightArm_noOffset_ground_2022-03-24-13-18-34/introspection_datavalues_throttle.csv'
    path_to_names = '/home/dvtnguyen/calibration/raw_data/talos_mars/2022_03_24/square_rightArm_noOffset_ground_2022-03-24-13-18-34/introspection_datanames_throttle.csv'
    ###################################### Talos 1 marker ###############
    # get full data
    talos_dict = extract_tf(path_to_tf, frame_names)
    W_pos = talos_dict[frame_names[0]]
    EE_pos = talos_dict[frame_names[1]]
    print("Endeffector and waist frame read from csv: ",
          EE_pos.shape, W_pos.shape)
    # plot_position(EE_pos)
    # plt.show()

    # select only data given timestamps
    t_list = [x + W_pos[0, 0] for x in t_pick]
    print("Endeffector and waist frame read from csv: ",
          EE_pos.shape, W_pos.shape)
    print("length of record:", EE_pos[-1, 0] -
          EE_pos[0, 0], W_pos[-1, 0] - W_pos[0, 0])

    W_sample = get_data_sample(W_pos, t_list)
    EE_sample = get_data_sample(EE_pos, t_list)
    print("Endeffector and waist frame at static postures: ",
          EE_sample.shape, W_sample.shape)
    plot_position(EE_pos, EE_sample)
    plt.show()
    # project endeffector onto waist
    EE_prj_sample = project_frame(EE_sample, W_sample)
    print("projected endefffector: ", EE_prj_sample.shape)
    ########################################################################

    # extract joint configurations data

    # # talos left arm
    # torso_1 = '- torso_1_joint_position'
    # torso_2 = '- torso_2_joint_position'
    # arm_left_1 = '- arm_left_1_joint_position'
    # arm_left_2 = '- arm_left_2_joint_position'
    # arm_left_3 = '- arm_left_3_joint_position'
    # arm_left_4 = '- arm_left_4_joint_position'
    # arm_left_5 = '- arm_left_5_joint_position'
    # arm_left_6 = '- arm_left_6_joint_position'
    # arm_left_7 = '- arm_left_7_joint_position'

    # joint_names = [torso_1, torso_2, arm_left_1, arm_left_2,
    #                arm_left_3, arm_left_4, arm_left_5, arm_left_6, arm_left_7]

    # talos right arm joint position
    torso_1 = '- torso_1_joint_position'
    torso_2 = '- torso_2_joint_position'
    arm_right_1 = '- arm_right_1_joint_position'
    arm_right_2 = '- arm_right_2_joint_position'
    arm_right_3 = '- arm_right_3_joint_position'
    arm_right_4 = '- arm_right_4_joint_position'
    arm_right_5 = '- arm_right_5_joint_position'
    arm_right_6 = '- arm_right_6_joint_position'
    arm_right_7 = '- arm_right_7_joint_position'

    joint_names = [torso_1, torso_2, arm_right_1, arm_right_2,
                   arm_right_3, arm_right_4, arm_right_5, arm_right_6, arm_right_7]

    actJoint_val = extract_instrospection(
        path_to_values, path_to_names, joint_names, t_list)
    print("expectedd NbSamplexNbjoints: ", actJoint_val.shape)

    # talos right arm joint encoder
    # # talos left arm
    # torso_1 = '- torso_1_joint_absolute_encoder_position'
    # torso_2 = '- torso_2_joint_absolute_encoder_position'
    # arm_left_1 = '- arm_left_1_joint_absolute_encoder_position'
    # arm_left_2 = '- arm_left_2_joint_absolute_encoder_position'
    # arm_left_3 = '- arm_left_3_joint_absolute_encoder_position'
    # arm_left_4 = '- arm_left_4_joint_absolute_encoder_position'
    # arm_left_5 = '- arm_left_5_joint_absolute_encoder_position'
    # arm_left_6 = '- arm_left_6_joint_absolute_encoder_position'
    # arm_left_7 = '- arm_left_7_joint_absolute_encoder_position'

    # joint_names = [torso_1, torso_2, arm_left_1, arm_left_2,
    #                arm_left_3, arm_left_4, arm_left_5, arm_left_6, arm_left_7]

    # talos right arm
    torso_1 = '- torso_1_joint_absolute_encoder_position'
    torso_2 = '- torso_2_joint_absolute_encoder_position'
    arm_right_1 = '- arm_right_1_joint_absolute_encoder_position'
    arm_right_2 = '- arm_right_2_joint_absolute_encoder_position'
    arm_right_3 = '- arm_right_3_joint_absolute_encoder_position'
    arm_right_4 = '- arm_right_4_joint_absolute_encoder_position'
    arm_right_5 = '- arm_right_5_joint_absolute_encoder_position'
    arm_right_6 = '- arm_right_6_joint_absolute_encoder_position'
    arm_right_7 = '- arm_right_7_joint_absolute_encoder_position'

    joint_names = [torso_1, torso_2, arm_right_1, arm_right_2,
                   arm_right_3, arm_right_4, arm_right_5, arm_right_6, arm_right_7]

    jointEncoder_val = extract_instrospection(
        path_to_values, path_to_names, joint_names, t_list)
    print("expectedd NbSamplexNbjoints: ", actJoint_val.shape)

    # # write to csv
    # save_csv(t_list, EE_prj_sample, actJoint_val,
    #          f"talos/sample.csv", side='right')

    from tools.robot import Robot

    from tiago_mocap_calib_fun_def import (
        get_param,
        init_var,
        get_PEE_fullvar)
    # 1/ Load robot model and create a dictionary containing reserved constants

    robot = Robot(
        "talos_data/robots",
        "talos_reduced.urdf"
        # "tiago_description/robots",
        # "tiago_no_hand_mod.urdf",
        # isFext=True  # add free-flyer joint at base
    )
    model = robot.model
    data = robot.data

    NbSample = len(t_list)
    param = get_param(
        robot, NbSample, TOOL_NAME='gripper_right_base_link', NbMarkers=1)

    # 2/ q and offset
    q_jp = np.empty((param['NbSample'], model.nq))
    q_en = np.empty((param['NbSample'], model.nq))
    for i in range(param['NbSample']):
        config_jp = param['q0']
        config_jp[param['Ind_joint']] = actJoint_val[i, :]
        q_jp[i, :] = config_jp

        config_en = param['q0']
        config_en[param['Ind_joint']] = jointEncoder_val[i, :]
        q_en[i, :] = config_en

    var_temp, _ = init_var(param, mode=0)
    baseframe_offset = np.array(
        [-0.1271, 0.0127, 0.0325, -0.0011, -0.0687, -0.0426])
    joint_offset = np.array([-0.001010,
                             -0.008193,
                             -0.002009,
                             -0.006550,
                             0.000440,
                             -0.004837,
                             0.000476,
                             -0.000980,
                             0.002468,
                             0.002604,
                             -0.006682,
                             -0.004562,
                             -0.001481,
                             -0.004474,
                             -0.009181,
                             0.016084,
                             0.001558,
                             -0.016645,
                             0.000730,
                             0.005638,
                             -0.013776,
                             0.002017,
                             -0.000611,
                             0.006237,
                             0.007078,
                             -0.009203,
                             -0.000518,
                             - 0.000750,
                             0.000022,
                             0.003164,
                             0.000003,
                             0.003300
                             ])

    var_zero = np.copy(var_temp)
    var_zero[0:6] = baseframe_offset

    var_offset = np.copy(var_temp)
    var_offset[0:6] = baseframe_offset
    var_offset[6:-3] = joint_offset

    PEE_jp_zero = get_PEE_fullvar(
        var_zero, q_jp, model, data, param)
    PEE_en_zero = get_PEE_fullvar(
        var_zero, q_en, model, data, param)
    PEE_jp_offset = get_PEE_fullvar(
        var_offset, q_jp, model, data, param)
    PEE_en_offset = get_PEE_fullvar(
        var_offset, q_en, model, data, param)

    # 3/ calculate the difference
    # between joint position and encoder
    PEE_errZero = cal_diff(PEE_jp_zero - PEE_en_zero, param) - \
        np.mean(cal_diff(PEE_jp_zero - PEE_en_zero, param))
    PEE_errOffset = cal_diff(PEE_jp_offset - PEE_en_offset, param) - \
        np.mean(cal_diff(PEE_jp_offset - PEE_en_offset, param))

    # plot
    ax1 = plt.subplot(111)
    w = 0.2
    x = np.arange(param["NbSample"])
    bar_zero = ax1.bar(
        x-w/2, PEE_errZero[0, :], width=w, color='b', align='center')
    bar_offset = ax1.bar(x+w/2, PEE_errOffset[0, :], width=w,
                         color='r', align='center')
    pt_name = [
        'C1',
        'M1',
        'C2',
        'M2',
        'C3',
        'M3',
        'C4',
        'M4']
    ax1.set_ylabel('Error (m)')
    ax1.set_xticklabels(pt_name)

    ax1.legend((bar_zero[0], bar_offset[0]),
               ('zero offset', 'mocap-based offset'))
    # between measured and estimated
    # PEE_m = np.transpose(EE_prj_sample).flatten('C')
    # PEE_merr_zero = cal_diff(PEE_m - PEE_jp_zero, param) - \
    #     np.mean(cal_diff(PEE_m - PEE_jp_zero, param))
    # PEE_merr_offset = cal_diff(
    #     PEE_m - PEE_jp_offset, param) - \
    #     np.mean(cal_diff(PEE_m - PEE_jp_offset, param))
    # print(np.linalg.norm(0.3*PEE_merr_offset[0, :])/np.sqrt(param['NbSample']),
    #       np.linalg.norm(0.3*PEE_merr_zero[0, :])/np.sqrt(param['NbSample']))
    # ax2 = plt.subplot(111)
    # w = 0.2
    # x = np.arange(param["NbSample"])
    # bar_zero = ax2.bar(
    #     x-w/2, 0.3*PEE_merr_zero[0, :], width=w, color='b', align='center')
    # bar_offset = ax2.bar(x+w/2, 0.3*PEE_merr_offset[0, :], width=w,
    #                      color='r', align='center')
    # pt_name = [
    #     'C1',
    #     'M1',
    #     'C2',
    #     'M2',
    #     'C3',
    #     'M3',
    #     'C4',
    #     'M4']
    # ax2.set_ylabel('Error (m)')
    # ax2.set_xticklabels(pt_name)

    # ax2.legend((bar_zero[0], bar_offset[0]),
    #            ('zero offset', 'mocap-based offset'))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()


# ax = plt.subplot(111)
# w = 0.2
# x = np.arange(8)
# pt_name = ['center',
#            'C1',
#            'M1',
#            'C2',
#            'M2',
#            'C3',
#            'M3',
#            'C4',
#            'M4']
# bar_zero = ax.bar(x-w, res_zero, width=w, color='b', align='center')
# bar_pal = ax.bar(x, res_pal, width=w, color='g', align='center')
# bar_mc = ax.bar(x+w, res_mc, width=w, color='r', align='center')

# ax.set_ylabel('Error (m)')
# ax.set_xticklabels(pt_name)

# ax.legend((bar_zero[0], bar_pal[0], bar_mc[0]),
#           ('zero offset', 'pal offset', 'mocap-based offset'))

# plt.grid()
# plt.show()
