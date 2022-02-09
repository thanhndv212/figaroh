import csv
from re import split
import pandas as pd
import numpy as np
import rospy
import dask.dataframe as dd

from sys import argv
import os
from os.path import dirname, join, abspath

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# for msg in read_values:
#     first_row = msg.replace('[', '')
#     first_row = first_row.replace(']', '')
#     split_data = first_row.split(',')
#     if len(split_data) - 4 == len(d):  # ignore first 4 columns
#         for i in range(4, len(split_data)):
#             # split_data[i] = float(split_data[i])
#             d[names[i-4]].append(float(split_data[i]))


def extract_tfbag(path_to_csv, frame_names):
    """ Extract Qualysis data from tf bag of PAL robots,
        frame_name: list of str, frame defined in Qualisys streamed and recorded in rosbag
        output: traslation x, y, z
                rotation roll, pitch, yaw
    """
    tf_dict = {}

    # create data frame
    df = pd.read_csv(path_to_csv)

    # get collumns

    frame_col = "child_frame_id"

    # translation
    x_col = "x"
    y_col = "y"
    z_col = "z"

    # time
    sec_col = "secs"
    nsec_col = "nsecs"

    # read values
    frame_val = df.loc[:, frame_col].values
    x_val = df.loc[:, x_col].values
    y_val = df.loc[:, y_col].values
    z_val = df.loc[:, z_col].values
    sec_val = df.loc[:, sec_col].values
    nsec_val = df.loc[:, nsec_col].values

    # t_val (list): extract and covert rostime to second
    t_val = []
    starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec()

    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)

    # tf_dict (dict): return a dict contain key/item = frame_name(str)/numpy array

    for frame_name in frame_names:
        t = []
        x = []
        y = []
        z = []
        for i in range(frame_val.shape[0]):
            if frame_val[i] == frame_name:
                t.append(t_val[i])
                x.append(x_val[i])
                y.append(y_val[i])
                z.append(z_val[i])
        tf_dict[frame_name] = np.array([t, x, y, z])
    return tf_dict


def extract_joint_pos(path_to_values, path_to_names, t_list=[5, 10, 15, 20]):
    """ Extracts joint angles
    """
    joint_dict = {}
    # read names and values from csv to dataframe
    dt_names = pd.read_csv(path_to_names)
    dt_values = pd.read_csv(path_to_values)

    # time
    sec_col = "secs"
    nsec_col = "nsecs"

    # t_val (list): extract and convert rostime to second
    sec_val = dt_values.loc[:, sec_col].values
    nsec_val = dt_values.loc[:, nsec_col].values
    t_val = []
    starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec()

    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)
    # t_idx (list): get list of instants where data samples are picked up
    t_idx = []
    eps = 0.001
    for t in t_list:
        t_min = min(t_val, key=lambda x: abs(x-t))
        if abs(t-t_min) < eps:
            t_idx.append(t_val.index(t_min))

    # joint names

    torso_1 = '- torso_1_joint_position'
    torso_2 = '- torso_2_joint_position'
    arm_left_1 = '- arm_left_1_joint_position'
    arm_left_2 = '- arm_left_2_joint_position'
    arm_left_3 = '- arm_left_3_joint_position'
    arm_left_4 = '- arm_left_4_joint_position'
    arm_left_5 = '- arm_left_5_joint_position'
    arm_left_6 = '- arm_left_6_joint_position'
    arm_left_7 = '- arm_left_7_joint_position'

    joint_names = [torso_1, torso_2, arm_left_1, arm_left_2,
                   arm_left_3, arm_left_4, arm_left_5, arm_left_6, arm_left_7]

    # names (list): slice names in datanames corressponding to "values" column in datavalues
    names = []
    if dt_names.columns[-1] == "names_version":
        last_col = "names_version"
        if dt_names.columns[7] == "names":
            first_col = "names"
            first_idx = dt_names.columns.get_loc(first_col)
            last_idx = dt_names.columns.get_loc(last_col)
            names = list(dt_names.columns[range(first_idx+1, last_idx)])

    # joint_idx (list): get indices of corresponding to active joints
    joint_idx = []
    for element in joint_names:
        if element in names:
            joint_idx.append(names.index(element))
        else:
            print("Mentioned joint is not present in the names list.")
            print(element)
            break

    # joint_val (np.darray): split data in "values" column (str) to numpy array
    dt_values_val = dt_values.loc[:, 'values'].values

    test_msg = dt_values_val[1]
    first_row = test_msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')
    print(np.asarray([split_data], dtype=np.float64).shape)

    if len(split_data) == len(names):
        # joint_val = np.empty((0, len(names)), dtype=np.float64)
        joint_val = []
        for i in t_idx:
            # each msg is A STRING, it needs to be splitted and group into a list of float
            msg = dt_values_val[i]
            first_row = msg.replace('[', '')
            first_row = first_row.replace(']', '')
            split_data = first_row.split(',')
            # curr_spl = np.asarray([split_data], dtype=np.float64)
            # joint_val = np.vstack((joint_val, curr_spl))
            joint_val.append(split_data)

        joint_val = np.asarray(joint_val, dtype=np.float64)

    # extract only active joint angle values from joint_val
    actJoint_val = np.empty((len(t_idx), len(joint_idx)))
    for i in range(len(joint_idx)):
        actJoint_val[:, i] = joint_val[:, joint_idx[i]]
    print(actJoint_val.shape)
    return actJoint_val


def main():
    # list of instants where data samples are picked up
    # NOTE: cycle is not periodic!!!
    # t0 = [20., 37.5, 54.]
    # period = 49
    # NbSample = 9
    # t_list = []
    # for i in t0:
    #     for j in range(NbSample):
    #         t_list.append(i + j*period)
    t_list = [20., 37.5, 54.,
              70.35, 88., 105.7,
              121.5, 139.2, 155.55,
              174.74, 189.5, 207.2,
              224.95, 242.65, 258.46,
              276.164, 291.95, 308.71,
              327.37, 342.69, 360.]
    t_list.sort()
    print(t_list)
    # extract mocap data
    path_to_csv = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_2_contact_gripper_2022-02-04-14-42-37/tf.csv'
    frame_names = ['"waist_frame"', '"left_hand_frame"']
    talos_dict = extract_tfbag(path_to_csv, frame_names)

    # TODO: project end effector onto waist frame, pick up data samples at exact t_list

    # # extract joint configurations data
    path_to_values = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_2_contact_gripper_2022-02-04-14-42-37/introspection_datavalues.csv'
    path_to_names = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_2_contact_gripper_2022-02-04-14-42-37/introspection_datanames.csv'
    actJoint_val = extract_joint_pos(path_to_values, path_to_names, t_list)

    W_pos = talos_dict[frame_names[0]]
    LH_pos = talos_dict[frame_names[1]]

    # waist frame
    Wt_idx = []
    eps = 0.01
    for t in t_list:
        Wt_min = min(list(W_pos[0, :]), key=lambda x: abs(x-t))
        if abs(t-Wt_min) < eps:
            curr_idx = list(W_pos[0, :]).index(Wt_min)
            Wt_idx.append(curr_idx)

    W_sample = np.empty((len(Wt_idx), W_pos.shape[0]))
    for i in range(len(Wt_idx)):
        W_sample[i, :] = W_pos[:, Wt_idx[i]]

    # left hand frame
    LHt_idx = []
    eps = 0.01

    for t in t_list:
        LHt_min = min(LH_pos[0, :], key=lambda x: abs(x-t))
        if abs(t-LHt_min) < eps:
            curr_idx = list(LH_pos[0, :]).index(LHt_min)
            LHt_idx.append(curr_idx)

    LH_sample = np.empty((len(LHt_idx), LH_pos.shape[0]))
    for i in range(len(LHt_idx)):
        LH_sample[i, :] = LH_pos[:, LHt_idx[i]]

    # project left hand frame onto waist frame
    LH_W_sample = np.empty_like(LH_sample)
    LH_W_sample[:, 0] = LH_sample[:, 0]
    LH_W_sample[:, 1] = LH_sample[:, 1] - W_sample[:, 1]
    LH_W_sample[:, 2] = LH_sample[:, 2] - W_sample[:, 2]
    LH_W_sample[:, 3] = LH_sample[:, 3] - W_sample[:, 3]

    # plot

    # fig = plt.figure()
    # ax = fig.subplots(1, 1)

    # ax.plot(LH_pos[3, :])
    # ax.scatter(LHt_idx, LH_sample[:, 3])
    # plt.show()

    # TODO:  write to csv file
    # write in order of (t,x,y,z,q_torso1, q_torso2, ..., q_arm7)
    path_save_ep = join(
        dirname(dirname(str(abspath(__file__)))),
        f"talos/talos_feb_arm_2_contact.csv")
    with open(path_save_ep, "w") as output_file:
        w = csv.writer(output_file)
        for i in range(len(t_list)):
            row = list(np.append(LH_W_sample[i, :], actJoint_val[i, :]))
            w.writerow(row)


if __name__ == '__main__':
    main()
