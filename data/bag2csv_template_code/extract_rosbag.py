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


def extract_tfbag(path_to_csv, frame_names):
    """ Extract Qualysis data from tf bag of PAL robots,
        Input:  path_to_csv: path to csv file
                frame_name: list of str, frame defined in Qualisys streamed and recorded in rosbag
        Output: a dictionary
                keys: frame_names /values: [time, xyzquaternion]
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
    # starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec()
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
        tf_dict[frame_name] = np.array([t, x, y, z, ux, uy, uz, w])
    return tf_dict


def extract_joint_pos(path_to_values, path_to_names, joint_names=[], t_list=[]):
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
    # starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec()
    starting_t = 0
    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)

    # t_idx (list): get list of instants where data samples are picked up
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

    print(len(names))
    # joint_idx (list): get indices of corresponding to active joints
    joint_idx = []
    for element in joint_names:
        if element in names:
            joint_idx.append(names.index(element))
        else:
            print(element, "Mentioned joint is not present in the names list.")
            break
    print(joint_idx)

    # joint_val (np.darray): split data in "values" column (str) to numpy array
    joint_val = []

    dt_values_val = dt_values.loc[:, 'values'].values

    test_msg = dt_values_val[1]
    first_row = test_msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')

    if len(split_data) == len(names):
        print("herreeeeee")
        # joint_val = np.empty((0, len(names)), dtype=np.float64)
        for i in t_idx:
            # each msg is A STRING, it needs to be splitted and group into a list of float
            msg = dt_values_val[i]
            first_row = msg.replace('[', '')
            first_row = first_row.replace(']', '')
            row_data = first_row.split(',')
            # curr_spl = np.asarray([split_data], dtype=np.float64)
            # joint_val = np.vstack((joint_val, curr_spl))
            joint_val.append(row_data)
    joint_val = np.asarray(joint_val, dtype=np.float64)

    # actJoint_val (np.darray):extract only active joint angle values from joint_val
    actJoint_val = np.empty((len(t_idx), len(joint_idx)))
    for i in range(len(joint_idx)):
        actJoint_val[:, i] = joint_val[:, joint_idx[i]]
    print(actJoint_val.shape)
    return actJoint_val


def main():
    # list of instants where data samples are picked up
    # NOTE: cycle is not periodic!!!
    # sample test
    t_pick = [207.6, 224.3, 242.5, 258.5,  295.27,
              308.7, 326.6, 344.5, 361.5, 376.7, 365.5, 410.7,
              428.6, 445.6, 460.8, 476.9, 496.6, 513.6, 531.5,
              550.3, 563.7, 583.3, 601., 617.3, 631.7, 650.5, 664.8,
              682.7, 701.5, 717.6, 737.3, 752.5, 768.6, 786.5, 802.6
              ]

    # 07/02/2022 air
    # t_pick = [22.4, 38.6, 56.3, 74.1, 90.6, 107., 125.7, 142.9, 158.2, 175.9, 192.4, 210.1, 226.6, 244.4,
    #           262.5, 279., 295.5, 313.5]
    # 09/02/2022 contact
    # t_pick = [20., 37.5, 54.,
    #           70.35, 88., 105.7,
    #           121.5, 139.2, 155.55,
    #           174.74, 189.5, 207.2,
    #           224.95, 242.65, 258.46,
    #           276.164, 291.95, 308.71,
    #           327.37, 342.69, 360.]
    # 10/02/2022 air
    # t_pick = [25., 42.4, 57.6, 75.,
    #           92.4, 109.7, 126.1, 143.5,
    #           160.88, 178.2, 193.5, 212.,
    #           229.37, 245.68, 262.2, 279.37,
    #           297.86, 315.25, 331.55, 350.,
    #           366.34, 382.65, 400., 416.35,
    #           433.74, 451.13, 468.53, 484.83,
    #           503.31, 553.32,
    #           570.71, 588.1, 604.41, 621.8,
    #           640.28, 655.5, 672.9, 690.29,
    #           708.77, 725.07, 742.47, 758.77,
    #           776.16, 792.47, 809.87, 826.17,
    #           843.57, 860.96, 877.26, 894.66,
    #           913.14, 928.36, 945.75, 963.14,
    #           980.54, 996.84, 1014.24, 1031.63,
    #           1050.11, 1065.33, 1093.81, 1100.12]
    t_pick.sort()
    print(t_pick)

    # extract mocap data
    # path_to_csv = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_2_contact_gripper_2022-02-04-14-42-37/tf.csv'
    # path_to_csv = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_1_air_2022-02-04-13-57-29/tf.csv'
    path_to_csv = '/home/thanhndv212/Cooking/bag2csv/Calibration/Tiago/calib_Nov/calib_mocap_2021-11-30-15-44-33/tf.csv'

    # frame_names = ['"waist_frame"', '"left_hand_frame"']
    frame_names = ['"base_frame"',
                   '"eeframe_BL"',
                   '"eeframe_BR"',
                   '"eeframe_TL"',
                   '"eeframe_TR"']
    talos_dict = extract_tfbag(path_to_csv, frame_names)

    W_pos = talos_dict[frame_names[0]]
    BL_pos = talos_dict[frame_names[1]]
    BR_pos = talos_dict[frame_names[2]]
    TL_pos = talos_dict[frame_names[3]]
    TR_pos = talos_dict[frame_names[4]]

    # update t_list take the first instant corres. base_frame data as zero
    t_list = [x + W_pos[0, 0] for x in t_pick]

    def get_data_sample(pos, t_list, eps=0.01):
        """ Extracts data samples give a list of specific instants
        """
        pos_idx = []
        for t in t_list:
            t_min = min(list(pos[0, :]), key=lambda x: abs(x-t))
            if abs(t-t_min) < eps:
                curr_idx = list(pos[0, :]).index(t_min)
                pos_idx.append(curr_idx)
            else:
                print("Missing data at %f" % t)
                break

        pos_sample = np.empty((len(pos_idx), pos.shape[0]))
        for i in range(len(pos_idx)):
            pos_sample[i, :] = pos[:, pos_idx[i]]
        return pos_sample

    W_sample = get_data_sample(W_pos, t_list)
    BL_sample = get_data_sample(BL_pos, t_list)
    BR_sample = get_data_sample(BR_pos, t_list)
    TL_sample = get_data_sample(TL_pos, t_list)
    TR_sample = get_data_sample(TR_pos, t_list)

#     # project prj_frame onto ref_frame
    def project_frame(prj_frame, ref_frame):
        projected_pos = np.empty((prj_frame.shape[0], 3))
        if prj_frame.shape != ref_frame.shape:
            print("projecting two frames have different sizes.")
        else:
            for i in range(prj_frame.shape[0]):
                ref_se3 = pin.XYZQUATToSE3(ref_frame[i, 1:])
                prj_se3 = pin.XYZQUATToSE3(prj_frame[i, 1:])
                projected_se3 = pin.SE3.inverse(ref_se3)*prj_se3
                projected_pos[i, :] = projected_se3.translation
        return projected_pos

    BL_prj_sample = project_frame(BL_sample, W_sample)
    BR_prj_sample = project_frame(BR_sample, W_sample)
    TL_prj_sample = project_frame(TL_sample, W_sample)
    TR_prj_sample = project_frame(TR_sample, W_sample)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter3D(BL_sample[:, 1], BL_sample[:, 2],
                  BL_sample[:, 3], color='blue')
    ax2.scatter3D(BR_sample[:, 1], BR_sample[:, 2],
                  BR_sample[:, 3], color='red')
    ax2.scatter3D(TL_sample[:, 1], TL_sample[:, 2],
                  TL_sample[:, 3], color='green')
    ax2.scatter3D(TR_sample[:, 1], TR_sample[:, 2],
                  TR_sample[:, 3], color='yellow')
    # ax2.scatter3D(W_sample[:, 1], W_sample[:, 2],
    #               W_sample[:, 3], color='red')
    plt.show()

    # extract joint configurations data
    # path_to_values = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_1_air_2022-02-04-13-57-29/introspection_datavalues.csv'
    # path_to_names = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_1_air_2022-02-04-13-57-29/introspection_datanames.csv'
    path_to_values = '/home/thanhndv212/Cooking/bag2csv/Calibration/Tiago/calib_Nov/calib_mocap_2021-11-30-15-44-33/introspection_datavalues.csv'
    path_to_names = '/home/thanhndv212/Cooking/bag2csv/Calibration/Tiago/calib_Nov/calib_mocap_2021-11-30-15-44-33/introspection_datanames.csv'

    # joint names
    # # tiago
    torso = '- torso_lift_joint_position'
    arm_1 = '- arm_1_joint_position'
    arm_2 = '- arm_2_joint_position'
    arm_3 = '- arm_3_joint_position'
    arm_4 = '- arm_4_joint_position'
    arm_5 = '- arm_5_joint_position'
    arm_6 = '- arm_6_joint_position'
    arm_7 = '- arm_7_joint_position'
    joint_names = [torso, arm_1, arm_2, arm_3, arm_4, arm_5, arm_6, arm_7]
    # # # talos left arm
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

    actJoint_val = extract_joint_pos(
        path_to_values, path_to_names, joint_names, t_list)
    print("expectedd NbSamplexNbjoints: ", actJoint_val.shape)

   # write to csv
    # path_save_ep = join(
    #     dirname(dirname(str(abspath(__file__)))),
    #     f"talos/talos_feb_arm_02_07_crane.csv")
    path_save_ep = join(
        dirname(dirname(str(abspath(__file__)))),
        f"tiago/tiago_nov_30_64.csv")
    headers = [
        "x1", "y1", "z1",
        "x2", "y2", "z2",
        "x3", "y3", "z3",
        "x4", "y4", "z4",
        "torso", "arm1", "arm2", "arm3", "arm4", "arm5", "arm6", "arm7"]
    with open(path_save_ep, "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(headers)
        for i in range(len(t_list)):
            row = list(np.concatenate((BL_prj_sample[i, :], BR_prj_sample[i, :],
                                      TR_prj_sample[i, :], TR_prj_sample[i, :],
                                      actJoint_val[i, :])))
            w.writerow(row)


if __name__ == '__main__':
    main()
