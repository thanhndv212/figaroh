import csv
from re import split
import pandas as pd
import numpy as np
import rospy
import dask.dataframe as dd

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

    # data arrange
    # return a dict contain key/item = frame_name(str)/numpy array
    tf_dict = {}
    t_val = []
    starting_t = rospy.rostime.Time(sec_val[0], nsec_val[0]).to_sec()

    for i in range(len(sec_val)):
        t_val.append(rospy.rostime.Time(
            sec_val[i], nsec_val[i]).to_sec() - starting_t)

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


def extract_joint_pos(path_to_values, path_to_names):
    """ Extracts joint angles
    """
    # read names and values from csv to dataframe
    dt_names = pd.read_csv(path_to_names)
    dt_values = pd.read_csv(path_to_values)

    # first_name = '- arm_1_joint_position'
    # last_name = '- /gravity_compensation_local_control_actual_effort_torso_lift_joint'

    # first_name_idx = dt_names.columns.get_loc(first_name)
    # last_name_idx = dt_names.columns.get_loc(last_name)

    # names = dt_names.columns[range(first_name_idx, last_name_idx+1)]
    # for name in names:
    #     d.update({name: []})

    dt_names_cols = list(dt_names.columns)
    print(len(dt_names_cols))
    dt_values_val = dt_values.loc[:, 'values'].values
    print(type(dt_values_val[1]), dt_values_val.shape)

    msg = dt_values_val[1]
    first_row = msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')
    print(type(split_data))
    print(len(split_data))


def main():
    # path_to_csv = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_2_contact_gripper_2022-02-04-14-42-37/tf.csv'
    # frame_names = ['"waist_frame"', '"left_hand_frame"']

    # talos_dict = extract_tfbag(path_to_csv, frame_names)
    # LH_pos = talos_dict[frame_names[0]]
    # W_pos = talos_dict[frame_names[1]]

    # fig = plt.figure()
    # ax = fig.subplots(4, 1)
    # ax[0].plot(W_pos[0, :], W_pos[3, :])

    # ax[1].plot(LH_pos[0, :], LH_pos[1, :])
    # # ax[0].plot(W_pos[0, :])

    # ax[2].plot(LH_pos[0, :], LH_pos[2, :])
    # # ax[1].plot(W_pos[1, :])

    # ax[3].plot(LH_pos[0, :], LH_pos[3, :])
    # # ax[2].plot(W_pos[2, :])

    # plt.show()
    path_to_values = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_2_contact_gripper_2022-02-04-14-42-37/introspection_datavalues.csv'

    path_to_names = '/home/dvtnguyen/calibration/raw_data/talos_feb/torso_arm_2_contact_gripper_2022-02-04-14-42-37/introspection_datanames.csv'

    extract_joint_pos(path_to_values, path_to_names)


if __name__ == '__main__':
    main()

    # values = pd.read_csv(
    #     '/home/thanhndv212/Cooking/bag2csv/calibration/introspection_datavalues.csv')
    # names = pd.read_csv(
    #     '/home/thanhndv212/Cooking/bag2csv/calibration/introspection_datanames.csv')

    # d = {}

    # # time
    # read_secs = values.loc[:, 'secs'].values
    # read_nsecs = values.loc[:, 'nsecs'].values

    # t = []
    # starting_time = rospy.rostime.Time(read_secs[0], read_nsecs[0]).to_sec()
    # for i in range(len(read_secs)):
    #     t.append(rospy.rostime.Time(
    #         read_secs[i], read_nsecs[i]).to_sec()-starting_time)
    # d.update({'t': t})
    # # joint names
    # first_name = '- arm_1_joint_position'
    # last_name = '- /gravity_compensation_local_control_actual_effort_torso_lift_joint'

    # first_name_idx = names.columns.get_loc(first_name)
    # last_name_idx = names.columns.get_loc(last_name)

    # names = names.columns[range(first_name_idx, last_name_idx+1)]
    # for name in names:
    #     d.update({name: []})

    # # joint values
    # read_values = values.loc[:, 'values'].values

    # for msg in read_values:
    #     first_row = msg.replace('[', '')
    #     first_row = first_row.replace(']', '')
    #     split_data = first_row.split(',')
    #     if len(split_data) - 4 == len(names):  # ignore first 4 columns
    #         for i in range(4, len(split_data)):
    #             d[names[i-4]].append(float(split_data[i]))

    # df_tiago = pd.DataFrame(data=d)
