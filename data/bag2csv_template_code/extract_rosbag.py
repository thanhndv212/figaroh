import csv
from re import split
import pandas as pd
import numpy as np
import rospy


# for msg in read_values:
#     first_row = msg.replace('[', '')
#     first_row = first_row.replace(']', '')
#     split_data = first_row.split(',')
#     if len(split_data) - 4 == len(d):  # ignore first 4 columns
#         for i in range(4, len(split_data)):
#             # split_data[i] = float(split_data[i])
#             d[names[i-4]].append(float(split_data[i]))


tiago_valuedf = pd.read_csv(
    '/home/thanhndv212/Cooking/bag2csv/calibration/introspection_datavalues.csv')
tiago_namedf = pd.read_csv(
    '/home/thanhndv212/Cooking/bag2csv/calibration/introspection_datanames.csv')

d = {}

# time
read_secs = tiago_valuedf.loc[:, 'secs'].values
read_nsecs = tiago_valuedf.loc[:, 'nsecs'].values

t = []
starting_time = rospy.rostime.Time(read_secs[0], read_nsecs[0]).to_sec()
for i in range(len(read_secs)):
    t.append(rospy.rostime.Time(
        read_secs[i], read_nsecs[i]).to_sec()-starting_time)
d.update({'t': t})
# joint names
first_name = '- arm_1_joint_position'
last_name = '- /gravity_compensation_local_control_actual_effort_torso_lift_joint'

first_name_idx = tiago_namedf.columns.get_loc(first_name)
last_name_idx = tiago_namedf.columns.get_loc(last_name)

names = tiago_namedf.columns[range(first_name_idx, last_name_idx+1)]
for name in names:
    d.update({name: []})

# joint values
read_values = tiago_valuedf.loc[:, 'values'].values

for msg in read_values:
    first_row = msg.replace('[', '')
    first_row = first_row.replace(']', '')
    split_data = first_row.split(',')
    if len(split_data) - 4 == len(names):  # ignore first 4 columns
        for i in range(4, len(split_data)):
            d[names[i-4]].append(float(split_data[i]))

df_tiago = pd.DataFrame(data=d)
