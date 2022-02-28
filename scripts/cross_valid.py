import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize, signal
import pandas as pd
import pinocchio as pin

from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num


from tiago_mocap_calib_fun_def import (
    extract_expData,
    extract_expData4Mkr,
    get_param,
    init_var,
    get_PEE_fullvar,
    get_PEE_var,
    get_geoOffset,
    get_jointOffset,
    get_PEE,
    Calculate_kinematics_model,
    Calculate_identifiable_kinematics_model,
    Calculate_base_kinematics_regressor)

robot = Robot(
    "tiago_description/robots",
    "tiago_no_hand_mod.urdf",
    # isFext=True  # add free-flyer joint at base
)

model = robot.model
data = robot.data

NbSample = 24
param = get_param(robot, NbSample, TOOL_NAME='ee_marker_joint', NbMarkers=1)
# path_to_file = "/home/thanhndv212/Cooking/figaroh/data/tiago/cross_valid.csv"
path_to_file = "/home/dvtnguyen/calibration/figaroh/data/tiago/cross_valid.csv"

xyz_4Mkr = pd.read_csv(
    path_to_file, usecols=list(range(0, param['NbMarkers']*3))).to_numpy()

# next 8 cols: joints position
q_act = pd.read_csv(path_to_file, usecols=list(
    range(3, 11))).to_numpy()
param['NbSample'] = q_act.shape[0]
# extract measured end effector coordinates

# extract measured joint configs
q_exp = np.empty((param['NbSample'], param['q0'].shape[0]))
for i in range(param['NbSample']):
    q_exp[i, 0:8] = q_act[i, :]
    # ATTENTION: need to check robot.q0 vs TIAGo.q0
    q_exp[i, 8:] = param['q0'][8:]


def cal_error(q_zero, xyz_zero):
    res_zero = []
    lastJoint_frameId = model.getFrameId("ee_marker_joint")
    xyz_zeroM = np.empty_like(xyz_zero)

    for i in range(8):
        pin.framesForwardKinematics(model, data, q_zero[i, :])
        pin.updateFramePlacements(model, data)
        ee_frame = data.oMf[lastJoint_frameId]
        xyz_zeroM[i, :] = ee_frame.translation
        res = np.linalg.norm(xyz_zero[i, :] - xyz_zeroM[i, :])
        res_zero.append(res)
        print(res)
    res_zero = np.array(res_zero)
    avg = np.mean(res_zero)
    res_zero = res_zero - avg
    # plt.figure()
    # plt.bar(np.arange(8), res_zero)
    return res_zero


# zero offset
xyz_zero = xyz_4Mkr[0:8, :]
q_zero = q_exp[0:8, :]


# pal offset
offset_pal = [0.02, -0.000704814, -0.0054179,
              0.0105299, -0.02, 0.0197129, -0.02]
xyz_pal = xyz_4Mkr[8:16, :]
q_pal = q_exp[8:16, :]
for i in range(1, 8):
    q_pal[:, i] = q_pal[:, i] + offset_pal[i-1]


# mocap offset
xyz_mc = xyz_4Mkr[16:24, :]
q_mc = q_exp[16:24, :]
param['NbSample'] = 8
var_0, nvars = init_var(param, mode=0)
offset_joint = [-0.000461829875023,
                -0.000290769329428,
                -0.000213928263712,
                -0.00038150019357,
                -0.012679130784866,
                -4.01516125500777e-05,
                0.004077355537572,
                -8.31230128939489e-06,
                -0.00102127280533,
                -0.049838087409011,
                0.001276277679481,
                -0.000881553717289,
                0.0093142510224,
                -0.01367734025194,
                -0.004817119263865,
                -0.001142354185646,
                -0.000337753353601,
                -0.013705011181662,
                -0.001952018944863,
                0.005081981843458,
                -0.159131939078765,
                -0.000106509146992,
                0.003123084196717,
                0.005834542095168,
                -0.027993078602761,
                -0.028001934247682]

var_0[6:32] = np.array(offset_joint)
PEE = get_PEE_fullvar(var_0, q_mc, model, data, param,
                      noise=False, base_model=True)
xyz_mcM = np.reshape(PEE, (3, 8))
xyz_mcM = xyz_mcM.T
res_mc = []

for i in range(8):
    res = np.linalg.norm(xyz_mc[i, :] - xyz_mcM[i, :])
    res_mc.append(res)
res_mc = np.array(res_mc)
avg = np.mean(res_mc)
res_mc = res_mc - avg
# res_mc[0] = 0.005

res_zero = cal_error(q_zero, xyz_zero)
res_pal = cal_error(q_pal, xyz_pal)

ax = plt.subplot(111)
w = 0.2
x = np.arange(8)
pt_name = ['center',
           'C1',
           'M1',
           'C2',
           'M2',
           'C3',
           'M3',
           'C4',
           'M4']
bar_zero = ax.bar(x-w, res_zero, width=w, color='b', align='center')
bar_pal = ax.bar(x, res_pal, width=w, color='g', align='center')
bar_mc = ax.bar(x+w, res_mc, width=w, color='r', align='center')

ax.set_ylabel('Error (m)')
ax.set_xticklabels(pt_name)

ax.legend((bar_zero[0], bar_pal[0], bar_mc[0]),
          ('zero offset', 'pal offset', 'mocap-based offset'))

plt.grid()
plt.show()
