import numpy as np
import time

import pinocchio as pin

from sys import argv
import os
from os.path import dirname, join, abspath


from tools.robot import Robot
from tools.regressor import eliminate_non_dynaffect
from tools.qrdecomposition import get_baseParams, cond_num
from meshcat_viewer_wrapper import MeshcatVisualizer

from tiago_mocap_calib_fun_def import (
    get_param,
    get_PEE_fullvar,
    get_PEE_var,
    extract_expData4Mkr,
    get_geoOffset,
    get_jointOffset,
    get_PEE,
    Calculate_kinematics_model,
    Calculate_identifiable_kinematics_model,
    Calculate_base_kinematics_regressor,
    cartesian_to_SE3,
    CIK_problem)
from tiago_simplified import Box

robot = Robot(
    "talos_data/robots",
    "talos_reduced.urdf",
    isFext=True  # add free-flyer joint at base
)
model = robot.model
data = robot.data


q0 = robot.q0
eps = 1e-4
IT_MAX = 1000
DT = 1e-2
damp = 1e-12

# absoluate location of two feet
pin.forwardKinematics(model, data, q0)
oMdes_rightF = data.oMi[model.getJointId('leg_right_6_joint')]
oMdes_leftF = data.oMi[model.getJointId('leg_left_6_joint')]

# # target frame
tool_name = 'gripper_right_base_link'
target_frameId = model.getFrameId(tool_name)
target_frame = model.frames[target_frameId]
joint_parentId = target_frame.parent

# create targes set on contact planes

# x plane
rpy_x = np.array([0, -np.pi/2, 0])
square_center_x = np.array([0.7, 0., 0.0])
square_dim_x = np.array([0., 0.4, 0.4])

NbGrid = 8
NbPoints = pow(NbGrid, 2)
PEEd_x = np.linspace(square_center_x[0] - square_dim_x[0]/2,
                     square_center_x[0] + square_dim_x[0]/2, NbGrid)
PEEd_y = np.linspace(square_center_x[1] - square_dim_x[1]/2,
                     square_center_x[1] + square_dim_x[1]/2, NbGrid)
PEEd_z = np.linspace(square_center_x[2] - square_dim_x[2]/2,
                     square_center_x[2] + square_dim_x[2]/2, NbGrid)

PEEd_2d_x = np.empty((NbPoints, 3))
for i in range(PEEd_y.shape[0]):
    for j in range(PEEd_z.shape[0]):
        idx = NbGrid*i + j
        PEEd_2d_x[idx, :] = np.array(
            [square_center_x[0], PEEd_y[i], PEEd_z[j]])
targets_x = np.empty((NbPoints, 6))
for ii in range(NbPoints):
    targets_x[ii, 0:3] = PEEd_2d_x[ii, :]
    targets_x[ii, 3:6] = rpy_x
plane_placement_x = cartesian_to_SE3(
    np.array([0.7+0.246525, 0., 0.0, 0, -np.pi/2, 0]))  # -0.246525
visual_model = robot.visual_model
geom_model = robot.geom_model
geom_model.addGeometryObject(
    Box("planex", model.getJointId("universe"), 0.8,
        0.8, 0.002, plane_placement_x)
)
visual_model.addGeometryObject(
    Box("planex", model.getJointId("universe"), 0.8,
        0.8, 0.002, plane_placement_x)
)

# y plane
rpy_y = np.array([np.pi/2, 0, 0])
square_center_y = np.array([0.5, 0.2, 0.0])
square_dim_y = np.array([0.4, 0, 0.4])
NbGrid = 8
NbPoints = pow(NbGrid, 2)

PEEd_x = np.linspace(square_center_y[0] - square_dim_y[0]/2,
                     square_center_y[0] + square_dim_y[0]/2, NbGrid)
PEEd_y = np.linspace(square_center_y[1] - square_dim_y[1]/2,
                     square_center_y[1] + square_dim_y[1]/2, NbGrid)
PEEd_z = np.linspace(square_center_y[2] - square_dim_y[2]/2,
                     square_center_y[2] + square_dim_y[2]/2, NbGrid)

PEEd_2d_y = np.empty((NbPoints, 3))
for i in range(PEEd_x.shape[0]):
    for j in range(PEEd_z.shape[0]):
        idx = NbGrid*i + j
        PEEd_2d_y[idx, :] = np.array(
            [PEEd_x[i], square_center_y[1], PEEd_z[j]])

targets_y = np.empty((NbPoints, 6))
for ii in range(NbPoints):
    targets_y[ii, 0:3] = PEEd_2d_y[ii, :]
    targets_y[ii, 3:6] = rpy_y

plane_placement_y = cartesian_to_SE3(
    np.array([0.5, 0.2+0.246525, 0.0, np.pi/2, 0, 0]))  # -0.246525
visual_model = robot.visual_model
geom_model = robot.geom_model
geom_model.addGeometryObject(
    Box("planey", model.getJointId("universe"), 0.8,
        0.8, 0.002, plane_placement_y)
)
visual_model.addGeometryObject(
    Box("planey", model.getJointId("universe"), 0.8,
        0.8, 0.002, plane_placement_y)
)

# z plane
rpy_z = np.array([0, 0, 0])
square_center_z = np.array([0.5, 0, -0.2])
square_dim_z = np.array([0.4, 0.4, 0])
NbGrid = 8
NbPoints = pow(NbGrid, 2)

PEEd_x = np.linspace(square_center_z[0] - square_dim_z[0]/2,
                     square_center_z[0] + square_dim_z[0]/2, NbGrid)
PEEd_y = np.linspace(square_center_z[1] - square_dim_z[1]/2,
                     square_center_z[1] + square_dim_z[1]/2, NbGrid)
PEEd_z = np.linspace(square_center_z[2] - square_dim_z[2]/2,
                     square_center_z[2] + square_dim_z[2]/2, NbGrid)

PEEd_2d_z = np.empty((NbPoints, 3))
for i in range(PEEd_x.shape[0]):
    for j in range(PEEd_y.shape[0]):
        idx = NbGrid*i + j
        PEEd_2d_z[idx, :] = np.array(
            [PEEd_x[i], PEEd_y[j], square_center_z[2]])
targets_z = np.empty((NbPoints, 6))
for ii in range(NbPoints):
    targets_z[ii, 0:3] = PEEd_2d_z[ii, :]
    targets_z[ii, 3:6] = rpy_z

plane_placement_z = cartesian_to_SE3(
    np.array([0.5, 0, -0.2 - 0.246525, 0, 0, 0]))  # -0.246525
visual_model = robot.visual_model
geom_model = robot.geom_model
geom_model.addGeometryObject(
    Box("planez", model.getJointId("universe"), 0.8,
        0.8, 0.002, plane_placement_z)
)
visual_model.addGeometryObject(
    Box("planez", model.getJointId("universe"), 0.8,
        0.8, 0.002, plane_placement_z)
)

targets = np.vstack((np.vstack((targets_x, targets_y)), targets_z))
NbSample = NbPoints*3

q_sample = []
q = q0
for iter in range(NbSample):
    i = 0
    oMdes = cartesian_to_SE3(targets[iter, :])
    while True:
        pin.forwardKinematics(model, data, q)
        dMi = oMdes.actInv(data.oMi[joint_parentId])
        err = pin.log(dMi).vector

        # to fix two feet on ground
        dMi_rightF = oMdes_rightF.actInv(data.oMi[model.getJointId('leg_right_6_joint')])
        err_rightF = pin.log(dMi_rightF).vector

        dMi_leftF = oMdes_leftF.actInv(data.oMi[model.getJointId('leg_left_6_joint')])
        err_leftF = pin.log(dMi_leftF).vector
        if np.linalg.norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeJointJacobian(model, data, q, joint_parentId)
        v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model, q, v*DT)
        # if not i % 10:
        #     print('%d: error = %s' % (i, err.T))
        i += 1

    if success:
        print("Sample %s Convergence achieved!" % iter)
        q_sample.append(q)
    else:
        print("\nSample %s the iterative algorithm has not reached convergence to the desired precision" % iter)

    # print('\nresult: %s' % q.flatten().tolist())
    # print('\nfinal error: %s' % err.T)

q_sample = np.array(q_sample)

# visualize motion
# viz = MeshcatVisualizer(
#     model=robot.model, collision_model=robot.collision_model,
#     visual_model=robot.visual_model, url='classical'
# )

# time.sleep(3)
# for i in range(q_sample.shape[0]):
#     viz.display(q_sample[i, :])
#     time.sleep(0.2)


# Normal vector of contact plane
norm_vec_x = np.array([-1/square_center_x[0], 0, 0])

norm_vec_y = np.array([0, -1/square_center_y[0],  0])

norm_vec_z = np.array([0, 0, -1/square_center_z[0]])

norm_vec = np.empty((3, 3))
norm_vec[0, :] = norm_vec_x
norm_vec[1, :] = norm_vec_y
norm_vec[2, :] = norm_vec_z

# create observation matrix
R = np.empty((q_sample.shape[0],  6*(model.njoints-1)))
for jj in range(3):
    for ii in range(int(q_sample.shape[0]/3)):
        q_i = q_sample[jj*int(q_sample.shape[0]/3) + ii, :]
        pin.forwardKinematics(model, data, q_i)
        pin.updateFramePlacements(model, data)
        # Pxyz_i = data.oMf[target_frameId]
        R_i = pin.computeFrameKinematicRegressor(
            model, data, target_frameId, pin.LOCAL)
        # R[ii, 0:3] = Pxyz_i.translation
        R[jj*int(q_sample.shape[0]/3) + ii, :] = norm_vec[jj, 0]*R_i[0, :] + norm_vec[jj, 1] * \
            R_i[1, :] + norm_vec[jj, 2]*R_i[2, :]
joint_names = [name for i, name in enumerate(model.names[1:])]
geo_params = get_geoOffset(joint_names)
# P_dict = {"delta_a": 0, "delta_b": 0, "delta_c": 0}
# P_dict.update(geo_params)

R_e, params_e = eliminate_non_dynaffect(
    R, geo_params, tol_e=1e-6)
print(len(params_e), R_e.shape)

R_b, params_base = get_baseParams(R_e, params_e)
for i, pb in enumerate(params_base):
    print( pb)

# q_rand = pin.randomConfiguration(model)

# pin.forwardKinematics(model, data, robot.q0)
# pin.updateFramePlacements(model, data)
# # tranformation from base link to joint placement
# oMp = data.oMi[14]*model.jointPlacements[15]

# # transformation from base link to target frame
# print(data.oMi[target_frame.parent])
# print(data.oMf[target_frameId])

# R = pin.computeFrameKinematicRegressor(
#     model, data, target_frameId, pin.LOCAL)
# print(R[:, -48:-23])


# create transformation chain from world frame to the gripper

# data.oMf(): give transformation from base link frame -> target frame

# kinematic chain: world_frame -> feet-baselink -> baselink-wrist -> wrist-gripper
# SE3(world_frame)*inv(data.oMf(leg_left_7))*data.oMf(arm_left_7)*SE3(gripper)

# to justify the observability: derive regressor matrix of above transformation matrix
# how?


# create set of configs where gripper lies on a plane + visualization with baselink is fixed (not free floating)
