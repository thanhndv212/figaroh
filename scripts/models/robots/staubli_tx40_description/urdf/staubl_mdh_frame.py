import numpy as np
from numpy import linalg as la
import math as m
pi = np.pi
n = 6

def rot_R(x):
	R_x = np.array([[1, 0, 0],
					[0, m.cos(x[0]),-m.sin(x[0])],
					[0, m.sin(x[0]), m.cos(x[0])]])

	R_y = np.array([[m.cos(x[1]), 0, m.sin(x[1])],
					[0,1,0],
					[-m.sin(x[1]),0,m.cos(x[1])]])
	R_z = np.array([[m.cos(x[2]),-m.sin(x[2]),0],
					[m.sin(x[2]),m.cos(x[2]),0],
					[0,0,1]])
	R = np.matmul(R_z,np.matmul(R_y,R_x))
	# print("R",R)
	return R  

def trans_T(x,t):
	R = rot_R(x)
	assert isRot_R(R), "This is not a rotation matrix!"
	T = np.zeros([4,4])
	T[0:3,0:3] = R
	T[0:3,3] = t
	T[3,3] = 1
	return T

def T_to_Rt(T):
	R = T[0:3,0:3]
	t = T[3,0:3]
	return R, t

def isRot_R(R):
	isRot_R = False
	check_R = np.matmul(R.T,R)
	# print("check_R",check_R)
	e = la.norm(check_R - np.identity(3, dtype = R.dtype))
	if e < 1e-6:
		isRot_R = True
	else: 
		isRot_R = False
	return isRot_R

def R_to_rpy(R):
	assert isRot_R(R), "This is not a rotation matrix!"
	cos_p = m.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])

	singular = cos_p < 1e-6

	if not singular:
		y = m.atan2(R[1,0],R[0,0])
		s_r = m.sin(y)
		c_r = m.cos(y)
		p = m.atan2(-R[2,0], c_r*R[0,0] + s_r*R[1,0])
		r = m.atan2(s_r*R[0,2] - c_r*R[1,2], c_r*R[1,1]-s_r*R[0,1])
	else:
		y = 0 
		p = m.atan2(-R[2,0],R[0,0])
		r = m.atan2(-R[1,2],R[1,1])

	return np.array([r,p,y])



staubli_joints_xyz = np.array([[0,0,0.188],
							[0,0.089,0.132],
							[0.225,0,0.037],
							[0.037,0,-0.091],
							[0,0,0.188],
							[0,0,0]])
staubli_joints_rpy = np.array([[0,0,0],
							[-pi/2,-pi/2,0],
							[0,0,0],
							[pi,-pi/2,0],
							[pi,-pi/2,0],
							[0,-pi/2,pi]])

mdh_joints_xyz = np.array([[0,0,0.320],
							[0,0,0],
							[0.225,0,0.035],
							[0,-0.225,0],
							[0,0,0],
							[0,0,0]])

mdh_joints_rpy = np.array([[0,0,0],
							[-pi/2,0,0],
							[0,0,0],
							[pi/2,0,0],
							[-pi/2,0,0],
							[pi/2,pi,0]])


com_rel_xyz = np.array([[0,0.020,0.089],
						[0.109,-0.002,0.045],
						[-0.006,0.008,-0.050],
						[0.002,0.004,0.081],
						[0.003,0.001,-0.001],
						[0,0,0.042]])

com_rel_rpy = np.array([[0,0,0],
						[0,0,0],
						[0,0,0],
						[0,0,0],
						[0,0,0],
						[0,0,0]])

visual_staubli_xyz = np.array([[0,0,-0.188],
						[-0.320,0,-0.089],
						[-0.545,0,-0.126],
						[-0.035,0,-0.582],
						[-0.770,0,-0.035],
						[-0.035,0,-0.770]])
visual_staubli_rpy = np.array([[0,0,0],
						[pi/2,0,pi/2],
						[pi/2,0,pi/2],
						[0,0,-pi/2],
						[pi/2,0,pi/2],
						[0,0,-pi/2]])



#calculate homo transformation matrix staubli 
T_relative_staubli = np.zeros((6,4,4))
for i in range(n):
	T_relative_staubli[i,:,:] = trans_T(staubli_joints_rpy[i,:],staubli_joints_xyz[i,:])
T_abs_staubli = np.zeros((6,4,4))
for i in range(n):
	if i == 0:
		T_abs_staubli[0,:,:] = T_relative_staubli[0,:,:]
	else:
		temp = T_abs_staubli[0,:,:]
		for j in range(1,i+1):
			temp = np.matmul(temp,T_relative_staubli[j,:,:])
		T_abs_staubli[i,:,:] = temp

#calculate homo transformation matrix mdh
T_relative_mdh = np.zeros((6,4,4))
for i in range(n):
	T_relative_mdh[i,:,:] = trans_T(mdh_joints_rpy[i,:],mdh_joints_xyz[i,:])
T_abs_mdh = np.zeros((6,4,4))
for i in range(n):
	if i == 0:
		T_abs_mdh[0,:,:] = T_relative_mdh[0,:,:]
	else:
		temp = T_abs_mdh[0,:,:]
		for j in range(1,i+1):
			temp = np.matmul(temp,T_relative_mdh[j,:,:])
		T_abs_mdh[i,:,:] = temp


# #calculate absolute coordinates of CoM
com_mdh_xyz = np.zeros([6,3])
com_mdh_rpy = np.zeros([6,3])
mdh_to_staubli = np.zeros((6,4,4))
com_mdh_T = np.zeros((6,4,4))
for i in range(n):
	mdh_to_staubli[i,:,:] = np.matmul(np.linalg.inv(T_abs_mdh[i,:,:]),T_abs_staubli[i,:,:])
	com_mdh_T[i,:,:] = np.matmul(mdh_to_staubli[i,:,:],trans_T(com_rel_rpy[i,:],com_rel_xyz[i,:]))
com_mdh_xyz = np.around(com_mdh_T[:,0:3,3],6)
com_mdh_R = com_mdh_T[:,0:3,0:3]
for i in range(n):
	com_mdh_rpy[i,:] = np.around(R_to_rpy(com_mdh_R[i,:,:]),6)
print("com mdh xyz", com_mdh_xyz)
print("com mdh rpy", com_mdh_rpy)

#calculate  coordinates of visual frames wrt staubli

visual_staubli_xyz = np.zeros([6,3])
visual_staubli_rpy = np.zeros([6,3])
visual_T_staubli = np.zeros((6,4,4))
for i in range(n):
	visual_T_staubli[i,:,:] = np.linalg.inv(T_abs_staubli[i,:,:])
# mdh_to_staubli = np.zeros((6,4,4))
# for i in range(n):
# 	mdh_to_staubli[i,:,:] = np.matmul(np.linalg.inv(T_abs_mdh[i,:,:]),T_abs_staubli[i,:,:])
# 	visual_T_mdh[i,:,:] = np.matmul(mdh_to_staubli[i,:,:],trans_T(visual_staubli_rpy[i,:],visual_staubli_xyz[i,:]))
visual_staubli_xyz = np.around(visual_T_staubli[:,0:3,3],6)
visual_R_staubli = visual_T_staubli[:,0:3,0:3]
for i in range(n):
	visual_staubli_rpy[i,:] = np.around(R_to_rpy(visual_R_staubli[i,:,:]),6)
print("visual staubli rpy: ",visual_staubli_rpy)
print("visual staubli xyz: ", visual_staubli_xyz)



#calculate coordinates of visual frames wrt mdh 

visual_mdh_xyz = np.zeros([6,3])
visual_mdh_rpy = np.zeros([6,3])
visual_T_mdh = np.zeros((6,4,4))
for i in range(n):
	visual_T_mdh[i,:,:] = np.linalg.inv(T_abs_mdh[i,:,:])
# mdh_to_staubli = np.zeros((6,4,4))
# for i in range(n):
# 	mdh_to_staubli[i,:,:] = np.matmul(np.linalg.inv(T_abs_mdh[i,:,:]),T_abs_staubli[i,:,:])
# 	visual_T_mdh[i,:,:] = np.matmul(mdh_to_staubli[i,:,:],trans_T(visual_staubli_rpy[i,:],visual_staubli_xyz[i,:]))
visual_mdh_xyz = np.around(visual_T_mdh[:,0:3,3],6)
visual_R_mdh = visual_T_mdh[:,0:3,0:3]
for i in range(n):
	visual_mdh_rpy[i,:] = np.around(R_to_rpy(visual_R_mdh[i,:,:]),6)
print("visual rpy: ",visual_mdh_rpy)
print("visual xyz: ", visual_mdh_xyz)


# #calculate absolute coordinates of CoM
# com_abs_xyz = np.zeros([6,3])
# com_abs_rpy = np.zeros([6,3])

# #calculate relative coor of CoM wrt mdh_frames
# com_mdh_xyz = np.zeros([6,3])
# com_mdh_rpy = np.zeros([6,3])

# for i in range(n):
# 	temp_xyz = mdh_joints_xyz[0,:]
# 	temp_rpy = mdh_joints_rpy[0,:]
# 	if i == 0:
# 		pass
# 	else:
# 		for j in range(1,i+1):
# 			temp_xyz += mdh_joints_xyz[j,:]
# 			temp_rpy += mdh_joints_rpy[j,:]
# 	com_mdh_xyz[i,:] = -temp_xyz + com_abs_xyz[i,:]
# 	com_mdh_rpy[i,:] = -temp_rpy + com_abs_rpy[i,:]


# #calculate absolute coordinates of visual frames
# visual_abs_xyz = np.zeros([6,3])
# visual_abs_rpy = np.zeros([6,3])
# for i in range(n):
# 	temp_xyz = staubli_joints_xyz[0,:]
# 	temp_rpy = staubli_joints_rpy[0,:]
# 	if i == 0:
# 		pass
# 	else:
# 		for j in range(1,i+1):
# 			temp_xyz += staubli_joints_xyz[j,:]
# 			temp_rpy += staubli_joints_rpy[j,:]
# 	visual_abs_xyz[i,:] = temp_xyz + visual_staubli_xyz[i,:]
# 	visual_abs_rpy[i,:] = temp_rpy + visual_staubli_rpy[i,:]
# #calculate relative coor of visual frames wrt mdh_frames
# visual_mdh_xyz = np.zeros([6,3])
# visual_mdh_rpy = np.zeros([6,3])

# for i in range(n):
# 	temp_xyz = mdh_joints_xyz[0,:]
# 	temp_rpy = mdh_joints_rpy[0,:]
# 	if i == 0:
# 		pass
# 	else:
# 		for j in range(1,i+1):
# 			temp_xyz += mdh_joints_xyz[j,:]
# 			temp_rpy += mdh_joints_rpy[j,:]
# 	visual_mdh_xyz[i,:] = -temp_xyz + visual_abs_xyz[i,:]
# 	visual_mdh_rpy[i,:] = -temp_rpy + visual_abs_rpy[i,:]
# print("Center of Mass in absolute frame (xyz):\n", com_abs_xyz)
# print("Center of Mass in absolute frame (rpy):\n", com_abs_rpy)
# print("Center of Mass in mdh_frames frame (xyz):\n", com_mdh_xyz)
# print("Center of Mass in mdh_frames frame (rpy):\n", com_mdh_rpy)
# print("Visual frame in mdh_frames frame (xyz):\n", visual_mdh_xyz)
# print("Visual of Mass in mdh_frames frame (rpy):\n", visual_mdh_rpy)
