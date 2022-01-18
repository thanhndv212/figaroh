import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # center point position
# center_p = np.array([0.7, 0.0, 0.7])

# # normal vector
# vec = np.array([1, 0, 0])
# normal_vec = vec/np.linalg.norm(vec)

# # size
# w = 0.5
# h = 0.5

# # one edge unit vector
# up_vec = np.array([0, 0, 1])

# # edge unit vector
# U = np.cross(normal_vec, up_vec)
# W = np.cross(normal_vec, U)

# # corner points
# C1 = center_p + w/2*U + h/2*W
# C2 = center_p - w/2*U + h/2*W
# C3 = center_p - w/2*U - h/2*W
# C4 = center_p + w/2*U - h/2*W

# # middle points
# M1 = center_p + w/2*U
# M2 = center_p + h/2*W
# M3 = center_p - w/2*U
# M4 = center_p - h/2*W

# #################
# # C4     M4    C3
# #
# # M1     P     M3
# #
# # C1     M2    C2
# #################
# corner_points = np.array([C1, C2, C3, C4])
# mid_points = np.array([M1, M2, M3, M4])


# for i in range(mid_points.shape[0]):
#     ax.scatter(mid_points[i, 0], mid_points[i, 1], mid_points[i, 2])
# for i in range(corner_points.shape[0]):
#     ax.scatter(corner_points[i, 0], corner_points[i, 1], corner_points[i, 2])
#     ax.plot([corner_points[i, 0], corner_points[i-1, 0]], [corner_points[i, 1],
#             corner_points[i-1, 1]], [corner_points[i, 2], corner_points[i-1, 2]])
# ax.scatter(center_p[0], center_p[1], center_p[2])
# ax.plot([0, center_p[0]], [0, center_p[1]], [0, center_p[2]])

# # origin and axes show
# x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# u, v, w = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
# ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")
# ax.grid()
# plt.show()


# data set of 4 corners and middle points
''' algo to extract 10 points near corner point and middle point
    store list of 2D arrays'''
dp_corner = []
dp_mid = []

# square fitting optimization problem


def create_square(var, width, height, edge):
    ''' var: an array containing position of center point normalized nomal vector
        w, h: size of the rectangle
        edge: a vector of upside edge
    '''
    # center point position
    center_p = np.copy(var)[0:3]

    # normal vector
    vec = np.copy(var)[3:6]
    normal_vec = vec/np.linalg.norm(vec)

    # size
    w = width
    h = height

    # one edge unit vector
    up_vec = np.copy(edge)

    # edge unit vector
    U = np.cross(normal_vec, up_vec)
    W = np.cross(normal_vec, U)

    # corner points
    C1 = center_p + w/2*U + h/2*W
    C2 = center_p - w/2*U + h/2*W
    C3 = center_p - w/2*U - h/2*W
    C4 = center_p + w/2*U - h/2*W

    # middle points
    M1 = center_p + w/2*U
    M2 = center_p + h/2*W
    M3 = center_p - w/2*U
    M4 = center_p - h/2*W

    corner_points = np.array([C1, C2, C3, C4])
    mid_points = np.array([M1, M2, M3, M4])
    return corner_points, mid_points


def plot_square(ax, corner_points, mid_points):
    for i in range(mid_points.shape[0]):
        ax.scatter(mid_points[i, 0], mid_points[i, 1], mid_points[i, 2])

    for i in range(corner_points.shape[0]):
        ax.scatter(corner_points[i, 0],
                   corner_points[i, 1], corner_points[i, 2])
        ax.plot([corner_points[i, 0], corner_points[i-1, 0]], [corner_points[i, 1],
                corner_points[i-1, 1]], [corner_points[i, 2], corner_points[i-1, 2]])

    # origin and axes show
    x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    u, v, w = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black")
    ax.grid()
    # plt.show()


def cost_function(var, w, h, edge, dp_corner, dp_mid):
    c_pts, m_pts = create_square(var, w, h, edge)
    res_vec = []
    for i in range(4):
        for j in range(dp_corner[i].shape[0]):
            res_vec.append(np.linalg.norm([c_pts[i, :] - dp_corner[i][j, :]]))
        for k in range(dp_mid[i].shape[0]):
            res_vec.append(np.linalg.norm([m_pts[i, :] - dp_mid[i][k, :]]))

    res_sum = np.linalg.norm(np.array(res_vec))
    if res_sum == 0:
        print(" Cost function cannot be calculated!!!")
    return res_sum


def main():
    # predefined constants
    edge = np.array([0, 0, 1])
    w = 0.5
    h = 0.5

    # optimization problem
    init_guess = np.array([0.7, 0.0, 0.7, 1, 0, 0])

    # fitting data
    c_pts_sample, m_pts_sample = create_square(init_guess, w, h, edge)

    # artificial test data
    dp_corner = []
    dp_mid = []
    Nb_pts = 10
    for i in range(4):
        k_corner = np.zeros([Nb_pts, 3])
        k_mid = np.zeros([Nb_pts, 3])
        for k in range(Nb_pts):
            k_corner[k, :] = c_pts_sample[i, :] + 0.005 * \
                np.random.rand(c_pts_sample[i, :].shape[0])
            k_mid[k, :] = m_pts_sample[i, :] + 0.005 * \
                np.random.rand(m_pts_sample[i, :].shape[0])
        dp_corner.append(k_corner)
        dp_mid.append(k_mid)
    print(len(dp_corner))

    rslt = optimize.least_squares(cost_function, init_guess, jac='3-point',
                                  args=(w, h, edge, dp_corner, dp_mid), verbose=1)

    print(rslt.x)
    print(rslt.cost)
    minimum = optimize.fmin(cost_function, init_guess,
                            args=(w, h, edge, dp_corner, dp_mid))
    print(minimum)

    # plot solution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    LS_cpts, LS_mpts = create_square(rslt.x, w, h, edge)
    fmin_cpts, fmin_mpts = create_square(minimum, w, h, edge)

    plot_square(ax, LS_cpts, LS_mpts)
    plot_square(ax, fmin_cpts, fmin_mpts)

    # plot give data points
    for i in range(4):
        for j in range(dp_corner[i].shape[0]):
            ax.scatter(dp_corner[i][j, 0], dp_corner[i]
                       [j, 1], dp_corner[i][j, 2])
        for k in range(dp_mid[i].shape[0]):
            ax.scatter(dp_mid[i][k, 0], dp_mid[i][k, 1], dp_mid[i][k, 2])
    plot_square(ax, c_pts_sample, m_pts_sample)

    plt.show()


if __name__ == '__main__':
    main()
