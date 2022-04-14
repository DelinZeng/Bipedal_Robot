import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig, norm
from scipy.linalg import svd
from math import cos, sin, acos, atan2
import matplotlib.pyplot as plt

from lx16a import *
import time


###############################################################################################################################
# IK Solver
###############################################################################################################################

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def IK_RRR_Newton(L1, L2, L3, q_i, Td, lamda, DLS):
    I = np.identity(3)

    # intial condition of all joints
    qk = q_i

    od_x = Td[0][3]
    od_y = Td[1][3]
    od_z = Td[2][3]
    xd = np.array([Td[0][0], Td[1][0], Td[2][0]])
    yd = np.array([Td[0][1], Td[1][1], Td[2][1]])
    zd = np.array([Td[0][2], Td[1][2], Td[2][2]])

    ee_x = []
    ee_y = []
    ee_z = []

    qk_history = []
    # qk_history.append(qk)

    error_norm_history = []
    error_norm = 100000
    previous_error_norm = 0
    iterations = []
    i = 0

    while abs(previous_error_norm - error_norm) > 0.000001:

        previous_error_norm = error_norm
        iterations.append(i)
        qk_history.append(qk)

        theta1 = qk[0]
        theta2 = qk[1]
        theta3 = qk[2]

        # Compute J(qk)
        J = np.array([
            [-L1 * sin(theta1) - L2 * sin(theta1 + theta2) - L3 * sin(theta1 + theta2 + theta3),
             -L2 * sin(theta1 + theta2) - L3 * sin(theta1 + theta2 + theta3), -L3 * sin(theta1 + theta2 + theta3)],
            [L1 * cos(theta1) + L2 * cos(theta1 + theta2) + L3 * cos(theta1 + theta2 + theta3),
             L2 * cos(theta1 + theta2) + L3 * cos(theta1 + theta2 + theta3), L3 * cos(theta1 + theta2 + theta3)],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]])

        # Compute pesudo-inverse of J(qk)
        # becasue m = 6, n = 3  >> overconstrainted >> left Pseudoinverse
        J_P_L = inv(J.transpose() @ J) @ J.transpose()
        # print(J_P_L)

        # Compute the DLS pesudo-inverse
        J_DLS = inv(J.transpose() @ J + lamda * I) @ J.transpose()

        # Compute end effector pose f(qk) using FK
        f_q = np.array([
            [cos(theta1 + theta2 + theta3), -sin(theta1 + theta2 + theta3), 0,
             L1 * cos(theta1) + L2 * cos(theta1 + theta2) + L3 * cos(theta1 + theta2 + theta3)],
            [sin(theta1 + theta2 + theta3), cos(theta1 + theta2 + theta3), 0,
             L1 * sin(theta1) + L2 * sin(theta1 + theta2) + L3 * sin(theta1 + theta2 + theta3)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # compute error e(qk)
        ok_x = f_q[0][3]
        ok_y = f_q[1][3]
        ok_z = f_q[2][3]
        xk = np.array([f_q[0][0], f_q[1][0], f_q[2][0]])
        yk = np.array([f_q[0][1], f_q[1][1], f_q[2][1]])
        zk = np.array([f_q[0][2], f_q[1][2], f_q[2][2]])

        ee_x.append(ok_x)
        ee_y.append(ok_y)
        ee_z.append(ok_z)

        delta_theta = (1 / 2) * (np.cross(xk, xd) + np.cross(yk, yd) + np.cross(zk, zd))
        # print(delta_theta)
        delta_o = np.array([od_x - ok_x, od_y - ok_y, od_z - ok_z])
        # print(delta_o)

        error = np.array([
            [delta_o[0]],
            [delta_o[1]],
            [delta_o[2]],
            [delta_theta[0]],
            [delta_theta[1]],
            [delta_theta[2]],
        ])
        # print('error', error)
        error_norm = norm(error)
        error_norm_history.append(error_norm)

        if DLS:
            qk = qk + J_DLS @ error
        else:
            qk = qk + J_P_L @ error
        # print('qk',qk)
        # qk_history.append(qk)
        i += 1

    return qk, ee_x, ee_y, qk_history


def IK_RRR_Gradient_Descent(L1,L2,L3,q_i,Td,alpha):
    #intial condition of all joints
    qk = q_i

    od_x = Td[0][3]
    od_y = Td[1][3]
    od_z = Td[2][3]
    xd = np.array([Td[0][0],Td[1][0],Td[2][0]])
    yd = np.array([Td[0][1],Td[1][1],Td[2][1]])
    zd = np.array([Td[0][2],Td[1][2],Td[2][2]])

    ee_x = []
    ee_y = []
    ee_z = []

    qk_history = []
    # qk_history.append(qk)

    error_norm_history = []
    error_norm = 100000
    previous_error_norm = 0

    iterations = []
    i = 0

    while abs(previous_error_norm - error_norm)  > 0.0001:

      previous_error_norm = error_norm
      iterations.append(i)
      qk_history.append(qk)

      theta1 = qk[0]
      theta2 = qk[1]
      theta3 = qk[2]

      #Compute J(qk)
      J = np.array([
        [-L1*sin(theta1) - L2*sin(theta1 + theta2) - L3*sin(theta1 + theta2 + theta3), -L2*sin(theta1 + theta2) - L3*sin(theta1 + theta2 + theta3), -L3*sin(theta1 + theta2 + theta3)],
        [L1*cos(theta1) + L2*cos(theta1 + theta2) + L3*cos(theta1 + theta2 + theta3), L2*cos(theta1 + theta2) + L3*cos(theta1 + theta2 + theta3), L3*cos(theta1 + theta2 + theta3)],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])


      # #Compute pesudo-inverse of J(qk)
      # #becasue m = 6, n = 3  >> overconstrainted >> left Pseudoinverse
      # J_P_L = inv(J.transpose() @ J ) @ J.transpose()
      # # print(J_P_L)

      #Compute end effector pose f(qk) using FK
      f_q = np.array([
        [cos(theta1 + theta2 + theta3), -sin(theta1 + theta2 + theta3), 0, L1*cos(theta1) + L2*cos(theta1 + theta2) + L3*cos(theta1 + theta2 + theta3)],
        [sin(theta1 + theta2 + theta3), cos(theta1 + theta2 + theta3), 0, L1*sin(theta1) + L2*sin(theta1 + theta2) + L3*sin(theta1 + theta2 + theta3)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

      #compute error e(qk)
      ok_x = f_q[0][3]
      ok_y = f_q[1][3]
      ok_z = f_q[2][3]
      xk = np.array([f_q[0][0],f_q[1][0],f_q[2][0]])
      yk = np.array([f_q[0][1],f_q[1][1],f_q[2][1]])
      zk = np.array([f_q[0][2],f_q[1][2],f_q[2][2]])

      ee_x.append(ok_x)
      ee_y.append(ok_y)
      ee_z.append(ok_z)

      delta_theta = (1/2) * (np.cross(xk,xd)+np.cross(yk,yd)+np.cross(zk,zd))
      # print(delta_theta)
      delta_o = np.array([od_x-ok_x, od_y-ok_y, od_z-ok_z ])
      # print(delta_o)

      error = np.array([
                      [delta_o[0]],
                      [delta_o[1]],
                      [delta_o[2]],
                      [delta_theta[0]],
                      [delta_theta[1]],
                      [delta_theta[2]],
      ])
      # print('error', error)
      error_norm = norm(error)
      error_norm_history.append(error_norm)

      qk = qk + alpha * J.transpose() @ error
      i += 1

    return qk, ee_x, ee_y, qk_history


def Bipedal_leg_IK(desired_x, desired_y, Newton, Gradient_Descent, first_elbow_up):
    '''
    Input: Deisred x, Desired y of the End-Effector
    Output: joints configuration for desired Deisred x, Desired y
    '''

    ################################################################################
    # Geomtry Params
    ################################################################################

    # initial Configuration
    q_i = np.array([[-3.14 / 2], [0], [0]])

    # Desired End effector Config
    desired_z = 0

    #VERTICAL TO GROUDN
    Td = np.array([
        [6.123233995736766e-17, 1, 0, desired_x],
        [-1, 6.123233995736766e-17, 0, desired_y],
        [0, 0, 1, desired_z],
        [0, 0, 0, 1]
    ])

    # #LEARNIN FORWARD
    # Td = np.array([
    #     [0, 1, 0, desired_x],
    #     [-1, 0.2, 0, desired_y],
    #     [0, 0, 1, desired_z],
    #     [0, 0, 0, 1]
    # ])
 


    # Define the length for each linkage
    L1 = 6
    L2 = 6
    L3 = 1
    L4 = 2
    L7 = 1.5
    L6 = 2
    L5 = 6.83

    ################################################################################
    # Control Params
    ################################################################################

    # Newton's Method Params
    lamda = 5
    DLS = True

    # Gradient_Descent Params
    alpha = 0.01  # learning rate

    ################################################################################
    # Numerically solve for the first RRR Linkage
    ################################################################################

    # NEWTOWN's METHOD#
    if Newton:
        qk, ee_x, ee_y, qk_history = IK_RRR_Newton(L1, L2, L3, q_i, Td, lamda, DLS)

    # Gradeint Descent
    if Gradient_Descent:
        qk, ee_x, ee_y, qk_history = IK_RRR_Gradient_Descent(L1, L2, L3, q_i, Td, alpha)

    #############################################################
    # manually flip the joint configuration if it is not physical
    if first_elbow_up:

        # convert all the angles to positive range
        if qk[0] < 0:
            qk[0] = qk[0] + 2 * np.pi
        if qk[1] < 0:
            qk[1] = qk[1] + 2 * np.pi
        # if qk[2] < 0:
        #     qk[2] = qk[2] + 2 * np.pi

        # print('Before Correction')n
        # print('Theta0 = ', qk[0]/np.pi*180)
        # print('Theta1 = ', qk[1]/np.pi*180)
        # print('Theta2 = ', qk[2]/np.pi*180)

        # Flip the joint angle if not in the desired range of motion: qk[1] < (3/4)*np.pi
        if qk[1] < (3 / 4) * np.pi:
            # Forward Kinematics to find the second joint position
            joint_1_x = L1 * cos(qk[0])
            joint_1_y = L1 * sin(qk[0])

            # second joint position from T02
            joint_2_x = L1 * cos(qk[0]) + L2 * cos(qk[0] + qk[1])
            joint_2_y = L1 * sin(qk[0]) + L2 * sin(qk[0] + qk[1])

            delta_angle_joint_0 = angle_between([joint_1_x, joint_1_y], [joint_2_x, joint_2_y])
            # print('delta_angle_joint_0',delta_angle_joint_0/np.pi*180)

            qk[0] = qk[0] + 2 * delta_angle_joint_0
            qk[1] = -qk[1]

            qk[2] = delta_angle_joint_0 + delta_angle_joint_0 - np.abs(qk[2])



        # print('After correction')
        # print('Theta0 = ', qk[0]/np.pi*180)
        # print('Theta1 = ', qk[1]/np.pi*180)
        # print('Theta2 = ', qk[2]/np.pi*180)

        # convert all the angles to positive range for output
        if qk[0] < 0:
            qk[0] = qk[0] + 2 * np.pi
        if qk[1] < 0:
            qk[1] = qk[1] + 2 * np.pi
        if qk[2] < 0:
            qk[2] = qk[2] + 2 * np.pi

        # print('Theta0 = ', qk[0]/np.pi*180)
        # print('Theta1 = ', qk[1]/np.pi*180)
        # print('Theta2 = ', qk[2]/np.pi*180)

    #############################################################
    # Forward Kinematics to find the joint position

    # calculate final first joint position from T01
    f_qk_x_1 = L1 * cos(qk[0])
    f_qk_y_1 = L1 * sin(qk[0])

    # calculate final second joint position from T02
    f_qk_x_2 = L1 * cos(qk[0]) + L2 * cos(qk[0] + qk[1])
    f_qk_y_2 = L1 * sin(qk[0]) + L2 * sin(qk[0] + qk[1])

    #Calculate final EE joint position from T03
    f_qk_x_ee = L1*cos(qk[0]) + L2*cos(qk[0] + qk[1]) + L3*cos(qk[0] + qk[1] + qk[2])
    f_qk_y_ee = L1*sin(qk[0]) + L2*sin(qk[0] + qk[1]) + L3*sin(qk[0] + qk[1] + qk[2])


    ####################################################################################
    # Gemotry Approach for Second RR Linkage
    ####################################################################################
    # Now with the joint 2 position we can find the
    # joint 4 position (End-Effector position for second
    # RR linkage)

    #############################################################
    # Find the joint 4 position wrt joint 6
    theta1 = qk[0]
    theta2 = qk[1]

    X_4_0 = L1 * cos(theta1) - L4 * sin(theta1) * sin(theta2) + L4 * cos(theta1) * cos(theta2)
    Y_4_0 = L1 * sin(theta1) + L4 * sin(theta1) * cos(theta2) + L4 * sin(theta2) * cos(theta1)
    # print(X_4_0,Y_4_0)

    X_4_6 = X_4_0 + L7
    Y_4_6 = Y_4_0
    # print(X_4_6,Y_4_6)

    #############################################################
    # check if joint 4 position is within the reachable range of
    # second RR linkage. If not, this inverse kinematics solution
    # either invalid or not physcially possible

    D = np.sqrt(X_4_6 ** 2 + Y_4_6 ** 2)
    if D < (L5 - L6) or D > (L5 + L6):
        raise ValueError("Error: Out of Second RR Linkage Range")


    #############################################################
    # Using Gemotry Approach, solve the IK of a RR arm
    theta5_1 = acos((X_4_6 ** 2 + Y_4_6 ** 2 - L6 ** 2 - L5 ** 2) / (2 * L5 * L6))

    # calculate corresponding theta6_1 using theta5_1
    sin_theta1 = (Y_4_6 * (L6 + L5 * cos(theta5_1)) - X_4_6 * L5 * sin(theta5_1)) / (X_4_6 ** 2 + Y_4_6 ** 2)
    cos_theta1 = ((X_4_6 * (L6 + L5 * cos(theta5_1)) + Y_4_6 * L5 * sin(theta5_1)) / (X_4_6 ** 2 + Y_4_6 ** 2))
    theta6_1 = atan2(sin_theta1, cos_theta1)

    theta5 = theta5_1
    theta6 = theta6_1

    # convert both theta5_1 and theta6_1 to postive range
    if theta5 < 0:
        theta5 = theta5 + 2 * np.pi
    if theta6 < 0:
        theta6 = theta6 + 2 * np.pi

    # print('Theta5 = ', theta5/np.pi*180)
    # print('Theta6 = ', theta6/np.pi*180)

    #############################################################
    # calculate joint 5 position using theta6
    X_5_6 = L6 * cos(theta6)
    Y_5_6 = L6 * sin(theta6)

    # joint 5 position wrt joint 0
    X_5_0 = X_5_6 - L7
    Y_5_0 = Y_5_6
    # print(X_5_0,Y_5_0)

    # joint 6 position wrt joint 0
    X_6_0 = -L7
    Y_6_0 = 0

    ####################################################################################
    # Double check the results !!!!
    ####################################################################################
    # recalculate each linkage length
    L1_fk = np.sqrt((0 - f_qk_x_1) ** 2 + (0 - f_qk_y_1) ** 2)
    L2_fk = np.sqrt((f_qk_x_1 - f_qk_x_2) ** 2 + (f_qk_y_1 - f_qk_y_2) ** 2)
    L3_fk = np.sqrt((f_qk_x_ee-f_qk_x_2)**2 + (f_qk_y_ee-f_qk_y_2)**2)
    L4_fk = np.sqrt((f_qk_x_1 - X_4_0) ** 2 + (f_qk_y_1 - Y_4_0) ** 2)
    L6_fk = np.sqrt((X_6_0 - X_5_0) ** 2 + (Y_6_0 - Y_5_0) ** 2)
    L5_fk = np.sqrt((X_5_0 - X_4_0) ** 2 + (Y_5_0 - Y_4_0) ** 2)

    np.testing.assert_allclose([L1_fk, L2_fk, L3_fk, L4_fk, L6_fk, L5_fk], [L1, L2, L3, L4, L6, L5],
                               err_msg='Linkage Length Check Failed')



    ####################################################################################
    # Outputs
    ####################################################################################

    # Joint Angles [rad]
    # 0 - 1 - 2 - 5 - 6
    Joints_config = [qk[0][0], qk[1][0], qk[2][0], theta5, theta6]
    # print(joints_config)

    # Joints Position [x,y]
    # 0 - 1 - 2 - ee - 6 - 5 - 4
    Joints_position_x = [0, f_qk_x_1, f_qk_x_2, f_qk_x_ee, X_6_0, X_5_0, X_4_0]
    Joints_position_y = [0, f_qk_y_1, f_qk_y_2, f_qk_y_ee, Y_6_0, Y_5_0, Y_4_0]
    # print(Joints_position_x,Joints_position_y)

    return Joints_config, Joints_position_x, Joints_position_y



###############################################################################################################################
# Path Planning
###############################################################################################################################

def first_path_planning(start, end, alpha, lamda):
    ############################################################################
    # before generating the moving path, we need to
    # make the start and end position is reachable

    # start
    try:
        Joints_config, Joints_position_x, Joints_position_y = Bipedal_leg_IK(start[0], start[1], True, False, True)
    except ValueError:
        print('Path Plannig failed at start configuration: x={}, y={}'.format(start[0], start[1]))
    else:
        print('Valid Start Position at x={}, y={}'.format(start[0], start[1]))

    # end
    try:
        Joints_config, Joints_position_x, Joints_position_y = Bipedal_leg_IK(end[0], end[1], True, False, True)
    except ValueError:
        print('Path Plannig failed at end configuration: x={}, y={}'.format(end[0], end[1]))
    else:
        print('Valid End Position at x={}, y={}'.format(end[0], end[1]))

    ############################################################################
    # Initial (Calibration) Configuration
    q_i = np.array([[-np.pi/2], [0], [0]])

    L1 = 6
    L2 = 6
    L3 = 1

    desired_z = 0

    Td_1 = np.array([
        [6.123233995736766e-17, 1, 0, start[0]],
        [-1, 6.123233995736766e-17, 0, start[1]],
        [0, 0, 1, desired_z],
        [0, 0, 0, 1]
    ])

    Td_2 = np.array([
        [6.123233995736766e-17, 1, 0, end[0]],
        [-1, 6.123233995736766e-17, 0, end[1]],
        [0, 0, 1, desired_z],
        [0, 0, 0, 1]
    ])

    qk_stage1, ee_x_stage1, ee_y_stage1, qk_history_stage1 = IK_RRR_Newton(L1, L2, L3, q_i, Td_1, lamda, True)
    # print(qk_history_stage1)

    q_i_2 = np.array([qk_history_stage1[-1][0], qk_history_stage1[-1][1], qk_history_stage1[-1][2]])
    # print(q_i_2)

    qk_stage2, ee_x_stage2, ee_y_stage2, qk_history_stage2 = IK_RRR_Gradient_Descent(L1, L2, L3, q_i_2, Td_2, alpha)

    q_end = np.array([qk_history_stage2[-1][0], qk_history_stage2[-1][1], qk_history_stage2[-1][2]])
    # print(q_i_2)

    # plt.scatter(ee_x_stage1, ee_y_stage1, c='r')
    # plt.scatter(ee_x_stage2, ee_y_stage2, c='g')

    path_x = ee_x_stage1 + ee_x_stage2
    path_y = ee_y_stage1 + ee_y_stage2
    print('# of steps planned: {}'.format(len(path_x)))

    # plt.figure()
    # plt.plot(path_x, path_y, c='b')
    #
    # plt.xlim(-15, 15)
    # plt.ylim(-15, 5)
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.grid()

    return path_x, path_y, q_end,


def path_planning(q_i, end, alpha):
    ############################################################################
    # before generating the moving path, we need to
    # make the start and end position is reachable

    # end
    try:
        Joints_config, Joints_position_x, Joints_position_y = Bipedal_leg_IK(end[0], end[1], True, False, True)
    except ValueError:
        print('Path Plannig failed at end configuration: x={}, y={}'.format(end[0], end[1]))
    else:
        print('Valid End Position at x={}, y={}'.format(end[0], end[1]))

    ############################################################################
    # # Start Configuration
    # q_i = np.array([[-3.14 / 2], [0], [0]])

    L1 = 6
    L2 = 6
    L3 = 1

    desired_z = 0

    Td_2 = np.array([
        [0, 1, 0, end[0]],
        [-1, 0, 0, end[1]],
        [0, 0, 1, desired_z],
        [0, 0, 0, 1]
    ])

    qk_stage2, ee_x_stage2, ee_y_stage2, qk_history_stage2 = IK_RRR_Gradient_Descent(L1, L2, L3, q_i, Td_2, alpha)

    q_end = np.array([qk_history_stage2[-1][0], qk_history_stage2[-1][1], qk_history_stage2[-1][2]])

    # plt.scatter(ee_x_stage2, ee_y_stage2, c='g')

    path_x = ee_x_stage2
    path_y = ee_y_stage2
    print('# of steps planned: {}'.format(len(path_x)))

    # plt.figure()
    # plt.plot(path_x, path_y, c='b')
    #
    # plt.xlim(-15, 15)
    # plt.ylim(-15, 5)
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # plt.grid()

    return path_x, path_y, q_end,\


def sample_points_on_planned_path(path_x, path_y, delta_displacement):
    # delta_displacement = 1

    sampled_x = []
    sampled_y = []

    sampled_x.append(path_x[0])
    sampled_y.append(path_y[0])

    start = [path_x[0], path_y[0]]
    for i in np.arange(0, len(path_x)):
        if np.sqrt((start[0] - path_x[i]) ** 2 + (start[1] - path_y[i]) ** 2) < delta_displacement:
            pass
        else:
            sampled_x.append(path_x[i])
            sampled_y.append(path_y[i])

            start = [path_x[i], path_y[i]]

    sampled_x.append(path_x[-1])
    sampled_y.append(path_y[-1])

    print('# of actual steps moved: {}'.format(len(sampled_x)))

    return sampled_x, sampled_y


###############################################################################################################################
# Simulation & Animation
###############################################################################################################################


def animation_bipedal_legs(Joint_position_x_history_L, Joint_position_y_history_L, desired_x_history_L, desired_y_history_L, Joint_position_x_history_R, Joint_position_y_history_R, desired_x_history_R, desired_y_history_R, moving_time):
    '''
    Input: all joints position history
      Joint_position_x_history [N,5]
      Joint_position_y_history [N,5]

    Output: animation of leg movement

    '''

    # figure = plt.figure()
    ax = plt.axes(projection='3d')

   
    for i in np.arange(len(Joint_position_x_history_L)):

        ##########LEFT LEG##########
        # desired moving path
        ax.scatter(desired_x_history_L, desired_y_history_L, 0, c='g')

        joint0_x = Joint_position_x_history_L[i][0]
        joint0_y = Joint_position_y_history_L[i][0]

        joint1_x = Joint_position_x_history_L[i][1]
        joint1_y = Joint_position_y_history_L[i][1]

        joint2_x = Joint_position_x_history_L[i][2]
        joint2_y = Joint_position_y_history_L[i][2]

        EE_x = Joint_position_x_history_L[i][3]
        EE_y = Joint_position_y_history_L[i][3]

        joint6_x = Joint_position_x_history_L[i][4]
        joint6_y = Joint_position_y_history_L[i][4]

        joint5_x = Joint_position_x_history_L[i][5]
        joint5_y = Joint_position_y_history_L[i][5]

        joint4_x = Joint_position_x_history_L[i][6]
        joint4_y = Joint_position_y_history_L[i][6]

        # First Linkage
        ax.scatter([joint0_x, joint1_x, joint2_x, EE_x], [joint0_y, joint1_y, joint2_y, EE_y], [0, 0, 0, 0], c='r')
        ax.plot3D([joint0_x, joint1_x, joint2_x, EE_x], [joint0_y, joint1_y, joint2_y, EE_y], [0, 0, 0, 0], c='k')

        # Second Linkage
        ax.scatter([joint6_x, joint5_x, joint4_x], [joint6_y, joint5_y, joint4_y], [0, 0, 0], c='r')
        ax.plot3D([joint6_x, joint5_x, joint4_x], [joint6_y, joint5_y, joint4_y], [0, 0, 0], c='k')


        ##########RIGHT LEG##########
        # desired moving path
        ax.scatter(desired_x_history_R, desired_y_history_R, 5, c='g')

        joint0_x = Joint_position_x_history_R[i][0]
        joint0_y = Joint_position_y_history_R[i][0]

        joint1_x = Joint_position_x_history_R[i][1]
        joint1_y = Joint_position_y_history_R[i][1]

        joint2_x = Joint_position_x_history_R[i][2]
        joint2_y = Joint_position_y_history_R[i][2]

        EE_x = Joint_position_x_history_R[i][3]
        EE_y = Joint_position_y_history_R[i][3]

        joint6_x = Joint_position_x_history_R[i][4]
        joint6_y = Joint_position_y_history_R[i][4]

        joint5_x = Joint_position_x_history_R[i][5]
        joint5_y = Joint_position_y_history_R[i][5]

        joint4_x = Joint_position_x_history_R[i][6]
        joint4_y = Joint_position_y_history_R[i][6]

        # First Linkage
        ax.scatter([joint0_x, joint1_x, joint2_x, EE_x], [joint0_y, joint1_y, joint2_y, EE_y], [5, 5, 5, 5], c='r')
        ax.plot3D([joint0_x, joint1_x, joint2_x, EE_x], [joint0_y, joint1_y, joint2_y, EE_y], [5, 5, 5, 5], c='k')

        # Second Linkage
        ax.scatter([joint6_x, joint5_x, joint4_x], [joint6_y, joint5_y, joint4_y], [5, 5, 5], c='r')
        ax.plot3D([joint6_x, joint5_x, joint4_x], [joint6_y, joint5_y, joint4_y], [5, 5, 5], c='k')

        plt.ylim(-15, 5)
        plt.xlim( -5, 5)
        plt.pause(moving_time)
        ax = plt.axes(projection='3d')

    return




def simulating_leg_movement(x_L,y_L,x_R,y_R,moving_time):

    #LEFT
    Joint_position_x_history_L = []
    Joint_position_y_history_L = []
    Joint_config_history_L = []

    for i in np.arange(len(x_L)):
      try:
          Joints_config_L, Joints_position_x_L, Joints_position_y_L = Bipedal_leg_IK(x_L[i],y_L[i],True,False,True)
      except ValueError:
          print('!!! IK failed at x={}, y={} !!!'.format(x_L[i],y_L[i]))
      else:
          print('Left Leg: IK solution founded at x={}, y={}'.format(x_L[i],y_L[i]))
          Joint_position_x_history_L.append(Joints_position_x_L)
          Joint_position_y_history_L.append(Joints_position_y_L)
          Joint_config_history_L.append(Joints_config_L)


    #RIGHT
    Joint_position_x_history_R = []
    Joint_position_y_history_R = []
    Joint_config_history_R = []

    for i in np.arange(len(x_R)):
      try:
          Joints_config_R, Joints_position_x_R, Joints_position_y_R = Bipedal_leg_IK(x_R[i],y_R[i],True,False,True)
      except ValueError:
          print('!!! IK failed at x={}, y={} !!!'.format(x_R[i],y_R[i]))
      else:
          print('Right Leg: IK solution founded at x={}, y={}'.format(x_R[i],y_R[i]))
          Joint_position_x_history_R.append(Joints_position_x_R)
          Joint_position_y_history_R.append(Joints_position_y_R)
          Joint_config_history_R.append(Joints_config_R)


    animation_bipedal_legs(Joint_position_x_history_L, Joint_position_y_history_L, x_L, y_L, Joint_position_x_history_R, Joint_position_y_history_R, x_R, y_R, moving_time)


    return Joint_config_history_L, Joint_config_history_R


###############################################################################################################################
# Motor Execuation
###############################################################################################################################
def convert_IK_sols_2_motor_execution_RIGHT_LEG(Joint_config_history):
    # this function outputs the change in degree for each motor on the
    #right leg

    thigh_motor = []
    shin_motor = []
    ankle_motor = []

    for i in np.arange(0, len(Joint_config_history)):
        
            thigh_motor.append(Joint_config_history[i][0] *180/np.pi - 270) 
        
            shin_motor.append(Joint_config_history[i][4] *180/np.pi - 237)

            if Joint_config_history[i][2] > np.pi: 
                Joint_config_history[i][2] = Joint_config_history[i][2] - 2*np.pi

            ankle_motor.append(-Joint_config_history[i][2] *180/np.pi) 

    return thigh_motor, shin_motor, ankle_motor


def convert_IK_sols_2_motor_execution_LEFT_LEG(Joint_config_history):
    # this function outputs the change in degree for each motor on the
    #right leg

    thigh_motor = []
    shin_motor = []
    ankle_motor = []

    for i in np.arange(0, len(Joint_config_history)):
        
            thigh_motor.append(-(Joint_config_history[i][0] *180/np.pi - 270))
        
            shin_motor.append(-(Joint_config_history[i][4] *180/np.pi - 237))

            if Joint_config_history[i][2] > np.pi: 
                Joint_config_history[i][2] = Joint_config_history[i][2] - 2*np.pi

            ankle_motor.append(-Joint_config_history[i][2] *180/np.pi) 

    return thigh_motor, shin_motor, ankle_motor




def move_the_robot_sinwave(thigh_motor_L,shin_motor_L,ankle_motor_L,thigh_motor_R,shin_motor_R,ankle_motor_R, intial_calibration):
    


    if intial_calibration: 

        ###############################
        #####DEFAULT POSITIONS!!!!#####
        ###############################

        #Right Leg
        servo1.move(68, time=550)
        servo2.move(105.0, time=550)
        servo3.move(63, time=550)
        print(servo1.get_physical_angle())
        print(servo2.get_physical_angle())
        print(servo3.get_physical_angle())
        print('RIGHT MOTOR INITIALIZED')

        time.sleep(3)

        #Left Leg
        #change servo4 57 to 60 bc motor swap
        servo4.move(60, time=550)
        servo5.move(150, time=550)
        servo6.move(70, time=550)


        print(servo4.get_physical_angle())
        print(servo5.get_physical_angle())
        print(servo6.get_physical_angle())
        print('LEFT MOTOR INITIALIZED')

        time.sleep(1)



    servo1_init_pos = 68
    servo2_init_pos = 105
    servo3_init_pos = 63

    servo4_init_pos = 60
    servo5_init_pos = 150
    servo6_init_pos = 70

    for i in np.arange(len(thigh_motor_L)):

        #Right Leg
        servo1_angle_goal = 68+thigh_motor_R[i]
        servo2_angle_goal = 105+shin_motor_R[i]
        servo3_angle_goal = 63+ankle_motor_R[i]
        print('servo1_angle_goal',servo1_angle_goal)
        print('servo2_angle_goal',servo2_angle_goal)
        print('servo3_angle_goal',servo3_angle_goal)

        #Left Leg
        servo4_angle_goal = 60+thigh_motor_L[i]
        servo5_angle_goal = 150+shin_motor_L[i]
        servo6_angle_goal = 70+ankle_motor_L[i]
        print('servo4_angle_goal',servo4_angle_goal)
        print('servo5_angle_goal',servo5_angle_goal)
        print('servo6_angle_goal',servo6_angle_goal)
        print('\n')

        if servo1_angle_goal < 0: 
            servo1_angle_goal = 0
        if servo2_angle_goal < 0: 
            servo2_angle_goal = 0
        if servo3_angle_goal < 0: 
            servo3_angle_goal = 0

        if servo4_angle_goal < 0: 
            servo4_angle_goal = 0
        if servo5_angle_goal < 0: 
            servo5_angle_goal = 0
        if servo6_angle_goal < 0: 
            servo6_angle_goal = 0

        diff1 = servo1_angle_goal-servo1_init_pos
        diff2 = servo2_angle_goal-servo2_init_pos
        diff3 = servo3_angle_goal-servo3_init_pos
        diff4 = servo4_angle_goal-servo4_init_pos
        diff5 = servo5_angle_goal-servo5_init_pos
        diff6 = servo6_angle_goal-servo6_init_pos

        dt = 0
        while dt < np.pi/2:

            servo1_angle_next = sin(dt) * diff1 + servo1_init_pos
            servo2_angle_next = sin(dt) * diff2 + servo2_init_pos
            servo3_angle_next = sin(dt) * diff3 + servo3_init_pos

            servo4_angle_next = sin(dt) * diff4 + servo4_init_pos
            servo5_angle_next = sin(dt) * diff5 + servo5_init_pos
            servo6_angle_next = sin(dt) * diff6 + servo6_init_pos

            servo1.move(servo1_angle_next) #right thigh
            servo2.move(servo2_angle_next) #right shin
            servo3.move(servo3_angle_next) #right ankle

            servo4.move(servo4_angle_next) #right thigh
            servo5.move(servo5_angle_next) #right shin
            servo6.move(servo6_angle_next) #right ankle
        

            time.sleep(0.005)

            # # OIGINAL 
            # dt += 0.2

            #MOVING FASTER
            dt += 0.4
       

        servo1_init_pos = servo1_angle_goal
        servo2_init_pos = servo2_angle_goal
        servo3_init_pos = servo3_angle_goal

        servo4_init_pos = servo4_angle_goal
        servo5_init_pos = servo5_angle_goal
        servo6_init_pos = servo6_angle_goal

        time.sleep(0.045) 



        # servo1_init_pos = servo1.get_physical_angle()
        # servo2_init_pos = servo2.get_physical_angle()
        # servo3_init_pos = servo3.get_physical_angle()

        # servo4_init_pos = servo4.get_physical_angle()
        # servo5_init_pos = servo5.get_physical_angle()
        # servo6_init_pos = servo6.get_physical_angle()


    time.sleep(2)

    print('End of Movement:')
    # print('servo 1 physical',servo1.get_physical_angle())
    # print('servo 2 physical',servo2.get_physical_angle())
    # print('servo 3 physical',servo3.get_physical_angle())
    # print('servo 4 physical',servo4.get_physical_angle())
    # print('servo 5 physical',servo5.get_physical_angle())
    # print('servo 6 physical',servo6.get_physical_angle())
    print('\n')

    return 



###############################################################################################################################
### Walking Planning ###
###############################################################################################################################
 

#####################
#Plan Some Paths here! 
#####################
#Squat
# N = 3
# y = np.linspace(-12, -6, N)

# sampled_y = np.concatenate([y,np.flip(y)])
# sampled_y = np.concatenate([sampled_y,sampled_y])
# sampled_x = -1* np.ones(len(sampled_y))

# #Circle
# N = 80
# path_x = np.linspace(-3,3,N)
# path_y = -np.sqrt(10**2 - path_x**2)
# sampled_x, sampled_y = sample_points_on_planned_path(path_x,path_y,delta_displacement)


def move_to_starting_stand_position():

    delta_displacement = 0.7
    moving_time = 0.005 

    #####################
    #Plan the Path here! 
    #####################
    #stand position
    path_x_0, path_y_0, q_end = first_path_planning([0,-13],[-1,-10],alpha,lamda)
    path_x_1, path_y_1, q_end = path_planning(q_end, [-1,-8], alpha)
    path_x_2, path_y_2, q_end = path_planning(q_end, [-2,-11.5], alpha)
    path_x_2, path_y_2, q_end = path_planning(q_end, [-2,-8], alpha)
    path_x_2, path_y_2, q_end = path_planning(q_end, [-2,-10.5], alpha)
    path_x_2, path_y_2, q_end = path_planning(q_end, [-2,-11.5], alpha)


    x_end = -2
    y_end = -11.5


    path_x = path_x_0 + path_x_1 + path_x_2
    path_y = path_y_0 + path_y_1 + path_y_2
    # path_x = path_x_0 + path_x_1 + path_x_2 + path_x_3 + path_x_4 + path_x_5 +  path_x_6
    # path_y = path_y_0 + path_y_1 + path_y_2 + path_y_3 + path_y_4 + path_y_5 +  path_y_6
    sampled_x, sampled_y = sample_points_on_planned_path(path_x,path_y,delta_displacement)

    #####################
    #Let's simulate first
    #####################

    Joint_config_history_L, Joint_config_history_R = simulating_leg_movement(sampled_x, sampled_y, sampled_x, sampled_y, moving_time)
    thigh_motor_L, shin_motor_L, ankle_motor_L = convert_IK_sols_2_motor_execution_LEFT_LEG(Joint_config_history_L)
    thigh_motor_R, shin_motor_R, ankle_motor_R = convert_IK_sols_2_motor_execution_RIGHT_LEG(Joint_config_history_R)
    
    print('\n')
    print('Delta displacement:', delta_displacement)
    print('Number of steps moved: ',len(sampled_y))
    print('End of Simulation')
    print('\n')


    val = input("Move the robot to starting stand position?: [Y/N]")
    if val == 'Y':
        print('Motors Execuation Start')
        move_the_robot_sinwave(thigh_motor_L,shin_motor_L,ankle_motor_L,thigh_motor_R,shin_motor_R,ankle_motor_R,True)
        # move_the_robot_angle(thigh_motor_R,shin_motor_R,ankle_motor_R,thigh_motor_L,shin_motor_L,ankle_motor_L,True)
        print('move to stand position ends')

    else: 
        print('Motors EXecuation Aborted, fail to move to starting stand position')

    return q_end


def get_ready_to_move(q_end):

    delta_displacement = 0.7
    moving_time = 0.0005

    #####################
    #baby step moving path 
    #####################


    # X_forward = [-2,    -1.785,  -1.560,  -1.232, -0.650, -0.012, -0.184]
    # Y_forward = [-11.5, -10.789, -10.200, -9.636, -9.498, -9.550, -10.559]

    # X_backward = [-0.305, -0.553,  -0.768,  -0.993,  -1.272,  -1.564,  -1.818]
    # Y_backward = [-11.5, -11.641, -11.657, -11.638, -11.616, -11.574, -11.526 ]


    #propell more 
    X_forward = [-3]
    Y_forward = [-11.5]

    X_backward = [-0.305]
    Y_backward = [-11.5]

    X_L = X_forward 
    Y_L = Y_forward 

    X_R = X_backward
    Y_R = Y_backward



    Joint_config_history_L, Joint_config_history_R = simulating_leg_movement(X_L, Y_L, X_R, Y_R, moving_time)
    thigh_motor_L, shin_motor_L, ankle_motor_L = convert_IK_sols_2_motor_execution_LEFT_LEG(Joint_config_history_L)
    thigh_motor_R, shin_motor_R, ankle_motor_R = convert_IK_sols_2_motor_execution_RIGHT_LEG(Joint_config_history_R)
        
    
    # print('\n')
    # print('Delta displacement:', delta_displacement)
    # print('Number of steps moved: ',len(sampled_y))
    # print('End of Simulation')
    # print('\n')


    #####################
    #Time to move it 
    #####################

    val = input("Move the robot?: [Y/N]")
    if val == 'Y':
        print('Motors Execuation Start')

        move_the_robot_sinwave(thigh_motor_L,shin_motor_L,ankle_motor_L,thigh_motor_R,shin_motor_R,ankle_motor_R,False)

        # move_the_robot_angle(thigh_motor_R,shin_motor_R,ankle_motor_R,thigh_motor_L,shin_motor_L,ankle_motor_L,False)
        print('get ready to move ends')

    else: 
        print('Motors Execuation Aborted, fail to move to starting stand position')

    
    return q_end 


def move_baby_step_forward(q_end):

    delta_displacement = 0.7
    moving_time = 0.00001

    #####################
    #baby step moving path 
    #####################


    # #Orginal
    # X_forward = [-2,    -1.785,  -1.560,  -1.232, -0.650, -0.012, -0.184]
    # Y_forward = [-11.5, -10.789, -10.200, -9.636, -9.498, -9.550, -10.559]

    # X_backward = [-0.305, -0.553,  -0.768,  -0.993,  -1.272,  -1.564,  -1.818]
    # Y_backward = [-11.5, -11.641, -11.657, -11.638, -11.616, -11.574, -11.526 ]


    #propell more (1 inch back)
    X_forward = [-3,    -2.701,  -2.371,  -1.695, -0.827, -0.005, -0.184]
    Y_forward = [-11.5, -10.639, -9.907, -9.611, -9.521, -9.563, -10.559]

    X_backward = [-0.305, -0.765,  -1.227,  -1.665,  -2.146,  -2.485,  -2.781]
    Y_backward = [-11.5, -11.648, -11.658, -11.658, -11.654, -11.634, -11.580 ]


    # #propell more (1 inch back) + lift less (1 inch lower)
    # X_forward = [-3,    -2.701,  -2.371,  -1.695, -0.827, -0.005, -0.184]
    # Y_forward = [-11.5, -10.639-0.5, -9.907-0.5, -9.611-0.5, -9.521-0.5, -9.563-0.5, -10.559-0.5]

    # X_backward = [-0.305, -0.765,  -1.227,  -1.665,  -2.146,  -2.485,  -2.781]
    # Y_backward = [-11.5, -11.648, -11.658, -11.658, -11.654, -11.634, -11.580 ]



    X_L = X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + [X_forward[0]]
    Y_L = Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + [Y_forward[0]]

    X_R = X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + X_backward + X_forward + [X_backward[0]]
    Y_R = Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + Y_backward + Y_forward + [Y_backward[0]]



    Joint_config_history_L, Joint_config_history_R = simulating_leg_movement(X_L, Y_L, X_R, Y_R, moving_time)
    thigh_motor_L, shin_motor_L, ankle_motor_L = convert_IK_sols_2_motor_execution_LEFT_LEG(Joint_config_history_L)
    thigh_motor_R, shin_motor_R, ankle_motor_R = convert_IK_sols_2_motor_execution_RIGHT_LEG(Joint_config_history_R)
        
    
    # print('\n')
    # print('Delta displacement:', delta_displacement)
    # print('Number of steps moved: ',len(sampled_y))
    # print('End of Simulation')
    # print('\n')


    #####################
    #Time to move it 
    #####################

    val = input("Move the robot?: [Y/N]")
    if val == 'Y':
        print('Motors Execuation Start')

        move_the_robot_sinwave(thigh_motor_L,shin_motor_L,ankle_motor_L,thigh_motor_R,shin_motor_R,ankle_motor_R,False)

        # move_the_robot_angle(thigh_motor_R,shin_motor_R,ankle_motor_R,thigh_motor_L,shin_motor_L,ankle_motor_L,False)
        print('baby step ends')

    else: 
        print('Motors Execuation Aborted, fail to move to starting stand position')


    return q_end




###############################################################################################################################
########################################################  Control Panel ################################################ 
###############################################################################################################################


# ########################################
# ## INITIALIZE MOTOR INSTANCES
########################################
# LX16A.initialize("/dev/tty.usbserial-110")

# try:
#     #RIGHT LEG
#     servo1 = LX16A(1) #right thigh
#     servo2 = LX16A(2) #right shin
#     servo3 = LX16A(3) #right ankle

#     #LEFT LEG
#     servo4 = LX16A(4) #left thigh
#     servo5 = LX16A(5) #left shin
#     servo6 = LX16A(6) #left ankle

# #Exception handling in the event one of the motors is disconnected or not connected to begin with
# except ServoTimeoutError as e:
#     print(f"Servo {e.id_} is not connected. Exiting...")
#     quit()

# start = t.time()


# wait = True
# #Pause for 1 seconds before beginning sequence
# while wait:
#     if t.time()-start > 1:
#         wait = False

# print('ready')


########################################
## Control Params
########################################

moving_time = 0.005 # Simulation speed
alpha = 0.001 #Gradient Descent (Don't change this)
lamda = 100 #Newton's method
# delta_displacement = 1



########################################
## Game Time !!! 
########################################

q_end = move_to_starting_stand_position()

# q_end = get_ready_to_move(q_end)

move_baby_step_forward(q_end)







