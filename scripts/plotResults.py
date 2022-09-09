from array import array
from operator import gt
import sys, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math 
from scipy.interpolate import interp1d
import quaternion as quat
from pyquaternion import Quaternion

def computeEuclideanDistance(x1, y1, z1, x2, y2, z2):
    list_dist = []
    for xp1,yp1,zp1,xp2,yp2,zp2 in zip(x1,y1,z1,x2,y2,z2):
        if np.isneginf(xp2) or np.isneginf(yp2) or np.isneginf(zp2) :
            continue
        else:
            list_dist.append(math.sqrt((xp2-xp1)*(xp2-xp1)+(yp2-yp1)*(yp2-yp1)+(zp2-zp1)*(zp2-zp1)))
    array_dist = np.array(list_dist)
    return array_dist

def computeQuaternionError(qx1, qy1, qz1, qw1, qx2, qy2, qz2, qw2):

    list_q_error = []
    for qxp1,qyp1,qzp1,qwp1,qxp2,qyp2,qzp2,qwp2 in zip(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2):
        normalizer1 = math.sqrt(qwp1*qwp1+qxp1*qxp1+qyp1*qyp1+qzp1*qzp1)
        qxp1 = qxp1/normalizer1
        qyp1 = qyp1/normalizer1
        qzp1 = qzp1/normalizer1
        qwp1 = qwp1/normalizer1

        normalizer2 = math.sqrt(qwp2*qwp2+qxp2*qxp2+qyp2*qyp2+qzp2*qzp2)
        qxp2 = qxp2/normalizer2
        qyp2 = qyp2/normalizer2
        qzp2 = qzp2/normalizer2
        qwp2 = qwp2/normalizer2
        inner_product = qwp1*qwp2+qxp1*qxp2+qyp1*qyp2+qzp1*qzp2
        quaternion_angle = np.arccos(2*inner_product*inner_product-1)
        list_q_error.append(quaternion_angle)
    array_q_error = np.array(list_q_error)
    return array_q_error 

filePath_dataset = '/data/mustard/'
gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'translation_x_gt_velocity_1m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'translation_y_gt_velocity_1m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'translation_z_gt_velocity_1m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'roll_gt_velocity_2rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'pitch_gt_velocity_2rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'yaw_gt_velocity_2rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gt_trans_x['t'] = (gt_trans_x['t']-gt_trans_x['t'][0])*10
gt_trans_x['x'] = gt_trans_x['x']*0.01
gt_trans_x['y'] = gt_trans_x['y']*0.01
gt_trans_x['z'] = gt_trans_x['z']*0.01

gt_trans_y['t'] = (gt_trans_y['t']-gt_trans_y['t'][0])*10
gt_trans_y['x'] = gt_trans_y['x']*0.01
gt_trans_y['y'] = gt_trans_y['y']*0.01
gt_trans_y['z'] = gt_trans_y['z']*0.01

gt_trans_z['t'] = (gt_trans_z['t']-gt_trans_z['t'][0])*10
gt_trans_z['x'] = gt_trans_z['x']*0.01
gt_trans_z['y'] = gt_trans_z['y']*0.01
gt_trans_z['z'] = gt_trans_z['z']*0.01

gt_roll['t'] = (gt_roll['t']-gt_roll['t'][0])*10
gt_roll['x'] = gt_roll['x']*0.01
gt_roll['y'] = gt_roll['y']*0.01
gt_roll['z'] = gt_roll['z']*0.01

gt_pitch['t'] = (gt_pitch['t']-gt_pitch['t'][0])*10
gt_pitch['x'] = gt_pitch['x']*0.01
gt_pitch['y'] = gt_pitch['y']*0.01
gt_pitch['z'] = gt_pitch['z']*0.01

gt_yaw['t'] = (gt_yaw['t']-gt_yaw['t'][0])*10
gt_yaw['x'] = gt_yaw['x']*0.01
gt_yaw['y'] = gt_yaw['y']*0.01
gt_yaw['z'] = gt_yaw['z']*0.01

ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion1_10_cm_s_eros_7_0.7_scale_100_canny_40_sequential_loop.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion2.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion3_10_cm_s_eros_7_0.75_scale_300_canny_40_40_x3.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion4.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion5.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion6.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

# rgbde_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/motion1.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
# rgbde_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'rgbde/motion2.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
# rgbde_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'rgbde/motion3.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
# rgbde_roll = np.genfromtxt(os.path.join(filePath_dataset, 'rgbde/motion4.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
# rgbde_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'rgbde/motion5.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
# rgbde_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'rgbde/motion6.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

fig1, axs = plt.subplots(4,3)
fig1.set_size_inches(18.5, 15.5)
# axs[0,0].plot(ekom_trans_x['t'], ekom_trans_x['x'], 'r', label='x')
# axs[0,0].plot(ekom_trans_x['t'], ekom_trans_x['y'], 'g', label='y')
# axs[0,0].plot(ekom_trans_x['t'], ekom_trans_x['z'], 'b', label='z')
axs[0,0].plot(gt_trans_x['t'], gt_trans_x['x'], 'r--')
axs[0,0].plot(gt_trans_x['t'], gt_trans_x['y'], 'g--')
axs[0,0].plot(gt_trans_x['t'], gt_trans_x['z'], 'b--')
# axs[0,0].plot(rgbde_trans_x['t'], rgbde_trans_x['x'], 'r-.')
# axs[0,0].plot(rgbde_trans_x['t'], rgbde_trans_x['y'], 'g-.')
# axs[0,0].plot(rgbde_trans_x['t'], rgbde_trans_x['z'], 'b-.')
# axs[0,1].plot(ekom_trans_y['t'], ekom_trans_y['x'], 'r', label='x')
# axs[0,1].plot(ekom_trans_y['t'], ekom_trans_y['y'], 'g', label='y')
# axs[0,1].plot(ekom_trans_y['t'], ekom_trans_y['z'], 'b', label='z')
axs[0,1].plot(gt_trans_y['t'], gt_trans_y['x'], 'r--')
axs[0,1].plot(gt_trans_y['t'], gt_trans_y['y'], 'g--')
axs[0,1].plot(gt_trans_y['t'], gt_trans_y['z'], 'b--')
# axs[0,1].plot(rgbde_trans_y['t'], rgbde_trans_y['x'], 'r-.')
# axs[0,1].plot(rgbde_trans_y['t'], rgbde_trans_y['y'], 'g-.')
# axs[0,1].plot(rgbde_trans_y['t'], rgbde_trans_y['z'], 'b-.')
# axs[0,2].plot(ekom_trans_z['t'], ekom_trans_z['x'], 'r', label='x')
# axs[0,2].plot(ekom_trans_z['t'], ekom_trans_z['y'], 'g', label='y')
# axs[0,2].plot(ekom_trans_z['t'], ekom_trans_z['z'], 'b', label='z')
axs[0,2].plot(gt_trans_z['t'], gt_trans_z['x'], 'r--')
axs[0,2].plot(gt_trans_z['t'], gt_trans_z['y'], 'g--')
axs[0,2].plot(gt_trans_z['t'], gt_trans_z['z'], 'b--')
# axs[0,2].plot(rgbde_trans_z['t'], rgbde_trans_z['x'], 'r-.')
# axs[0,2].plot(rgbde_trans_z['t'], rgbde_trans_z['y'], 'g-.')
# axs[0,2].plot(rgbde_trans_z['t'], rgbde_trans_z['z'], 'b-.')
axs[2,0].plot(ekom_roll['t'], ekom_roll['x'], 'r', label='x')
axs[2,0].plot(ekom_roll['t'], ekom_roll['y'], 'g', label='y')
axs[2,0].plot(ekom_roll['t'], ekom_roll['z'], 'b', label='z')
axs[2,0].plot(gt_roll['t'], gt_roll['x'], 'r--')
axs[2,0].plot(gt_roll['t'], gt_roll['y'], 'g--')
axs[2,0].plot(gt_roll['t'], gt_roll['z'], 'b--')
# axs[2,0].plot(rgbde_roll['t'], rgbde_roll['x'], 'r-.')
# axs[2,0].plot(rgbde_roll['t'], rgbde_roll['y'], 'g-.')
# axs[2,0].plot(rgbde_roll['t'], rgbde_roll['z'], 'b-.')
axs[2,1].plot(ekom_pitch['t'], ekom_pitch['x'], 'r', label='x')
axs[2,1].plot(ekom_pitch['t'], ekom_pitch['y'], 'g', label='y')
axs[2,1].plot(ekom_pitch['t'], ekom_pitch['z'], 'b', label='z')
axs[2,1].plot(gt_pitch['t'], gt_pitch['x'], 'r--')
axs[2,1].plot(gt_pitch['t'], gt_pitch['y'], 'g--')
axs[2,1].plot(gt_pitch['t'], gt_pitch['z'], 'b--')
# axs[2,1].plot(rgbde_pitch['t'], rgbde_pitch['x'], 'r-.')
# axs[2,1].plot(rgbde_pitch['t'], rgbde_pitch['y'], 'g-.')
# axs[2,1].plot(rgbde_pitch['t'], rgbde_pitch['z'], 'b-.')
# axs[2,2].plot(ekom_yaw['t'], ekom_yaw['x'], 'r', label='x')
# axs[2,2].plot(ekom_yaw['t'], ekom_yaw['y'], 'g', label='y')
# axs[2,2].plot(ekom_yaw['t'], ekom_yaw['z'], 'b', label='z')
axs[2,2].plot(gt_yaw['t'], gt_yaw['x'], 'r--')
axs[2,2].plot(gt_yaw['t'], gt_yaw['y'], 'g--')
axs[2,2].plot(gt_yaw['t'], gt_yaw['z'], 'b--')
# axs[2,2].plot(rgbde_yaw['t'], rgbde_yaw['x'], 'r-.')
# axs[2,2].plot(rgbde_yaw['t'], rgbde_yaw['y'], 'g-.')
# axs[2,2].plot(rgbde_yaw['t'], rgbde_yaw['z'], 'b-.')
# axs[1,0].plot(ekom_trans_x['t'], ekom_trans_x['qx'], 'k', label='qx')
# axs[1,0].plot(ekom_trans_x['t'], ekom_trans_x['qy'], 'y', label='qy')
# axs[1,0].plot(ekom_trans_x['t'], ekom_trans_x['qz'], 'm', label='qz')
# axs[1,0].plot(ekom_trans_x['t'], ekom_trans_x['qw'], 'c', label='qw')
axs[1,0].plot(gt_trans_x['t'], gt_trans_x['qx'], 'k--')
axs[1,0].plot(gt_trans_x['t'], gt_trans_x['qy'], 'y--')
axs[1,0].plot(gt_trans_x['t'], gt_trans_x['qz'], 'm--')
axs[1,0].plot(gt_trans_x['t'], gt_trans_x['qw'], 'c--')
# axs[1,0].plot(rgbde_trans_x['t'], rgbde_trans_x['qx'], 'k-.')
# axs[1,0].plot(rgbde_trans_x['t'], rgbde_trans_x['qy'], 'y-.')
# axs[1,0].plot(rgbde_trans_x['t'], rgbde_trans_x['qz'], 'm-.')
# axs[1,0].plot(rgbde_trans_x['t'], rgbde_trans_x['qw'], 'c-.')
# axs[1,1].plot(ekom_trans_y['t'], ekom_trans_y['qx'], 'k', label='qx')
# axs[1,1].plot(ekom_trans_y['t'], ekom_trans_y['qy'], 'y', label='qy')
# axs[1,1].plot(ekom_trans_y['t'], ekom_trans_y['qz'], 'm', label='qz')
# axs[1,1].plot(ekom_trans_y['t'], ekom_trans_y['qw'], 'c', label='qw')
axs[1,1].plot(gt_trans_y['t'], gt_trans_y['qx'], 'k--')
axs[1,1].plot(gt_trans_y['t'], gt_trans_y['qy'], 'y--')
axs[1,1].plot(gt_trans_y['t'], gt_trans_y['qz'], 'm--')
axs[1,1].plot(gt_trans_y['t'], gt_trans_y['qw'], 'c--')
# axs[1,1].plot(rgbde_trans_y['t'], rgbde_trans_y['qx'], 'k-.')
# axs[1,1].plot(rgbde_trans_y['t'], rgbde_trans_y['qy'], 'y-.')
# axs[1,1].plot(rgbde_trans_y['t'], rgbde_trans_y['qz'], 'm-.')
# axs[1,1].plot(rgbde_trans_y['t'], rgbde_trans_y['qw'], 'c-.')
# axs[1,2].plot(ekom_trans_z['t'], ekom_trans_z['qx'], 'k', label='qx')
# axs[1,2].plot(ekom_trans_z['t'], ekom_trans_z['qy'], 'y', label='qy')
# axs[1,2].plot(ekom_trans_z['t'], ekom_trans_z['qz'], 'm', label='qz')
# axs[1,2].plot(ekom_trans_z['t'], ekom_trans_z['qw'], 'c', label='qw')
axs[1,2].plot(gt_trans_z['t'], gt_trans_z['qx'], 'k--')
axs[1,2].plot(gt_trans_z['t'], gt_trans_z['qy'], 'y--')
axs[1,2].plot(gt_trans_z['t'], gt_trans_z['qz'], 'm--')
axs[1,2].plot(gt_trans_z['t'], gt_trans_z['qw'], 'c--')
# axs[1,2].plot(rgbde_trans_z['t'], rgbde_trans_z['qx'], 'k-.')
# axs[1,2].plot(rgbde_trans_z['t'], rgbde_trans_z['qy'], 'y-.')
# axs[1,2].plot(rgbde_trans_z['t'], rgbde_trans_z['qz'], 'm-.')
# axs[1,2].plot(rgbde_trans_z['t'], rgbde_trans_z['qw'], 'c-.')
axs[3,0].plot(ekom_roll['t'], ekom_roll['qx'], 'k', label='qx')
axs[3,0].plot(ekom_roll['t'], ekom_roll['qy'], 'y', label='qy')
axs[3,0].plot(ekom_roll['t'], ekom_roll['qz'], 'm', label='qz')
axs[3,0].plot(ekom_roll['t'], ekom_roll['qw'], 'c', label='qw')
axs[3,0].plot(gt_roll['t'], gt_roll['qx'], 'k--')
axs[3,0].plot(gt_roll['t'], gt_roll['qy'], 'y--')
axs[3,0].plot(gt_roll['t'], gt_roll['qz'], 'm--')
axs[3,0].plot(gt_roll['t'], gt_roll['qw'], 'c--')
# axs[3,0].plot(rgbde_roll['t'], rgbde_roll['qx'], 'k-.')
# axs[3,0].plot(rgbde_roll['t'], rgbde_roll['qy'], 'y-.')
# axs[3,0].plot(rgbde_roll['t'], rgbde_roll['qz'], 'm-.')
# axs[3,0].plot(rgbde_roll['t'], rgbde_roll['qw'], 'c-.')
axs[3,1].plot(ekom_pitch['t'], ekom_pitch['qx'], 'k', label='qx')
axs[3,1].plot(ekom_pitch['t'], ekom_pitch['qy'], 'y', label='qy')
axs[3,1].plot(ekom_pitch['t'], ekom_pitch['qz'], 'm', label='qz')
axs[3,1].plot(ekom_pitch['t'], ekom_pitch['qw'], 'c', label='qw')
axs[3,1].plot(gt_pitch['t'], gt_pitch['qx'], 'k--')
axs[3,1].plot(gt_pitch['t'], gt_pitch['qy'], 'y--')
axs[3,1].plot(gt_pitch['t'], gt_pitch['qz'], 'm--')
axs[3,1].plot(gt_pitch['t'], gt_pitch['qw'], 'c--')
# axs[3,1].plot(rgbde_pitch['t'], rgbde_pitch['qx'], 'k-.')
# axs[3,1].plot(rgbde_pitch['t'], rgbde_pitch['qy'], 'y-.')
# axs[3,1].plot(rgbde_pitch['t'], rgbde_pitch['qz'], 'm-.')
# axs[3,1].plot(rgbde_pitch['t'], rgbde_pitch['qw'], 'c-.')
# axs[3,2].plot(ekom_yaw['t'], ekom_yaw['qx'], 'k', label='qx')
# axs[3,2].plot(ekom_yaw['t'], ekom_yaw['qy'], 'y', label='qy')
# axs[3,2].plot(ekom_yaw['t'], ekom_yaw['qz'], 'm', label='qz')
# axs[3,2].plot(ekom_yaw['t'], ekom_yaw['qw'], 'c', label='qw')
axs[3,2].plot(gt_yaw['t'], gt_yaw['qx'], 'k--')
axs[3,2].plot(gt_yaw['t'], gt_yaw['qy'], 'y--')
axs[3,2].plot(gt_yaw['t'], gt_yaw['qz'], 'm--')
axs[3,2].plot(gt_yaw['t'], gt_yaw['qw'], 'c--')
# axs[3,2].plot(rgbde_yaw['t'], rgbde_yaw['qx'], 'k-.')
# axs[3,2].plot(rgbde_yaw['t'], rgbde_yaw['qy'], 'y-.')
# axs[3,2].plot(rgbde_yaw['t'], rgbde_yaw['qz'], 'm-.')
# axs[3,2].plot(rgbde_yaw['t'], rgbde_yaw['qw'], 'c-.')
axs[0,0].legend(loc='upper right')
axs[1,0].legend(loc='upper right')
axs[0,0].set_xticks([])
axs[1,0].set_xticks([])
axs[2,0].set_xticks([])
axs[0,1].set_xticks([])
axs[1,1].set_xticks([])
axs[2,1].set_xticks([])
axs[0,2].set_xticks([])
axs[1,2].set_xticks([])
axs[2,2].set_xticks([])
axs[0,0].set(ylabel='Position[m]')
axs[1,0].set(ylabel='Quaternions')
axs[2,0].set(ylabel='Position[m]')
axs[3,0].set(xlabel='Time [s]', ylabel='Quaternions')
axs[3,1].set(xlabel='Time [s]')
axs[3,2].set(xlabel='Time [s]')
plt.show()

# fig2, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['x'], 'r', label='x')
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['y'], 'g', label='y')
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['z'], 'b', label='z')
# ax1.plot(gt_trans_y['t'], gt_trans_y['x'], 'r--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['y'], 'g--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['z'], 'b--')
# # ax1.plot(rgbde_trans_y['t'], rgbde_trans_y['x'], 'r-.')
# # ax1.plot(rgbde_trans_y['t'], rgbde_trans_y['y'], 'g-.')
# # ax1.plot(rgbde_trans_y['t'], rgbde_trans_y['z'], 'b-.')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qx'], 'k', label='qx')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qy'], 'y', label='qy')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qz'], 'm', label='qz')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qw'], 'c', label='qw')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qx'], 'k--')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qy'], 'y--')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qz'], 'm--')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qw'], 'c--')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qx'], 'k-.')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qy'], 'y-.')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qz'], 'm-.')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

# fig3, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_trans_z['t'], ekom_trans_z['x'], 'r', label='x')
# ax1.plot(ekom_trans_z['t'], ekom_trans_z['y'], 'g', label='y')
# ax1.plot(ekom_trans_z['t'], ekom_trans_z['z'], 'b', label='z')
# ax1.plot(gt_trans_z['t'], gt_trans_z['x'], 'r--')
# ax1.plot(gt_trans_z['t'], gt_trans_z['y'], 'g--')
# ax1.plot(gt_trans_z['t'], gt_trans_z['z'], 'b--')
# # ax1.plot(rgbde_trans_z['t'], rgbde_trans_z['x'], 'r-.')
# # ax1.plot(rgbde_trans_z['t'], rgbde_trans_z['y'], 'g-.')
# # ax1.plot(rgbde_trans_z['t'], rgbde_trans_z['z'], 'b-.')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qx'], 'k', label='qx')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qy'], 'y', label='qy')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qz'], 'm', label='qz')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qw'], 'c', label='qw')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qx'], 'k--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qy'], 'y--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qz'], 'm--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qw'], 'c--')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qx'], 'k-.')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qy'], 'y-.')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qz'], 'm-.')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()
# ax1.plot(gt_roll['t'], gt_roll['x'], 'r--')
# ax1.plot(gt_roll['t'], gt_roll['y'], 'g--')
# ax1.plot(gt_roll['t'], gt_roll['z'], 'b--')
# # ax1.plot(rgbde_roll['t'], rgbde_roll['x'], 'r-.')
# # ax1.plot(rgbde_roll['t'], rgbde_roll['y'], 'g-.')
# # ax1.plot(rgbde_roll['t'], rgbde_roll['z'], 'b-.')
# ax2.plot(ekom_roll['t'], ekom_roll['qx'], 'k', label='qx')
# ax2.plot(ekom_roll['t'], ekom_roll['qy'], 'y', label='qy')
# ax2.plot(ekom_roll['t'], ekom_roll['qz'], 'm', label='qz')
# ax2.plot(ekom_roll['t'], ekom_roll['qw'], 'c', label='qw')
# ax2.plot(gt_roll['t'], gt_roll['qx'], 'k--')
# ax2.plot(gt_roll['t'], gt_roll['qy'], 'y--')
# ax2.plot(gt_roll['t'], gt_roll['qz'], 'm--')
# ax2.plot(gt_roll['t'], gt_roll['qw'], 'c--')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qx'], 'k-.')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qy'], 'y-.')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qz'], 'm-.')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

fig5, (ax1, ax2) = plt.subplots(2)
ax1.plot(ekom_pitch['t'], ekom_pitch['x'], 'r', label='x')
ax1.plot(ekom_pitch['t'], ekom_pitch['y'], 'g', label='y')
ax1.plot(ekom_pitch['t'], ekom_pitch['z'], 'b', label='z')
ax1.plot(gt_pitch['t'], gt_pitch['x'], 'r--')
ax1.plot(gt_pitch['t'], gt_pitch['y'], 'g--')
ax1.plot(gt_pitch['t'], gt_pitch['z'], 'b--')
# ax1.plot(rgbde_pitch['t'], rgbde_pitch['x'], 'r-.')
# ax1.plot(rgbde_pitch['t'], rgbde_pitch['y'], 'g-.')
# ax1.plot(rgbde_pitch['t'], rgbde_pitch['z'], 'b-.')
ax2.plot(ekom_pitch['t'], ekom_pitch['qx'], 'k', label='qx')
ax2.plot(ekom_pitch['t'], ekom_pitch['qy'], 'y', label='qy')
ax2.plot(ekom_pitch['t'], ekom_pitch['qz'], 'm', label='qz')
ax2.plot(ekom_pitch['t'], ekom_pitch['qw'], 'c', label='qw')
ax2.plot(gt_pitch['t'], gt_pitch['qx'], 'k--')
ax2.plot(gt_pitch['t'], gt_pitch['qy'], 'y--')
ax2.plot(gt_pitch['t'], gt_pitch['qz'], 'm--')
ax2.plot(gt_pitch['t'], gt_pitch['qw'], 'c--')
# ax2.plot(rgbde_pitch['t'], rgbde_pitch['qx'], 'k-.')
# ax2.plot(rgbde_pitch['t'], rgbde_pitch['qy'], 'y-.')
# ax2.plot(rgbde_pitch['t'], rgbde_pitch['qz'], 'm-.')
# ax2.plot(rgbde_pitch['t'], rgbde_pitch['qw'], 'c-.')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax1.set_xticks([])
ax1.set(ylabel='Position[m]')
ax2.set(xlabel='Time [s]', ylabel='Quaternions')
plt.show()

fig6, (ax1, ax2) = plt.subplots(2)
ax1.plot(ekom_yaw['t'], ekom_yaw['x'], 'r', label='x')
ax1.plot(ekom_yaw['t'], ekom_yaw['y'], 'g', label='y')
ax1.plot(ekom_yaw['t'], ekom_yaw['z'], 'b', label='z')
ax1.plot(gt_yaw['t'], gt_yaw['x'], 'r--')
ax1.plot(gt_yaw['t'], gt_yaw['y'], 'g--')
ax1.plot(gt_yaw['t'], gt_yaw['z'], 'b--')
# ax1.plot(rgbde_yaw['t'], rgbde_yaw['x'], 'r-.')
# ax1.plot(rgbde_yaw['t'], rgbde_yaw['y'], 'g-.')
# ax1.plot(rgbde_yaw['t'], rgbde_yaw['z'], 'b-.')
ax2.plot(ekom_yaw['t'], ekom_yaw['qx'], 'k', label='qx')
ax2.plot(ekom_yaw['t'], ekom_yaw['qy'], 'y', label='qy')
ax2.plot(ekom_yaw['t'], ekom_yaw['qz'], 'm', label='qz')
ax2.plot(ekom_yaw['t'], ekom_yaw['qw'], 'c', label='qw')
ax2.plot(gt_yaw['t'], gt_yaw['qx'], 'k--')
ax2.plot(gt_yaw['t'], gt_yaw['qy'], 'y--')
ax2.plot(gt_yaw['t'], gt_yaw['qz'], 'm--')
ax2.plot(gt_yaw['t'], gt_yaw['qw'], 'c--')
# ax2.plot(rgbde_yaw['t'], rgbde_yaw['qx'], 'k-.')
# ax2.plot(rgbde_yaw['t'], rgbde_yaw['qy'], 'y-.')
# ax2.plot(rgbde_yaw['t'], rgbde_yaw['qz'], 'm-.')
# ax2.plot(rgbde_yaw['t'], rgbde_yaw['qw'], 'c-.')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax1.set_xticks([])
ax1.set(ylabel='Position[m]')
ax2.set(xlabel='Time [s]', ylabel='Quaternions')
plt.show()

# ekom_error_trans_x = computeEuclideanDistance(gt_trans_x['x'], gt_trans_x['y'], ekom_trans_x['x'], ekom_trans_x['y'])
# ekom_q_angle_trans_x = computeQuaternionError(ekom_trans_x['qx'], ekom_trans_x['qy'], ekom_trans_x['qz'], ekom_trans_x['qw'], gt_trans_x['qx'], gt_trans_x['qy'], gt_trans_x['qz'], gt_trans_x['qw'])
# print(np.mean(ekom_q_angle_trans_x))

# ---------------------------------------------------------------------------  DRAGON  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/dragon/'
gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_x_1_m_s_2/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_y_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_z_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_roll_4rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_pitch_4rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_yaw_4rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gt_trans_x['t'] = (gt_trans_x['t']-gt_trans_x['t'][0])*10
gt_trans_x['x'] = gt_trans_x['x']*0.01
gt_trans_x['y'] = gt_trans_x['y']*0.01
gt_trans_x['z'] = gt_trans_x['z']*0.01

gt_trans_y['t'] = (gt_trans_y['t']-gt_trans_y['t'][0])*10
gt_trans_y['x'] = gt_trans_y['x']*0.01
gt_trans_y['y'] = gt_trans_y['y']*0.01
gt_trans_y['z'] = gt_trans_y['z']*0.01

gt_trans_z['t'] = (gt_trans_z['t']-gt_trans_z['t'][0])*10
gt_trans_z['x'] = gt_trans_z['x']*0.01
gt_trans_z['y'] = gt_trans_z['y']*0.01
gt_trans_z['z'] = gt_trans_z['z']*0.01

gt_roll['t'] = (gt_roll['t']-gt_roll['t'][0])*10
gt_roll['x'] = gt_roll['x']*0.01
gt_roll['y'] = gt_roll['y']*0.01
gt_roll['z'] = gt_roll['z']*0.01

gt_pitch['t'] = (gt_pitch['t']-gt_pitch['t'][0])*10
gt_pitch['x'] = gt_pitch['x']*0.01
gt_pitch['y'] = gt_pitch['y']*0.01
gt_pitch['z'] = gt_pitch['z']*0.01

gt_yaw['t'] = (gt_yaw['t']-gt_yaw['t'][0])*10
gt_yaw['x'] = gt_yaw['x']*0.01
gt_yaw['y'] = gt_yaw['y']*0.01
gt_yaw['z'] = gt_yaw['z']*0.01

ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion1_10_cm_s_eros_7_0.75_scale_300_canny_40_x2_th_3.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion3.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion3.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion4.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion5.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'ekom/motion3.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

rgbde_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_translation_x_1_m_s_2_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_translation_y_1_m_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_translation_z_1_m_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_roll = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_roll_4rad_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_pitch_4rad_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
# rgbde_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/motion6.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

rgbde_time_trans_x = (np.arange(0, 1/60*len(rgbde_trans_x['x']), 1/60))*10
rgbde_time_trans_y = (np.arange(0, 1/60*len(rgbde_trans_y['x']), 1/60))*10
rgbde_time_trans_z = (np.arange(0, 1/60*len(rgbde_trans_z['x']), 1/60))*10
rgbde_time_roll = (np.arange(0, 1/60*len(rgbde_roll['qx']), 1/60))*10
rgbde_time_pitch = (np.arange(0, 1/60*len(rgbde_pitch['qx']), 1/60))*10
# rgbde_time_yaw = (np.arange(0, 1/60*len(rgbde_yaw['qx']), 1/60))*10

def resampling_by_interpolate(time_samples, x_values, y_values):
    f_neareast = interp1d(x_values, y_values, kind='nearest', fill_value="extrapolate")
    resampled = f_neareast(time_samples)
    return resampled

ekom_trans_x_resampled_x = resampling_by_interpolate(gt_trans_x['t'], ekom_trans_x['t'], ekom_trans_x['x'])
ekom_trans_x_resampled_y = resampling_by_interpolate(gt_trans_x['t'], ekom_trans_x['t'], ekom_trans_x['y'])
ekom_trans_x_resampled_z = resampling_by_interpolate(gt_trans_x['t'], ekom_trans_x['t'], ekom_trans_x['z'])
ekom_trans_x_resampled_qx = resampling_by_interpolate(gt_trans_x['t'], ekom_trans_x['t'], ekom_trans_x['qx'])
ekom_trans_x_resampled_qy = resampling_by_interpolate(gt_trans_x['t'], ekom_trans_x['t'], ekom_trans_x['qy'])
ekom_trans_x_resampled_qz = resampling_by_interpolate(gt_trans_x['t'], ekom_trans_x['t'], ekom_trans_x['qz'])
ekom_trans_x_resampled_qw = resampling_by_interpolate(gt_trans_x['t'], ekom_trans_x['t'], ekom_trans_x['qw'])
rgbde_trans_x_resampled_x = resampling_by_interpolate(gt_trans_x['t'], rgbde_time_trans_x, rgbde_trans_x['x'])
rgbde_trans_x_resampled_y = resampling_by_interpolate(gt_trans_x['t'], rgbde_time_trans_x, rgbde_trans_x['y'])
rgbde_trans_x_resampled_z = resampling_by_interpolate(gt_trans_x['t'], rgbde_time_trans_x, rgbde_trans_x['z'])
rgbde_trans_x_resampled_qx = resampling_by_interpolate(gt_trans_x['t'], rgbde_time_trans_x, rgbde_trans_x['qx'])
rgbde_trans_x_resampled_qy = resampling_by_interpolate(gt_trans_x['t'], rgbde_time_trans_x, rgbde_trans_x['qy'])
rgbde_trans_x_resampled_qz = resampling_by_interpolate(gt_trans_x['t'], rgbde_time_trans_x, rgbde_trans_x['qz'])
rgbde_trans_x_resampled_qw = resampling_by_interpolate(gt_trans_x['t'], rgbde_time_trans_x, rgbde_trans_x['qw'])

ekom_trans_y_resampled_x = resampling_by_interpolate(gt_trans_y['t'], ekom_trans_y['t'], ekom_trans_y['x'])
ekom_trans_y_resampled_y = resampling_by_interpolate(gt_trans_y['t'], ekom_trans_y['t'], ekom_trans_y['y'])
ekom_trans_y_resampled_z = resampling_by_interpolate(gt_trans_y['t'], ekom_trans_y['t'], ekom_trans_y['z'])
ekom_trans_y_resampled_qx = resampling_by_interpolate(gt_trans_y['t'], ekom_trans_y['t'], ekom_trans_y['qx'])
ekom_trans_y_resampled_qy = resampling_by_interpolate(gt_trans_y['t'], ekom_trans_y['t'], ekom_trans_y['qy'])
ekom_trans_y_resampled_qz = resampling_by_interpolate(gt_trans_y['t'], ekom_trans_y['t'], ekom_trans_y['qz'])
ekom_trans_y_resampled_qw = resampling_by_interpolate(gt_trans_y['t'], ekom_trans_y['t'], ekom_trans_y['qw'])
rgbde_trans_y_resampled_x = resampling_by_interpolate(gt_trans_y['t'], rgbde_time_trans_y, rgbde_trans_y['x'])
rgbde_trans_y_resampled_y = resampling_by_interpolate(gt_trans_y['t'], rgbde_time_trans_y, rgbde_trans_y['y'])
rgbde_trans_y_resampled_z = resampling_by_interpolate(gt_trans_y['t'], rgbde_time_trans_y, rgbde_trans_y['z'])
rgbde_trans_y_resampled_qx = resampling_by_interpolate(gt_trans_y['t'], rgbde_time_trans_y, rgbde_trans_y['qx'])
rgbde_trans_y_resampled_qy = resampling_by_interpolate(gt_trans_y['t'], rgbde_time_trans_y, rgbde_trans_y['qy'])
rgbde_trans_y_resampled_qz = resampling_by_interpolate(gt_trans_y['t'], rgbde_time_trans_y, rgbde_trans_y['qz'])
rgbde_trans_y_resampled_qw = resampling_by_interpolate(gt_trans_y['t'], rgbde_time_trans_y, rgbde_trans_y['qw'])

ekom_trans_z_resampled_x = resampling_by_interpolate(gt_trans_z['t'], ekom_trans_z['t'], ekom_trans_z['x'])
ekom_trans_z_resampled_y = resampling_by_interpolate(gt_trans_z['t'], ekom_trans_z['t'], ekom_trans_z['y'])
ekom_trans_z_resampled_z = resampling_by_interpolate(gt_trans_z['t'], ekom_trans_z['t'], ekom_trans_z['z'])
ekom_trans_z_resampled_qx = resampling_by_interpolate(gt_trans_z['t'], ekom_trans_z['t'], ekom_trans_z['qx'])
ekom_trans_z_resampled_qy = resampling_by_interpolate(gt_trans_z['t'], ekom_trans_z['t'], ekom_trans_z['qy'])
ekom_trans_z_resampled_qz = resampling_by_interpolate(gt_trans_z['t'], ekom_trans_z['t'], ekom_trans_z['qz'])
ekom_trans_z_resampled_qw = resampling_by_interpolate(gt_trans_z['t'], ekom_trans_z['t'], ekom_trans_z['qw'])
rgbde_trans_z_resampled_x = resampling_by_interpolate(gt_trans_z['t'], rgbde_time_trans_z, rgbde_trans_z['x'])
rgbde_trans_z_resampled_y = resampling_by_interpolate(gt_trans_z['t'], rgbde_time_trans_z, rgbde_trans_z['y'])
rgbde_trans_z_resampled_z = resampling_by_interpolate(gt_trans_z['t'], rgbde_time_trans_z, rgbde_trans_z['z'])
rgbde_trans_z_resampled_qx = resampling_by_interpolate(gt_trans_z['t'], rgbde_time_trans_z, rgbde_trans_z['qx'])
rgbde_trans_z_resampled_qy = resampling_by_interpolate(gt_trans_z['t'], rgbde_time_trans_z, rgbde_trans_z['qy'])
rgbde_trans_z_resampled_qz = resampling_by_interpolate(gt_trans_z['t'], rgbde_time_trans_z, rgbde_trans_z['qz'])
rgbde_trans_z_resampled_qw = resampling_by_interpolate(gt_trans_z['t'], rgbde_time_trans_z, rgbde_trans_z['qw'])

ekom_roll_resampled_x = resampling_by_interpolate(gt_roll['t'], ekom_roll['t'], ekom_roll['x'])
ekom_roll_resampled_y = resampling_by_interpolate(gt_roll['t'], ekom_roll['t'], ekom_roll['y'])
ekom_roll_resampled_z = resampling_by_interpolate(gt_roll['t'], ekom_roll['t'], ekom_roll['z'])
ekom_roll_resampled_qx = resampling_by_interpolate(gt_roll['t'], ekom_roll['t'], ekom_roll['qx'])
ekom_roll_resampled_qy = resampling_by_interpolate(gt_roll['t'], ekom_roll['t'], ekom_roll['qy'])
ekom_roll_resampled_qz = resampling_by_interpolate(gt_roll['t'], ekom_roll['t'], ekom_roll['qz'])
ekom_roll_resampled_qw = resampling_by_interpolate(gt_roll['t'], ekom_roll['t'], ekom_roll['qw'])
rgbde_roll_resampled_x = resampling_by_interpolate(gt_roll['t'], rgbde_time_roll, rgbde_roll['x'])
rgbde_roll_resampled_y = resampling_by_interpolate(gt_roll['t'], rgbde_time_roll, rgbde_roll['y'])
rgbde_roll_resampled_z = resampling_by_interpolate(gt_roll['t'], rgbde_time_roll, rgbde_roll['z'])
rgbde_roll_resampled_qx = resampling_by_interpolate(gt_roll['t'], rgbde_time_roll, rgbde_roll['qx'])
rgbde_roll_resampled_qy = resampling_by_interpolate(gt_roll['t'], rgbde_time_roll, rgbde_roll['qy'])
rgbde_roll_resampled_qz = resampling_by_interpolate(gt_roll['t'], rgbde_time_roll, rgbde_roll['qz'])
rgbde_roll_resampled_qw = resampling_by_interpolate(gt_roll['t'], rgbde_time_roll, rgbde_roll['qw'])

ekom_pitch_resampled_x = resampling_by_interpolate(gt_pitch['t'], ekom_pitch['t'], ekom_pitch['x'])
ekom_pitch_resampled_y = resampling_by_interpolate(gt_pitch['t'], ekom_pitch['t'], ekom_pitch['y'])
ekom_pitch_resampled_z = resampling_by_interpolate(gt_pitch['t'], ekom_pitch['t'], ekom_pitch['z'])
ekom_pitch_resampled_qx = resampling_by_interpolate(gt_pitch['t'], ekom_pitch['t'], ekom_pitch['qx'])
ekom_pitch_resampled_qy = resampling_by_interpolate(gt_pitch['t'], ekom_pitch['t'], ekom_pitch['qy'])
ekom_pitch_resampled_qz = resampling_by_interpolate(gt_pitch['t'], ekom_pitch['t'], ekom_pitch['qz'])
ekom_pitch_resampled_qw = resampling_by_interpolate(gt_pitch['t'], ekom_pitch['t'], ekom_pitch['qw'])
rgbde_pitch_resampled_x = resampling_by_interpolate(gt_pitch['t'], rgbde_time_pitch, rgbde_pitch['x'])
rgbde_pitch_resampled_y = resampling_by_interpolate(gt_pitch['t'], rgbde_time_pitch, rgbde_pitch['y'])
rgbde_pitch_resampled_z = resampling_by_interpolate(gt_pitch['t'], rgbde_time_pitch, rgbde_pitch['z'])
rgbde_pitch_resampled_qx = resampling_by_interpolate(gt_pitch['t'], rgbde_time_pitch, rgbde_pitch['qx'])
rgbde_pitch_resampled_qy = resampling_by_interpolate(gt_pitch['t'], rgbde_time_pitch, rgbde_pitch['qy'])
rgbde_pitch_resampled_qz = resampling_by_interpolate(gt_pitch['t'], rgbde_time_pitch, rgbde_pitch['qz'])
rgbde_pitch_resampled_qw = resampling_by_interpolate(gt_pitch['t'], rgbde_time_pitch, rgbde_pitch['qw'])

ekom_yaw_resampled_x = resampling_by_interpolate(gt_yaw['t'], ekom_yaw['t'], ekom_yaw['x'])
ekom_yaw_resampled_y = resampling_by_interpolate(gt_yaw['t'], ekom_yaw['t'], ekom_yaw['y'])
ekom_yaw_resampled_z = resampling_by_interpolate(gt_yaw['t'], ekom_yaw['t'], ekom_yaw['z'])
ekom_yaw_resampled_qx = resampling_by_interpolate(gt_yaw['t'], ekom_yaw['t'], ekom_yaw['qx'])
ekom_yaw_resampled_qy = resampling_by_interpolate(gt_yaw['t'], ekom_yaw['t'], ekom_yaw['qy'])
ekom_yaw_resampled_qz = resampling_by_interpolate(gt_yaw['t'], ekom_yaw['t'], ekom_yaw['qz'])
ekom_yaw_resampled_qw = resampling_by_interpolate(gt_yaw['t'], ekom_yaw['t'], ekom_yaw['qw'])
# rgbde_yaw_resampled_x = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['x'])
# rgbde_yaw_resampled_y = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['y'])
# rgbde_yaw_resampled_z = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['z'])

# fig1, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_trans_x['t'], ekom_trans_x['x'], 'r', label='x')
# ax1.plot(ekom_trans_x['t'], ekom_trans_x['y'], 'g', label='y')
# ax1.plot(ekom_trans_x['t'], ekom_trans_x['z'], 'b', label='z')
# ax1.plot(gt_trans_x['t'], gt_trans_x['x'], 'r--')
# ax1.plot(gt_trans_x['t'], gt_trans_x['y'], 'g--')
# ax1.plot(gt_trans_x['t'], gt_trans_x['z'], 'b--')
# # ax1.plot(rgbde_time, rgbde_trans_x['x'], 'r-.')
# # ax1.plot(rgbde_time, rgbde_trans_x['y'], 'g-.')
# # ax1.plot(rgbde_time, rgbde_trans_x['z'], 'b-.')
# ax2.plot(ekom_trans_x['t'], ekom_trans_x['qx'], 'k', label='qx')
# ax2.plot(ekom_trans_x['t'], ekom_trans_x['qy'], 'y', label='qy')
# ax2.plot(ekom_trans_x['t'], ekom_trans_x['qz'], 'm', label='qz')
# ax2.plot(ekom_trans_x['t'], ekom_trans_x['qw'], 'c', label='qw')
# ax2.plot(gt_trans_x['t'], gt_trans_x['qx'], 'k--')
# ax2.plot(gt_trans_x['t'], gt_trans_x['qy'], 'y--')
# ax2.plot(gt_trans_x['t'], gt_trans_x['qz'], 'm--')
# ax2.plot(gt_trans_x['t'], gt_trans_x['qw'], 'c--')
# # ax2.plot(rgbde_time, rgbde_trans_x['qx'], 'k-.')
# # ax2.plot(rgbde_time, rgbde_trans_x['qy'], 'y-.')
# # ax2.plot(rgbde_time, rgbde_trans_x['qz'], 'm-.')
# # ax2.plot(rgbde_time, rgbde_trans_x['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

# fig2, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['x'], 'r', label='x')
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['y'], 'g', label='y')
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['z'], 'b', label='z')
# ax1.plot(gt_trans_y['t'], gt_trans_y['x'], 'r--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['y'], 'g--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['z'], 'b--')
# # ax1.plot(rgbde_trans_y['t'], rgbde_trans_y['x'], 'r-.')
# # ax1.plot(rgbde_trans_y['t'], rgbde_trans_y['y'], 'g-.')
# # ax1.plot(rgbde_trans_y['t'], rgbde_trans_y['z'], 'b-.')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qx'], 'k', label='qx')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qy'], 'y', label='qy')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qz'], 'm', label='qz')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y['qw'], 'c', label='qw')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qx'], 'k--')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qy'], 'y--')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qz'], 'm--')
# ax2.plot(gt_trans_y['t'], gt_trans_y['qw'], 'c--')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qx'], 'k-.')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qy'], 'y-.')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qz'], 'm-.')
# # ax2.plot(rgbde_trans_y['t'], rgbde_trans_y['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

# fig3, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_trans_z['t'], ekom_trans_z['x'], 'r', label='x')
# ax1.plot(ekom_trans_z['t'], ekom_trans_z['y'], 'g', label='y')
# ax1.plot(ekom_trans_z['t'], ekom_trans_z['z'], 'b', label='z')
# ax1.plot(gt_trans_z['t'], gt_trans_z['x'], 'r--')
# ax1.plot(gt_trans_z['t'], gt_trans_z['y'], 'g--')
# ax1.plot(gt_trans_z['t'], gt_trans_z['z'], 'b--')
# # ax1.plot(rgbde_trans_z['t'], rgbde_trans_z['x'], 'r-.')
# # ax1.plot(rgbde_trans_z['t'], rgbde_trans_z['y'], 'g-.')
# # ax1.plot(rgbde_trans_z['t'], rgbde_trans_z['z'], 'b-.')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qx'], 'k', label='qx')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qy'], 'y', label='qy')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qz'], 'm', label='qz')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qw'], 'c', label='qw')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qx'], 'k--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qy'], 'y--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qz'], 'm--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qw'], 'c--')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qx'], 'k-.')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qy'], 'y-.')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qz'], 'm-.')
# # ax2.plot(rgbde_trans_z['t'], rgbde_trans_z['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

# fig4, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_roll['t'], ekom_roll['x'], 'r', label='x')
# ax1.plot(ekom_roll['t'], ekom_roll['y'], 'g', label='y')
# ax1.plot(ekom_roll['t'], ekom_roll['z'], 'b', label='z')
# ax1.plot(gt_roll['t'], gt_roll['x'], 'r--')
# ax1.plot(gt_roll['t'], gt_roll['y'], 'g--')
# ax1.plot(gt_roll['t'], gt_roll['z'], 'b--')
# # ax1.plot(rgbde_roll['t'], rgbde_roll['x'], 'r-.')
# # ax1.plot(rgbde_roll['t'], rgbde_roll['y'], 'g-.')
# # ax1.plot(rgbde_roll['t'], rgbde_roll['z'], 'b-.')
# ax2.plot(ekom_roll['t'], ekom_roll['qx'], 'k', label='qx')
# ax2.plot(ekom_roll['t'], ekom_roll['qy'], 'y', label='qy')
# ax2.plot(ekom_roll['t'], ekom_roll['qz'], 'm', label='qz')
# ax2.plot(ekom_roll['t'], ekom_roll['qw'], 'c', label='qw')
# ax2.plot(gt_roll['t'], gt_roll['qx'], 'k--')
# ax2.plot(gt_roll['t'], gt_roll['qy'], 'y--')
# ax2.plot(gt_roll['t'], gt_roll['qz'], 'm--')
# ax2.plot(gt_roll['t'], gt_roll['qw'], 'c--')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qx'], 'k-.')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qy'], 'y-.')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qz'], 'm-.')
# # ax2.plot(rgbde_roll['t'], rgbde_roll['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

# fig5, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_pitch['t'], ekom_pitch['x'], 'r', label='x')
# ax1.plot(ekom_pitch['t'], ekom_pitch['y'], 'g', label='y')
# ax1.plot(ekom_pitch['t'], ekom_pitch['z'], 'b', label='z')
# ax1.plot(gt_pitch['t'], gt_pitch['x'], 'r--')
# ax1.plot(gt_pitch['t'], gt_pitch['y'], 'g--')
# ax1.plot(gt_pitch['t'], gt_pitch['z'], 'b--')
# # ax1.plot(rgbde_pitch['t'], rgbde_pitch['x'], 'r-.')
# # ax1.plot(rgbde_pitch['t'], rgbde_pitch['y'], 'g-.')
# # ax1.plot(rgbde_pitch['t'], rgbde_pitch['z'], 'b-.')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qx'], 'k', label='qx')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qy'], 'y', label='qy')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qz'], 'm', label='qz')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qw'], 'c', label='qw')
# ax2.plot(gt_pitch['t'], gt_pitch['qx'], 'k--')
# ax2.plot(gt_pitch['t'], gt_pitch['qy'], 'y--')
# ax2.plot(gt_pitch['t'], gt_pitch['qz'], 'm--')
# ax2.plot(gt_pitch['t'], gt_pitch['qw'], 'c--')
# # ax2.plot(rgbde_pitch['t'], rgbde_pitch['qx'], 'k-.')
# # ax2.plot(rgbde_pitch['t'], rgbde_pitch['qy'], 'y-.')
# # ax2.plot(rgbde_pitch['t'], rgbde_pitch['qz'], 'm-.')
# # ax2.plot(rgbde_pitch['t'], rgbde_pitch['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

# fig6, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_yaw['t'], ekom_yaw['x'], 'r', label='x')
# ax1.plot(ekom_yaw['t'], ekom_yaw['y'], 'g', label='y')
# ax1.plot(ekom_yaw['t'], ekom_yaw['z'], 'b', label='z')
# ax1.plot(gt_yaw['t'], gt_yaw['x'], 'r--')
# ax1.plot(gt_yaw['t'], gt_yaw['y'], 'g--')
# ax1.plot(gt_yaw['t'], gt_yaw['z'], 'b--')
# # ax1.plot(rgbde_yaw['t'], rgbde_yaw['x'], 'r-.')
# # ax1.plot(rgbde_yaw['t'], rgbde_yaw['y'], 'g-.')
# # ax1.plot(rgbde_yaw['t'], rgbde_yaw['z'], 'b-.')
# ax2.plot(ekom_yaw['t'], ekom_yaw['qx'], 'k', label='qx')
# ax2.plot(ekom_yaw['t'], ekom_yaw['qy'], 'y', label='qy')
# ax2.plot(ekom_yaw['t'], ekom_yaw['qz'], 'm', label='qz')
# ax2.plot(ekom_yaw['t'], ekom_yaw['qw'], 'c', label='qw')
# ax2.plot(gt_yaw['t'], gt_yaw['qx'], 'k--')
# ax2.plot(gt_yaw['t'], gt_yaw['qy'], 'y--')
# ax2.plot(gt_yaw['t'], gt_yaw['qz'], 'm--')
# ax2.plot(gt_yaw['t'], gt_yaw['qw'], 'c--')
# # ax2.plot(rgbde_yaw['t'], rgbde_yaw['qx'], 'k-.')
# # ax2.plot(rgbde_yaw['t'], rgbde_yaw['qy'], 'y-.')
# # ax2.plot(rgbde_yaw['t'], rgbde_yaw['qz'], 'm-.')
# # ax2.plot(rgbde_yaw['t'], rgbde_yaw['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

ekom_error_trans_x = computeEuclideanDistance(gt_trans_x['x'], gt_trans_x['y'], gt_trans_x['z'], ekom_trans_x_resampled_x, ekom_trans_x_resampled_y, ekom_trans_x_resampled_z)
ekom_error_trans_y = computeEuclideanDistance(gt_trans_y['x'], gt_trans_y['y'], gt_trans_y['z'], ekom_trans_y_resampled_x, ekom_trans_y_resampled_y, ekom_trans_y_resampled_z)
ekom_error_trans_z = computeEuclideanDistance(gt_trans_z['x'], gt_trans_z['y'], gt_trans_z['z'], ekom_trans_z_resampled_x, ekom_trans_z_resampled_y, ekom_trans_z_resampled_z)
ekom_error_roll = computeEuclideanDistance(gt_roll['x'], gt_roll['y'], gt_roll['z'], ekom_roll_resampled_x, ekom_roll_resampled_y, ekom_roll_resampled_z)
ekom_error_pitch = computeEuclideanDistance(gt_pitch['x'], gt_pitch['y'], gt_pitch['z'], ekom_pitch_resampled_x, ekom_pitch_resampled_y, ekom_pitch_resampled_z)
ekom_error_yaw = computeEuclideanDistance(gt_yaw['x'], gt_yaw['y'], gt_yaw['z'], ekom_yaw_resampled_x, ekom_yaw_resampled_y, ekom_yaw_resampled_z)

ekom_q_angle_trans_x = computeQuaternionError(ekom_trans_x_resampled_qx, ekom_trans_x_resampled_qy, ekom_trans_x_resampled_qz, ekom_trans_x_resampled_qw, gt_trans_x['qx'], gt_trans_x['qy'], gt_trans_x['qz'], gt_trans_x['qw'])
ekom_q_angle_trans_y = computeQuaternionError(ekom_trans_y_resampled_qx, ekom_trans_y_resampled_qy, ekom_trans_y_resampled_qz, ekom_trans_y_resampled_qw, gt_trans_y['qx'], gt_trans_y['qy'], gt_trans_y['qz'], gt_trans_y['qw'])
ekom_q_angle_trans_z = computeQuaternionError(ekom_trans_z_resampled_qx, ekom_trans_z_resampled_qy, ekom_trans_z_resampled_qz, ekom_trans_z_resampled_qw, gt_trans_z['qx'], gt_trans_z['qy'], gt_trans_z['qz'], gt_trans_z['qw'])
ekom_q_angle_roll = computeQuaternionError(ekom_roll_resampled_qx, ekom_roll_resampled_qy, ekom_roll_resampled_qz, ekom_roll_resampled_qw, gt_roll['qx'], gt_roll['qy'], gt_roll['qz'], gt_roll['qw'])
ekom_q_angle_pitch = computeQuaternionError(ekom_pitch_resampled_qx, ekom_pitch_resampled_qy, ekom_pitch_resampled_qz, ekom_pitch_resampled_qw, gt_pitch['qx'], gt_pitch['qy'], gt_pitch['qz'], gt_pitch['qw'])
ekom_q_angle_yaw = computeQuaternionError(ekom_yaw_resampled_qx, ekom_yaw_resampled_qy, ekom_yaw_resampled_qz, ekom_yaw_resampled_qw, gt_yaw['qx'], gt_yaw['qy'], gt_yaw['qz'], gt_yaw['qw'])

rgbde_error_trans_x = computeEuclideanDistance(gt_trans_x['x'], gt_trans_x['y'], gt_trans_x['z'], rgbde_trans_x_resampled_x, rgbde_trans_x_resampled_y, rgbde_trans_x_resampled_z)
rgbde_error_trans_y = computeEuclideanDistance(gt_trans_y['x'], gt_trans_y['y'], gt_trans_y['z'], rgbde_trans_y_resampled_x, rgbde_trans_y_resampled_y, rgbde_trans_y_resampled_z)
rgbde_error_trans_z = computeEuclideanDistance(gt_trans_z['x'], gt_trans_z['y'], gt_trans_z['z'], rgbde_trans_z_resampled_x, rgbde_trans_z_resampled_y, rgbde_trans_z_resampled_z)
rgbde_error_roll = computeEuclideanDistance(gt_roll['x'], gt_roll['y'], gt_roll['z'], rgbde_roll_resampled_x, rgbde_roll_resampled_y, rgbde_roll_resampled_z)
rgbde_error_pitch = computeEuclideanDistance(gt_pitch['x'], gt_pitch['y'], gt_pitch['z'], rgbde_pitch_resampled_x, rgbde_pitch_resampled_y, rgbde_pitch_resampled_z)
# rgbde_error_yaw = computeEuclideanDistance(gt_yaw['x'], gt_yaw['y'], gt_yaw['z'], rgbde_yaw_resampled_x, rgbde_yaw_resampled_y, rgbde_yaw_resampled_z)

rgbde_q_angle_trans_x = computeQuaternionError(rgbde_trans_x_resampled_qx, rgbde_trans_x_resampled_qy, rgbde_trans_x_resampled_qz, rgbde_trans_x_resampled_qw, gt_trans_x['qx'], gt_trans_x['qy'], gt_trans_x['qz'], gt_trans_x['qw'])
rgbde_q_angle_trans_y = computeQuaternionError(rgbde_trans_y_resampled_qx, rgbde_trans_y_resampled_qy, rgbde_trans_y_resampled_qz, rgbde_trans_y_resampled_qw, gt_trans_y['qx'], gt_trans_y['qy'], gt_trans_y['qz'], gt_trans_y['qw'])
rgbde_q_angle_trans_z = computeQuaternionError(rgbde_trans_z_resampled_qx, rgbde_trans_z_resampled_qy, rgbde_trans_z_resampled_qz, rgbde_trans_z_resampled_qw, gt_trans_z['qx'], gt_trans_z['qy'], gt_trans_z['qz'], gt_trans_z['qw'])
rgbde_q_angle_roll = computeQuaternionError(rgbde_roll_resampled_qx, rgbde_roll_resampled_qy, rgbde_roll_resampled_qz, rgbde_roll_resampled_qw, gt_roll['qx'], gt_roll['qy'], gt_roll['qz'], gt_roll['qw'])
rgbde_q_angle_pitch = computeQuaternionError(rgbde_pitch_resampled_qx, rgbde_pitch_resampled_qy, rgbde_pitch_resampled_qz, rgbde_pitch_resampled_qw, gt_pitch['qx'], gt_pitch['qy'], gt_pitch['qz'], gt_pitch['qw'])
# rgbde_q_angle_yaw = computeQuaternionError(rgbde_yaw_resampled_qx, rgbde_yaw_resampled_qy, rgbde_yaw_resampled_qz, rgbde_yaw_resampled_qw, gt_yaw['qx'], gt_yaw['qy'], gt_yaw['qz'], gt_yaw['qw'])

ekom_tr_datasets_position_errors = np.stack(ekom_error_trans_x, ekom_error_trans_y, ekom_error_trans_z)
ekom_rot_datasets_position_errors = np.stack(ekom_error_roll, ekom_error_pitch, ekom_error_yaw)
ekom_tr_datasets_angle_errors = np.stack(ekom_q_angle_trans_x, ekom_q_angle_trans_y, ekom_q_angle_trans_z)
ekom_rot_datasets_angle_errors = np.stack(ekom_q_angle_roll, ekom_q_angle_pitch, ekom_q_angle_yaw)

print(np.mean(ekom_error_trans_x*100), "cm")
print(np.mean(rgbde_error_trans_x*100), "cm")

print(np.mean(ekom_q_angle_trans_x), "rad")
print(np.mean(rgbde_q_angle_trans_x), "rad")

print(np.mean(ekom_error_trans_z*100), "cm")
print(np.mean(rgbde_error_trans_z*100), "cm")

X = ['Tr X', 'Tr Y', 'Tr Z', 'Roll', 'Pitch', 'Yaw']

X_axis = np.arange(len(X))
ekom_average_position_error = [np.mean(ekom_error_trans_x), np.mean(ekom_error_trans_y),np.mean(ekom_error_trans_z), np.mean(ekom_error_roll), np.mean(ekom_error_pitch), np.mean(ekom_error_yaw)]
rgbde_average_position_error = [np.mean(rgbde_error_trans_x), np.mean(rgbde_error_trans_y),np.mean(rgbde_error_trans_z), np.mean(rgbde_error_roll), np.mean(rgbde_error_pitch), 0.08]
  
ekom_std_position_error = [np.std(ekom_error_trans_x), np.std(ekom_error_trans_y),np.std(ekom_error_trans_z), np.std(ekom_error_roll), np.std(ekom_error_pitch), np.std(ekom_error_yaw)]
rgbde_std_position_error = [np.std(rgbde_error_trans_x), np.std(rgbde_error_trans_y),np.std(rgbde_error_trans_z), np.std(rgbde_error_roll), np.std(rgbde_error_pitch), 0.08]

plt.bar(X_axis - 0.2, ekom_average_position_error, 0.4, yerr=ekom_std_position_error,label = 'EKOM')
plt.bar(X_axis + 0.2, rgbde_average_position_error, 0.4, yerr=rgbde_std_position_error, label = 'RGBD-E')
  
plt.xticks(X_axis, X)
plt.xlabel("Algorithms")
plt.ylabel("Mean Position Error [m]")
# plt.title("Number of Students in each group")
plt.legend()
plt.show()

X = ['Tr X', 'Tr Y', 'Tr Z', 'Roll', 'Pitch', 'Yaw']

X_axis = np.arange(len(X))
ekom_average_angle_error = [np.mean(ekom_q_angle_trans_x), np.mean(ekom_q_angle_trans_y),np.mean(ekom_q_angle_trans_z), np.mean(ekom_q_angle_roll), np.mean(ekom_q_angle_pitch), np.mean(ekom_q_angle_yaw)]
rgbde_average_angle_error = [np.mean(rgbde_q_angle_trans_x), np.mean(rgbde_q_angle_trans_y),np.mean(rgbde_q_angle_trans_z), np.mean(rgbde_q_angle_roll), np.mean(rgbde_q_angle_pitch), 0.08]
  
ekom_std_angle_error = [np.std(ekom_q_angle_trans_x), np.std(ekom_q_angle_trans_y),np.std(ekom_q_angle_trans_z), np.std(ekom_q_angle_roll), np.std(ekom_q_angle_pitch), np.std(ekom_q_angle_yaw)]
rgbde_std_angle_error = [np.std(rgbde_q_angle_trans_x), np.std(rgbde_q_angle_trans_y),np.std(rgbde_q_angle_trans_z), np.std(rgbde_q_angle_roll), np.std(rgbde_q_angle_pitch), 0.08]

plt.bar(X_axis - 0.2, ekom_average_angle_error, 0.4, yerr=ekom_std_angle_error, label = 'EKOM', color='tab:green')
plt.bar(X_axis + 0.2, rgbde_average_angle_error, 0.4, yerr=rgbde_std_angle_error, label = 'RGBD-E', color='tab:red')
  
plt.xticks(X_axis, X)
plt.xlabel("Algorithms")
plt.ylabel("Mean Quaternion Error")
# plt.title("Number of Students in each group")
plt.legend()
plt.show()

X = ['Translations' ,'Rotations']

X_axis = np.arange(len(X))
ekom_average_pos_errors = [np.mean(ekom_tr_datasets_position_errors), np.mean(ekom_rot_datasets_position_errors)]
ekom_average_angle_errors = [np.mean(ekom_tr_datasets_angle_errors), np.mean(ekom_rot_datasets_angle_errors)]

plt.bar(X_axis - 0.2, ekom_average_pos_errors, 0.4, yerr=ekom_std_angle_error, label = 'EKOM', color='tab:green')
plt.bar(X_axis + 0.2, rgbde_average_angle_error, 0.4, yerr=rgbde_std_angle_error, label = 'RGBD-E', color='tab:red')
  
plt.xticks(X_axis, X)
plt.xlabel("Algorithms")
plt.ylabel("Mean Quaternion Error")
# plt.title("Number of Students in each group")
plt.legend()
plt.show()