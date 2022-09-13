from array import array
from operator import gt
import sys, os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import math 
from scipy.interpolate import interp1d
import quaternion as quat
from pyquaternion import Quaternion

color_x = '#CC1EEA'
color_y = '#06AE97'
color_z = '#FFA409'

color_ekom = 'tab:blue'
color_rgbde = 'tab:orange'

SMALL_SIZE = 18
MEDIUM_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def computeEuclideanDistance(x1, y1, z1, x2, y2, z2):
    list_dist = []
    for xp1,yp1,zp1,xp2,yp2,zp2 in zip(x1,y1,z1,x2,y2,z2):
        if np.isneginf(xp2) or np.isneginf(yp2) or np.isneginf(zp2) or np.isnan(xp2) or np.isnan(yp2) or np.isnan(zp2):
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
        quaternion_angle = np.arccos(np.clip(2*inner_product*inner_product-1, -1, 1))
        if np.isnan(quaternion_angle):
            continue
        else:
            list_q_error.append(quaternion_angle)

    array_q_error = np.array(list_q_error)
    return array_q_error 

def resampling_by_interpolate(time_samples, x_values, y_values):
    f_neareast = interp1d(x_values, y_values, kind='nearest', fill_value="extrapolate")
    resampled = f_neareast(time_samples)
    return resampled

def quaternion_to_euler_angle(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """

        roll_list = []
        pitch_list = []
        yaw_list = []

        degrees = 57.2958

        for qx,qy,qz,qw in zip(x,y,z,w):

            t0 = +2.0 * (qw * qx + qy * qz)
            t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
            roll_x = math.atan2(t0, t1)
        
            t2 = +2.0 * (qw * qy - qz * qx)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
        
            t3 = +2.0 * (qw * qz + qx * qy)
            t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
            yaw_z = math.atan2(t3, t4)

            roll_list.append(roll_x*degrees)
            pitch_list.append(pitch_y*degrees)  
            yaw_list.append(yaw_z*degrees)

            roll_array = np.array(roll_list)
            pitch_array = np.array(pitch_list)
            yaw_array = np.array(yaw_list)
     
        return roll_array, pitch_array, yaw_array # in degrees

def cleanEuler(angle, angle_type): 

    # angle = angle[~np.isnan(angle)]
    diff_arrays = angle[1:-1]-angle[0:-2]
    prev_x = 0
    th = 180
    filtered_angles_list = [] 
    diff_list=[]
    for idx, x in enumerate(angle):
        if idx == 0:
            if angle_type==2 and x < 0:
                x+=360 
            prev_x = x
        else:
            diff = abs(x - prev_x)
            diff_list.append(diff)
            if diff > th:
                x += 360
            else:
                if angle_type==2 and x<0:
                    x += 360
            prev_x = x
        filtered_angles_list.append(x)

    return(np.array(filtered_angles_list))
    
# ---------------------------------------------------------------------------  DRAGON  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/dragon/'
gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_x_1_m_s_2/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_y_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_translation_z_1_m_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_roll_4rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_pitch_4rad_s/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'dragon_yaw_4rad_s_2/ground_truth.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

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

ekom_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_roll = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
ekom_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'results/dragon/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

rgbde_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_translation_x_1_m_s_2_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_translation_y_1_m_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_translation_z_1_m_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_roll = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_roll_4rad_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_pitch_4rad_s_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])
rgbde_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'rgbd-e/dragon_yaw_2_tracked_pose_quaternion.csv'), delimiter=",", names=["x", "y", "z", "qw", "qx", "qy", "qz"])

rgbde_time_trans_x = (np.arange(0, 1/60*len(rgbde_trans_x['x']), 1/60))*10
rgbde_time_trans_y = (np.arange(0, 1/60*len(rgbde_trans_y['x']), 1/60))*10
rgbde_time_trans_z = (np.arange(0, 1/60*len(rgbde_trans_z['x']), 1/60))*10
rgbde_time_roll = (np.arange(0, 1/60*len(rgbde_roll['qx']), 1/60))*10
rgbde_time_pitch = (np.arange(0, 1/60*len(rgbde_pitch['qx']), 1/60))*10
rgbde_time_yaw = (np.arange(0, 1/60*len(rgbde_yaw['qx']), 1/60))*10

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
rgbde_yaw_resampled_x = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['x'])
rgbde_yaw_resampled_y = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['y'])
rgbde_yaw_resampled_z = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['z'])
rgbde_yaw_resampled_qx = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['qx'])
rgbde_yaw_resampled_qy = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['qy'])
rgbde_yaw_resampled_qz = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['qz'])
rgbde_yaw_resampled_qw = resampling_by_interpolate(gt_yaw['t'], rgbde_time_yaw, rgbde_yaw['qw'])

ekom_trans_x_alpha,ekom_trans_x_beta,ekom_trans_x_gamma = quaternion_to_euler_angle(ekom_trans_x['qw'], ekom_trans_x['qx'], ekom_trans_x['qy'], ekom_trans_x['qz'])
rgbde_trans_x_alpha,rgbde_trans_x_beta,rgbde_trans_x_gamma = quaternion_to_euler_angle(rgbde_trans_x['qw'], rgbde_trans_x['qx'], rgbde_trans_x['qy'], rgbde_trans_x['qz'])
gt_trans_x_alpha,gt_trans_x_beta,gt_trans_x_gamma = quaternion_to_euler_angle(gt_trans_x['qw'], gt_trans_x['qx'], gt_trans_x['qy'], gt_trans_x['qz'])

ekom_trans_x_alpha_cleaned = cleanEuler(ekom_trans_x_alpha,0)
ekom_trans_x_beta_cleaned = cleanEuler(ekom_trans_x_beta,1)
ekom_trans_x_gamma_cleaned = cleanEuler(ekom_trans_x_gamma,2)

rgbde_trans_x_alpha_cleaned = cleanEuler(rgbde_trans_x_alpha,0)
rgbde_trans_x_beta_cleaned = cleanEuler(rgbde_trans_x_beta,1)
rgbde_trans_x_gamma_cleaned = cleanEuler(rgbde_trans_x_gamma,2)

gt_trans_x_alpha_cleaned = cleanEuler(gt_trans_x_alpha,0)
gt_trans_x_beta_cleaned = cleanEuler(gt_trans_x_beta,1)
gt_trans_x_gamma_cleaned = cleanEuler(gt_trans_x_gamma,1)

ekom_trans_y_alpha,ekom_trans_y_beta,ekom_trans_y_gamma = quaternion_to_euler_angle(ekom_trans_y['qw'], ekom_trans_y['qx'], ekom_trans_y['qy'], ekom_trans_y['qz'])
rgbde_trans_y_alpha,rgbde_trans_y_beta,rgbde_trans_y_gamma = quaternion_to_euler_angle(rgbde_trans_y['qw'], rgbde_trans_y['qx'], rgbde_trans_y['qy'], rgbde_trans_y['qz'])
gt_trans_y_alpha,gt_trans_y_beta,gt_trans_y_gamma = quaternion_to_euler_angle(gt_trans_y['qw'], gt_trans_y['qx'], gt_trans_y['qy'], gt_trans_y['qz'])

ekom_trans_y_alpha_cleaned = cleanEuler(ekom_trans_y_alpha,0)
ekom_trans_y_beta_cleaned = cleanEuler(ekom_trans_y_beta,1)
ekom_trans_y_gamma_cleaned = cleanEuler(ekom_trans_y_gamma,2)

rgbde_trans_y_alpha_cleaned = cleanEuler(rgbde_trans_y_alpha,0)
rgbde_trans_y_beta_cleaned = cleanEuler(rgbde_trans_y_beta,1)
rgbde_trans_y_gamma_cleaned = cleanEuler(rgbde_trans_y_gamma,2)

gt_trans_y_alpha_cleaned = cleanEuler(gt_trans_y_alpha,0)
gt_trans_y_beta_cleaned = cleanEuler(gt_trans_y_beta,1)
gt_trans_y_gamma_cleaned = cleanEuler(gt_trans_y_gamma,2)

ekom_trans_z_alpha,ekom_trans_z_beta,ekom_trans_z_gamma = quaternion_to_euler_angle(ekom_trans_z['qw'], ekom_trans_z['qx'], ekom_trans_z['qy'], ekom_trans_z['qz'])
rgbde_trans_z_alpha,rgbde_trans_z_beta,rgbde_trans_z_gamma = quaternion_to_euler_angle(rgbde_trans_z['qw'], rgbde_trans_z['qx'], rgbde_trans_z['qy'], rgbde_trans_z['qz'])
gt_trans_z_alpha,gt_trans_z_beta,gt_trans_z_gamma = quaternion_to_euler_angle(gt_trans_z['qw'], gt_trans_z['qx'], gt_trans_z['qy'], gt_trans_z['qz'])

ekom_trans_z_alpha_cleaned = cleanEuler(ekom_trans_z_alpha,0)
ekom_trans_z_beta_cleaned = cleanEuler(ekom_trans_z_beta,1)
ekom_trans_z_gamma_cleaned = cleanEuler(ekom_trans_z_gamma,2)

rgbde_trans_z_alpha_cleaned = cleanEuler(rgbde_trans_z_alpha,0)
rgbde_trans_z_beta_cleaned = cleanEuler(rgbde_trans_z_beta,1)
rgbde_trans_z_gamma_cleaned = cleanEuler(rgbde_trans_z_gamma,2)

gt_trans_z_alpha_cleaned = cleanEuler(gt_trans_z_alpha,0)
gt_trans_z_beta_cleaned = cleanEuler(gt_trans_z_beta,1)
gt_trans_z_gamma_cleaned = cleanEuler(gt_trans_z_gamma,2)

ekom_roll_alpha,ekom_roll_beta,ekom_roll_gamma = quaternion_to_euler_angle(ekom_roll['qw'], ekom_roll['qx'], ekom_roll['qy'], ekom_roll['qz'])
rgbde_roll_alpha,rgbde_roll_beta,rgbde_roll_gamma = quaternion_to_euler_angle(rgbde_roll['qw'], rgbde_roll['qx'], rgbde_roll['qy'], rgbde_roll['qz'])
gt_roll_alpha,gt_roll_beta,gt_roll_gamma = quaternion_to_euler_angle(gt_roll['qw'], gt_roll['qx'], gt_roll['qy'], gt_roll['qz'])

ekom_roll_alpha_cleaned = cleanEuler(ekom_roll_alpha,0)
ekom_roll_beta_cleaned = cleanEuler(ekom_roll_beta,1)
ekom_roll_gamma_cleaned = cleanEuler(ekom_roll_gamma,2)

rgbde_roll_alpha_cleaned = cleanEuler(rgbde_roll_alpha,0)
rgbde_roll_beta_cleaned = cleanEuler(rgbde_roll_beta,1)
rgbde_roll_gamma_cleaned = cleanEuler(rgbde_roll_gamma,2)

gt_roll_alpha_cleaned = cleanEuler(gt_roll_alpha,0)
gt_roll_beta_cleaned = cleanEuler(gt_roll_beta,1)
gt_roll_gamma_cleaned = cleanEuler(gt_roll_gamma,2)

ekom_pitch_alpha,ekom_pitch_beta,ekom_pitch_gamma = quaternion_to_euler_angle(ekom_pitch['qw'], ekom_pitch['qx'], ekom_pitch['qy'], ekom_pitch['qz'])
rgbde_pitch_alpha,rgbde_pitch_beta,rgbde_pitch_gamma = quaternion_to_euler_angle(rgbde_pitch['qw'], rgbde_pitch['qx'], rgbde_pitch['qy'], rgbde_pitch['qz'])
gt_pitch_alpha,gt_pitch_beta,gt_pitch_gamma = quaternion_to_euler_angle(gt_pitch['qw'], gt_pitch['qx'], gt_pitch['qy'], gt_pitch['qz'])

ekom_pitch_alpha_cleaned = cleanEuler(ekom_pitch_alpha,0)
ekom_pitch_beta_cleaned = cleanEuler(ekom_pitch_beta,1)
ekom_pitch_gamma_cleaned = cleanEuler(ekom_pitch_gamma,2)

rgbde_pitch_alpha_cleaned = cleanEuler(rgbde_pitch_alpha,0)
rgbde_pitch_beta_cleaned = cleanEuler(rgbde_pitch_beta,1)
rgbde_pitch_gamma_cleaned = cleanEuler(rgbde_pitch_gamma,2)

gt_pitch_alpha_cleaned = cleanEuler(gt_pitch_alpha,0)
gt_pitch_beta_cleaned = cleanEuler(gt_pitch_beta,1)
gt_pitch_gamma_cleaned = cleanEuler(gt_pitch_gamma,2)

ekom_yaw_alpha,ekom_yaw_beta,ekom_yaw_gamma = quaternion_to_euler_angle(ekom_yaw['qw'], ekom_yaw['qx'], ekom_yaw['qy'], ekom_yaw['qz'])
rgbde_yaw_alpha,rgbde_yaw_beta,rgbde_yaw_gamma = quaternion_to_euler_angle(rgbde_yaw['qw'], rgbde_yaw['qx'], rgbde_yaw['qy'], rgbde_yaw['qz'])
gt_yaw_alpha,gt_yaw_beta,gt_yaw_gamma = quaternion_to_euler_angle(gt_yaw['qw'], gt_yaw['qx'], gt_yaw['qy'], gt_yaw['qz'])

ekom_yaw_alpha_cleaned = cleanEuler(ekom_yaw_alpha,0)
ekom_yaw_beta_cleaned = cleanEuler(ekom_yaw_beta,1)
ekom_yaw_gamma_cleaned = cleanEuler(ekom_yaw_gamma,2)

rgbde_yaw_alpha_cleaned = cleanEuler(rgbde_yaw_alpha,0)
rgbde_yaw_beta_cleaned = cleanEuler(rgbde_yaw_beta,1)
rgbde_yaw_gamma_cleaned = cleanEuler(rgbde_yaw_gamma,2)

gt_yaw_alpha_cleaned = cleanEuler(gt_yaw_alpha,0)
gt_yaw_beta_cleaned = cleanEuler(gt_yaw_beta,1)
gt_yaw_gamma_cleaned = cleanEuler(gt_yaw_gamma,2)

fig_summary, axs = plt.subplots(4,3)
fig_summary.set_size_inches(18, 12)
axs[0,0].plot(ekom_trans_x['t'], ekom_trans_x['x'], color=color_x, label='x')
axs[0,0].plot(ekom_trans_x['t'], ekom_trans_x['y'], color=color_y, label='y')
axs[0,0].plot(ekom_trans_x['t'], ekom_trans_x['z'], color=color_z, label='z')
axs[0,0].plot(gt_trans_x['t'], gt_trans_x['x'], color=color_x, ls='--')
axs[0,0].plot(gt_trans_x['t'], gt_trans_x['y'], color=color_y, ls='--')
axs[0,0].plot(gt_trans_x['t'], gt_trans_x['z'], color=color_z, ls='--')
axs[0,0].plot(rgbde_time_trans_x, rgbde_trans_x['x'], color=color_x, ls='-.')
axs[0,0].plot(rgbde_time_trans_x, rgbde_trans_x['y'], color=color_y, ls='-.')
axs[0,0].plot(rgbde_time_trans_x, rgbde_trans_x['z'], color=color_z, ls='-.')
axs[0,1].plot(ekom_trans_y['t'], ekom_trans_y['x'], color=color_x, label='x')
axs[0,1].plot(ekom_trans_y['t'], ekom_trans_y['y'], color=color_y, label='y')
axs[0,1].plot(ekom_trans_y['t'], ekom_trans_y['z'], color=color_z, label='z')
axs[0,1].plot(gt_trans_y['t'], gt_trans_y['x'], color=color_x, ls='--')
axs[0,1].plot(gt_trans_y['t'], gt_trans_y['y'], color=color_y, ls='--')
axs[0,1].plot(gt_trans_y['t'], gt_trans_y['z'], color=color_z, ls='--')
axs[0,1].plot(rgbde_time_trans_y, rgbde_trans_y['x'], color=color_x, ls='-.')
axs[0,1].plot(rgbde_time_trans_y, rgbde_trans_y['y'], color=color_y, ls='-.')
axs[0,1].plot(rgbde_time_trans_y, rgbde_trans_y['z'], color=color_z, ls='-.')
axs[0,2].plot(ekom_trans_z['t'], ekom_trans_z['x'], color=color_x, label='x')
axs[0,2].plot(ekom_trans_z['t'], ekom_trans_z['y'], color=color_y, label='y')
axs[0,2].plot(ekom_trans_z['t'], ekom_trans_z['z'], color=color_z, label='z')
axs[0,2].plot(gt_trans_z['t'], gt_trans_z['x'], color=color_x, ls='--')
axs[0,2].plot(gt_trans_z['t'], gt_trans_z['y'], color=color_y, ls='--')
axs[0,2].plot(gt_trans_z['t'], gt_trans_z['z'], color=color_z, ls='--')
axs[0,2].plot(rgbde_time_trans_z, rgbde_trans_z['x'], color=color_x, ls='-.')
axs[0,2].plot(rgbde_time_trans_z, rgbde_trans_z['y'], color=color_y, ls='-.')
axs[0,2].plot(rgbde_time_trans_z, rgbde_trans_z['z'], color=color_z, ls='-.')
axs[2,0].plot(ekom_roll['t'], ekom_roll['x'], color=color_x, label='x')
axs[2,0].plot(ekom_roll['t'], ekom_roll['y'], color=color_y, label='y')
axs[2,0].plot(ekom_roll['t'], ekom_roll['z'], color=color_z, label='z')
axs[2,0].plot(gt_roll['t'], gt_roll['x'], color=color_x, ls='--')
axs[2,0].plot(gt_roll['t'], gt_roll['y'], color=color_y, ls='--')
axs[2,0].plot(gt_roll['t'], gt_roll['z'], color=color_z, ls='--')
axs[2,0].plot(rgbde_time_roll, rgbde_roll['x'], color=color_x, ls='-.')
axs[2,0].plot(rgbde_time_roll, rgbde_roll['y'], color=color_y, ls='-.')
axs[2,0].plot(rgbde_time_roll, rgbde_roll['z'], color=color_z, ls='-.')
axs[2,1].plot(ekom_pitch['t'], ekom_pitch['x'], color=color_x, label='x')
axs[2,1].plot(ekom_pitch['t'], ekom_pitch['y'], color=color_y, label='y')
axs[2,1].plot(ekom_pitch['t'], ekom_pitch['z'], color=color_z, label='z')
axs[2,1].plot(gt_pitch['t'], gt_pitch['x'], color=color_x, ls='--')
axs[2,1].plot(gt_pitch['t'], gt_pitch['y'], color=color_y, ls='--')
axs[2,1].plot(gt_pitch['t'], gt_pitch['z'], color=color_z, ls='--')
axs[2,1].plot(rgbde_time_pitch, rgbde_pitch['x'], color=color_x, ls='-.')
axs[2,1].plot(rgbde_time_pitch, rgbde_pitch['y'], color=color_y, ls='-.')
axs[2,1].plot(rgbde_time_pitch, rgbde_pitch['z'], color=color_z, ls='-.')
axs[2,2].plot(ekom_yaw['t'], ekom_yaw['x'], color=color_x, label='x')
axs[2,2].plot(ekom_yaw['t'], ekom_yaw['y'], color=color_y, label='y')
axs[2,2].plot(ekom_yaw['t'], ekom_yaw['z'], color=color_z, label='z')
axs[2,2].plot(gt_yaw['t'], gt_yaw['x'], color=color_x, ls='--')
axs[2,2].plot(gt_yaw['t'], gt_yaw['y'], color=color_y, ls='--')
axs[2,2].plot(gt_yaw['t'], gt_yaw['z'], color=color_z, ls='--')
axs[2,2].plot(rgbde_time_yaw, rgbde_yaw['x'], color=color_x, ls='-.')
axs[2,2].plot(rgbde_time_yaw, rgbde_yaw['y'], color=color_y, ls='-.')
axs[2,2].plot(rgbde_time_yaw, rgbde_yaw['z'], color=color_z, ls='-.')
axs[1,0].plot(ekom_trans_x['t'], ekom_trans_x_alpha_cleaned, color=color_x, label='qx')
axs[1,0].plot(ekom_trans_x['t'], ekom_trans_x_beta_cleaned, color=color_y, label='qy')
axs[1,0].plot(ekom_trans_x['t'], ekom_trans_x_gamma_cleaned, color=color_z, label='qz')
axs[1,0].plot(gt_trans_x['t'], gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
axs[1,0].plot(gt_trans_x['t'], gt_trans_x_beta_cleaned, color=color_y, ls = '--')
axs[1,0].plot(gt_trans_x['t'], gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
axs[1,0].plot(rgbde_time_trans_x, rgbde_trans_x_alpha_cleaned, color=color_x, ls = '-.')
axs[1,0].plot(rgbde_time_trans_x, rgbde_trans_x_beta_cleaned, color=color_y, ls = '-.')
axs[1,0].plot(rgbde_time_trans_x, rgbde_trans_x_gamma_cleaned, color=color_z, ls = '-.')
axs[1,1].plot(ekom_trans_y['t'], ekom_trans_y_alpha_cleaned, color=color_x, label='qx')
axs[1,1].plot(ekom_trans_y['t'], ekom_trans_y_beta_cleaned, color=color_y, label='qy')
axs[1,1].plot(ekom_trans_y['t'], ekom_trans_y_gamma_cleaned, color=color_z, label='qz')
axs[1,1].plot(gt_trans_y['t'], gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
axs[1,1].plot(gt_trans_y['t'], gt_trans_y_beta_cleaned, color=color_y, ls = '--')
axs[1,1].plot(gt_trans_y['t'], gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
axs[1,1].plot(rgbde_time_trans_y, rgbde_trans_y_alpha_cleaned, color=color_x, ls = '-.')
axs[1,1].plot(rgbde_time_trans_y, rgbde_trans_y_beta_cleaned, color=color_y, ls = '-.')
axs[1,1].plot(rgbde_time_trans_y, rgbde_trans_y_gamma_cleaned, color=color_z, ls = '-.')
axs[1,2].plot(ekom_trans_z['t'], ekom_trans_z_alpha_cleaned, color=color_x, label='qx')
axs[1,2].plot(ekom_trans_z['t'], ekom_trans_z_beta_cleaned, color=color_y, label='qy')
axs[1,2].plot(ekom_trans_z['t'], ekom_trans_z_gamma_cleaned, color=color_z, label='qz')
axs[1,2].plot(gt_trans_z['t'], gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
axs[1,2].plot(gt_trans_z['t'], gt_trans_z_beta_cleaned, color=color_y, ls = '--')
axs[1,2].plot(gt_trans_z['t'], gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
axs[1,2].plot(rgbde_time_trans_z, rgbde_trans_z_alpha_cleaned, color=color_x, ls = '-.')
axs[1,2].plot(rgbde_time_trans_z, rgbde_trans_z_beta_cleaned, color=color_y, ls = '-.')
axs[1,2].plot(rgbde_time_trans_z, rgbde_trans_z_gamma_cleaned, color=color_z, ls = '-.')
axs[3,0].plot(ekom_roll['t'], ekom_roll_alpha_cleaned, color=color_x, label='qx')
axs[3,0].plot(ekom_roll['t'], ekom_roll_beta_cleaned, color=color_y, label='qy')
axs[3,0].plot(ekom_roll['t'], ekom_roll_gamma_cleaned, color=color_z, label='qz')
axs[3,0].plot(gt_roll['t'], gt_roll_alpha_cleaned, color=color_x, ls = '--')
axs[3,0].plot(gt_roll['t'], gt_roll_beta_cleaned, color=color_y, ls = '--')
axs[3,0].plot(gt_roll['t'], gt_roll_gamma_cleaned, color=color_z, ls = '--')
axs[3,0].plot(rgbde_time_roll, rgbde_roll_alpha_cleaned, color=color_x, ls = '-.')
axs[3,0].plot(rgbde_time_roll, rgbde_roll_beta_cleaned, color=color_y, ls = '-.')
axs[3,0].plot(rgbde_time_roll, rgbde_roll_gamma_cleaned, color=color_z, ls = '-.')
axs[3,1].plot(ekom_pitch['t'], ekom_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - EKOM")
axs[3,1].plot(ekom_pitch['t'], ekom_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - EKOM")
axs[3,1].plot(ekom_pitch['t'], ekom_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - EKOM")
axs[3,1].plot(gt_pitch['t'], gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - GT")
axs[3,1].plot(gt_pitch['t'], gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - GT")
axs[3,1].plot(gt_pitch['t'], gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - GT")
axs[3,1].plot(rgbde_time_pitch, rgbde_pitch_alpha_cleaned, color=color_x, ls = '-.', label="x / "+r'$\alpha$'+" (pitch)"+ " - RGB-D-E")
axs[3,1].plot(rgbde_time_pitch, rgbde_pitch_beta_cleaned, color=color_y, ls = '-.',  label="y / "+r'$\beta$'+" (yaw) - RGB-D-E")
axs[3,1].plot(rgbde_time_pitch, rgbde_pitch_gamma_cleaned, color=color_z, ls = '-.', label="z / "+r'$\gamma$'+" (roll) - RGB-D-E")
axs[3,2].plot(ekom_yaw['t'], ekom_yaw_alpha_cleaned, color=color_x, label='x or roll')
axs[3,2].plot(ekom_yaw['t'], ekom_yaw_beta_cleaned, color=color_y, label='y or pitch')
axs[3,2].plot(ekom_yaw['t'], ekom_yaw_gamma_cleaned, color=color_z, label='z or yaw')
axs[3,2].plot(gt_yaw['t'], gt_yaw_alpha_cleaned, color=color_x, ls = '--')
axs[3,2].plot(gt_yaw['t'], gt_yaw_beta_cleaned, color=color_y, ls = '--')
axs[3,2].plot(gt_yaw['t'], gt_yaw_gamma_cleaned, color=color_z, ls = '--')
axs[3,2].plot(rgbde_time_yaw, rgbde_yaw_alpha_cleaned, color=color_x, ls = '-.')
axs[3,2].plot(rgbde_time_yaw, rgbde_yaw_beta_cleaned, color=color_y, ls = '-.')
axs[3,2].plot(rgbde_time_yaw, rgbde_yaw_gamma_cleaned, color=color_z, ls = '-.')
for i in range(0, 4):
    for j in range(0, 3):
        axs[i,j].set_xlim([-2, 50])
axs[0,0].set_ylim(-0.3,  1)
axs[0,1].set_ylim(-0.3,  1)
axs[0,2].set_ylim(-0.3,  1)
axs[2,0].set_ylim(-0.1,  0.9)
axs[2,1].set_ylim(-0.1,  0.9)
axs[2,2].set_ylim(-0.1,  0.9)
axs[1,0].set_ylim(-120,  200)
axs[1,1].set_ylim(-120,  200)
axs[1,2].set_ylim(-120,  200)
axs[3,0].set_ylim(-200,  300)
axs[3,1].set_ylim(-200,  300)
axs[3,2].set_ylim(-200,  300)
axs[0,0].set_xticks([])
axs[1,0].set_xticks([])
axs[2,0].set_xticks([])
axs[0,1].set_xticks([])
axs[1,1].set_xticks([])
axs[2,1].set_xticks([])
axs[0,2].set_xticks([])
axs[1,2].set_xticks([])
axs[2,2].set_xticks([])
axs[0,0].set_xticklabels([])
axs[1,0].set_xticklabels([])
axs[2,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_xticklabels([])
axs[2,1].set_xticklabels([])
axs[0,2].set_xticklabels([])
axs[1,2].set_xticklabels([])
axs[2,2].set_xticklabels([])
axs[0,1].set_yticklabels([])
axs[0,2].set_yticklabels([])
axs[1,1].set_yticklabels([])
axs[1,2].set_yticklabels([])
axs[2,1].set_yticklabels([])
axs[2,2].set_yticklabels([])
axs[3,1].set_yticklabels([])
axs[3,2].set_yticklabels([])
axs[0,0].set(ylabel='Position [m]')
axs[1,0].set(ylabel='Euler angles [deg]')
axs[2,0].set(ylabel='Position [m]')
axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
axs[3,1].set(xlabel='Time [s]')
axs[3,2].set(xlabel='Time [s]')
axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=3)
fig_summary.align_ylabels(axs[:, 0])
fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
plt.show()

# fig1, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_trans_x['t'], ekom_trans_x['x'], color=color_x, label='x')
# ax1.plot(ekom_trans_x['t'], ekom_trans_x['y'], color=color_y, label='y')
# ax1.plot(ekom_trans_x['t'], ekom_trans_x['z'], color=color_z, label='z')
# ax1.plot(gt_trans_x['t'], gt_trans_x['x'], color=color_x, ls='--')
# ax1.plot(gt_trans_x['t'], gt_trans_x['y'], color=color_y, ls='--')
# ax1.plot(gt_trans_x['t'], gt_trans_x['z'], color=color_z, ls='--')
# ax1.plot(rgbde_time_trans_x, rgbde_trans_x['x'], color=color_x, ls='-.')
# ax1.plot(rgbde_time_trans_x, rgbde_trans_x['y'], color=color_y, ls='-.')
# ax1.plot(rgbde_time_trans_x, rgbde_trans_x['z'], color=color_z, ls='-.')
# ax2.plot(ekom_trans_x['t'], ekom_trans_x_alpha, color=color_x, label='roll')
# ax2.plot(ekom_trans_x['t'], ekom_trans_x_beta, color=color_y, label='pitch')
# ax2.plot(ekom_trans_x['t'], ekom_trans_x_gamma, color=color_z, label='yaw')
# ax2.plot(rgbde_time_trans_x, rgbde_trans_x_alpha, color=color_x, ls='-.')
# ax2.plot(rgbde_time_trans_x, rgbde_trans_x_beta, color=color_y, ls='-.')
# ax2.plot(rgbde_time_trans_x, rgbde_trans_x_gamma, color=color_z, ls='-.')
# ax2.plot(gt_trans_x['t'], gt_trans_x_alpha, color=color_x, ls='--')
# ax2.plot(gt_trans_x['t'], gt_trans_x_beta, color=color_y, ls='--')
# ax2.plot(gt_trans_x['t'], gt_trans_x_gamma, color=color_z, ls='--')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()

# fig2, (ax1, ax2) = plt.subplots(2)
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['x'], color=color_x, label='x')
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['y'], color=color_y, label='y')
# ax1.plot(ekom_trans_y['t'], ekom_trans_y['z'], color=color_z, label='z')
# ax1.plot(gt_trans_y['t'], gt_trans_y['x'], color=color_x, ls='--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['y'], color=color_y, ls='--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['z'], color=color_z, ls='--')
# ax1.plot(rgbde_time_trans_y, rgbde_trans_y['x'], color=color_x, ls='-.')
# ax1.plot(rgbde_time_trans_y, rgbde_trans_y['y'], color=color_y, ls='-.')
# ax1.plot(rgbde_time_trans_y, rgbde_trans_y['z'], color=color_z, ls='-.')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y_alpha, color=color_x, label='roll')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y_beta, color=color_y, label='pitch')
# ax2.plot(ekom_trans_y['t'], ekom_trans_y_gamma, color=color_z, label='yaw')
# ax2.plot(rgbde_time_trans_y, rgbde_trans_y_alpha, color=color_x, ls='-.')
# ax2.plot(rgbde_time_trans_y, rgbde_trans_y_beta, color=color_y, ls='-.')
# ax2.plot(rgbde_time_trans_y, rgbde_trans_y_gamma, color=color_z, ls='-.')
# ax2.plot(gt_trans_y['t'], gt_trans_y_alpha, color=color_x, ls='--')
# ax2.plot(gt_trans_y['t'], gt_trans_y_beta, color=color_y, ls='--')
# ax2.plot(gt_trans_y['t'], gt_trans_y_gamma, color=color_z, ls='--')
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
# ax1.plot(rgbde_time_trans_z, rgbde_trans_z['x'], 'r-.')
# ax1.plot(rgbde_time_trans_z, rgbde_trans_z['y'], 'g-.')
# ax1.plot(rgbde_time_trans_z, rgbde_trans_z['z'], 'b-.')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qx'], 'k', label='qx')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qy'], 'y', label='qy')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qz'], 'm', label='qz')
# ax2.plot(ekom_trans_z['t'], ekom_trans_z['qw'], 'c', label='qw')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qx'], 'k--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qy'], 'y--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qz'], 'm--')
# ax2.plot(gt_trans_z['t'], gt_trans_z['qw'], 'c--')
# ax2.plot(rgbde_time_trans_z, rgbde_trans_z['qx'], 'k-.')
# ax2.plot(rgbde_time_trans_z, rgbde_trans_z['qy'], 'y-.')
# ax2.plot(rgbde_time_trans_z, rgbde_trans_z['qz'], 'm-.')
# ax2.plot(rgbde_time_trans_z, rgbde_trans_z['qw'], 'c-.')
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
# ax1.plot(rgbde_time_roll, rgbde_roll['x'], 'r-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['y'], 'g-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['z'], 'b-.')
# ax2.plot(ekom_roll['t'], -ekom_roll['qx'], 'k', label='qx')
# ax2.plot(ekom_roll['t'], -ekom_roll['qy'], 'y', label='qy')
# ax2.plot(ekom_roll['t'], -ekom_roll['qz'], 'm', label='qz')
# ax2.plot(ekom_roll['t'], -ekom_roll['qw'], 'c', label='qw')
# ax2.plot(gt_roll['t'], gt_roll['qx'], 'k--')
# ax2.plot(gt_roll['t'], gt_roll['qy'], 'y--')
# ax2.plot(gt_roll['t'], gt_roll['qz'], 'm--')
# ax2.plot(gt_roll['t'], gt_roll['qw'], 'c--')
# ax2.plot(rgbde_time_roll, -rgbde_roll['qx'], 'k-.')
# ax2.plot(rgbde_time_roll, rgbde_roll['qy'], 'y-.')
# ax2.plot(rgbde_time_roll, rgbde_roll['qz'], 'm-.')
# ax2.plot(rgbde_time_roll, rgbde_roll['qw'], 'c-.')
# ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
# ax1.set_xticks([])
# ax1.set(ylabel='Position[m]')
# ax2.set(xlabel='Time [s]', ylabel='Quaternions')
# plt.show()


# fig4bis, (ax1, ax2) = plt.subplots(2)
# # ax1.plot(ekom_roll['t'], ekom_roll['x'], 'r', label='x')
# # ax1.plot(ekom_roll['t'], ekom_roll['y'], 'g', label='y')
# # ax1.plot(ekom_roll['t'], ekom_roll['z'], 'b', label='z')
# # ax1.plot(gt_roll['t'], gt_roll['x'], 'r--')
# # ax1.plot(gt_roll['t'], gt_roll['y'], 'g--')
# # ax1.plot(gt_roll['t'], gt_roll['z'], 'b--')
# ax1.plot(rgbde_time_roll, rgbde_roll['x'], 'r-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['y'], 'g-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['z'], 'b-.')
# # ax2.plot(ekom_roll['t'], -ekom_roll['qx'], 'k', label='qx')
# # ax2.plot(ekom_roll['t'], -ekom_roll['qy'], 'y', label='qy')
# # ax2.plot(ekom_roll['t'], -ekom_roll['qz'], 'm', label='qz')
# # ax2.plot(ekom_roll['t'], -ekom_roll['qw'], 'c', label='qw')
# # ax2.plot(gt_roll['t'], gt_roll['qx'], 'k--')
# # ax2.plot(gt_roll['t'], gt_roll['qy'], 'y--')
# # ax2.plot(gt_roll['t'], gt_roll['qz'], 'm--')
# # ax2.plot(gt_roll['t'], gt_roll['qw'], 'c--')
# ax2.plot(rgbde_time_roll, -rgbde_roll['qx'], 'k-.')
# ax2.plot(rgbde_time_roll, rgbde_roll['qy'], 'y-.')
# ax2.plot(rgbde_time_roll, rgbde_roll['qz'], 'm-.')
# ax2.plot(rgbde_time_roll, rgbde_roll['qw'], 'c-.')
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
# ax1.plot(rgbde_time_pitch, rgbde_pitch['x'], 'r-.')
# ax1.plot(rgbde_time_pitch, rgbde_pitch['y'], 'g-.')
# ax1.plot(rgbde_time_pitch, rgbde_pitch['z'], 'b-.')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qx'], 'k', label='qx')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qy'], 'y', label='qy')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qz'], 'm', label='qz')
# ax2.plot(ekom_pitch['t'], ekom_pitch['qw'], 'c', label='qw')
# ax2.plot(gt_pitch['t'], gt_pitch['qx'], 'k--')
# ax2.plot(gt_pitch['t'], gt_pitch['qy'], 'y--')
# ax2.plot(gt_pitch['t'], gt_pitch['qz'], 'm--')
# ax2.plot(gt_pitch['t'], gt_pitch['qw'], 'c--')
# ax2.plot(rgbde_time_pitch, rgbde_pitch['qx'], 'k-.')
# ax2.plot(rgbde_time_pitch, rgbde_pitch['qy'], 'y-.')
# ax2.plot(rgbde_time_pitch, rgbde_pitch['qz'], 'm-.')
# ax2.plot(rgbde_time_pitch, rgbde_pitch['qw'], 'c-.')
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
# # ax1.plot(rgbde_time_yaw, rgbde_yaw['x'], 'r-.')
# # ax1.plot(rgbde_time_yaw, rgbde_yaw['y'], 'g-.')
# # ax1.plot(rgbde_time_yaw, rgbde_yaw['z'], 'b-.')
# ax2.plot(ekom_yaw['t'], -ekom_yaw['qx'], 'k', label='qx')
# ax2.plot(ekom_yaw['t'], -ekom_yaw['qy'], 'y', label='qy')
# ax2.plot(ekom_yaw['t'], -ekom_yaw['qz'], 'm', label='qz')
# ax2.plot(ekom_yaw['t'], -ekom_yaw['qw'], 'c', label='qw')
# ax2.plot(gt_yaw['t'], gt_yaw['qx'], 'k--')
# ax2.plot(gt_yaw['t'], gt_yaw['qy'], 'y--')
# ax2.plot(gt_yaw['t'], gt_yaw['qz'], 'm--')
# ax2.plot(gt_yaw['t'], gt_yaw['qw'], 'c--')
# # ax2.plot(rgbde_time_yaw, rgbde_yaw['qx'], 'k-.')
# # ax2.plot(rgbde_time_yaw, rgbde_yaw['qy'], 'y-.')
# # ax2.plot(rgbde_time_yaw, rgbde_yaw['qz'], 'm-.')
# # ax2.plot(rgbde_time_yaw, rgbde_yaw['qw'], 'c-.')
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
rgbde_error_yaw = computeEuclideanDistance(gt_yaw['x'], gt_yaw['y'], gt_yaw['z'], rgbde_yaw_resampled_x, rgbde_yaw_resampled_y, rgbde_yaw_resampled_z)

rgbde_q_angle_trans_x = computeQuaternionError(rgbde_trans_x_resampled_qx, rgbde_trans_x_resampled_qy, rgbde_trans_x_resampled_qz, rgbde_trans_x_resampled_qw, gt_trans_x['qx'], gt_trans_x['qy'], gt_trans_x['qz'], gt_trans_x['qw'])
rgbde_q_angle_trans_y = computeQuaternionError(rgbde_trans_y_resampled_qx, rgbde_trans_y_resampled_qy, rgbde_trans_y_resampled_qz, rgbde_trans_y_resampled_qw, gt_trans_y['qx'], gt_trans_y['qy'], gt_trans_y['qz'], gt_trans_y['qw'])
rgbde_q_angle_trans_z = computeQuaternionError(rgbde_trans_z_resampled_qx, rgbde_trans_z_resampled_qy, rgbde_trans_z_resampled_qz, rgbde_trans_z_resampled_qw, gt_trans_z['qx'], gt_trans_z['qy'], gt_trans_z['qz'], gt_trans_z['qw'])
rgbde_q_angle_roll = computeQuaternionError(rgbde_roll_resampled_qx, rgbde_roll_resampled_qy, rgbde_roll_resampled_qz, rgbde_roll_resampled_qw, gt_roll['qx'], gt_roll['qy'], gt_roll['qz'], gt_roll['qw'])
rgbde_q_angle_pitch = computeQuaternionError(rgbde_pitch_resampled_qx, rgbde_pitch_resampled_qy, rgbde_pitch_resampled_qz, rgbde_pitch_resampled_qw, gt_pitch['qx'], gt_pitch['qy'], gt_pitch['qz'], gt_pitch['qw'])
rgbde_q_angle_yaw = computeQuaternionError(rgbde_yaw_resampled_qx, rgbde_yaw_resampled_qy, rgbde_yaw_resampled_qz, rgbde_yaw_resampled_qw, gt_yaw['qx'], gt_yaw['qy'], gt_yaw['qz'], gt_yaw['qw'])

ekom_tr_datasets_position_errors = np.concatenate((ekom_error_trans_x, ekom_error_trans_y, ekom_error_trans_z))
ekom_rot_datasets_position_errors = np.concatenate((ekom_error_roll, ekom_error_pitch, ekom_error_yaw))
ekom_tr_datasets_angle_errors = np.concatenate((ekom_q_angle_trans_x, ekom_q_angle_trans_y, ekom_q_angle_trans_z))
ekom_rot_datasets_angle_errors = np.concatenate((ekom_q_angle_roll, ekom_q_angle_pitch, ekom_q_angle_yaw))

rgbde_tr_datasets_position_errors = np.concatenate((rgbde_error_trans_x, rgbde_error_trans_y, rgbde_error_trans_z))
rgbde_rot_datasets_position_errors = np.concatenate((rgbde_error_roll, rgbde_error_pitch))
rgbde_tr_datasets_angle_errors = np.concatenate((rgbde_q_angle_trans_x, rgbde_q_angle_trans_y, rgbde_q_angle_trans_z))
rgbde_rot_datasets_angle_errors = np.concatenate((rgbde_q_angle_roll, rgbde_q_angle_pitch))

# print(np.mean(ekom_error_trans_x*100), "cm")
# print(np.mean(rgbde_error_trans_x*100), "cm")

# print(np.mean(ekom_q_angle_trans_x), "rad")
# print(np.mean(rgbde_q_angle_trans_x), "rad")

# print(np.mean(ekom_error_trans_z*100), "cm")
# print(np.mean(rgbde_error_trans_z*100), "cm")

# X = ['Tr X', 'Tr Y', 'Tr Z', 'Roll', 'Pitch', 'Yaw']

# X_axis = np.arange(len(X))
# ekom_average_position_error = [np.mean(ekom_error_trans_x), np.mean(ekom_error_trans_y),np.mean(ekom_error_trans_z), np.mean(ekom_error_roll), np.mean(ekom_error_pitch), np.mean(ekom_error_yaw)]
# rgbde_average_position_error = [np.mean(rgbde_error_trans_x), np.mean(rgbde_error_trans_y),np.mean(rgbde_error_trans_z), np.mean(rgbde_error_roll), np.mean(rgbde_error_pitch), np.mean(rgbde_error_yaw)]
  
# ekom_std_position_error = [np.std(ekom_error_trans_x), np.std(ekom_error_trans_y),np.std(ekom_error_trans_z), np.std(ekom_error_roll), np.std(ekom_error_pitch), np.std(ekom_error_yaw)]
# rgbde_std_position_error = [np.std(rgbde_error_trans_x), np.std(rgbde_error_trans_y),np.std(rgbde_error_trans_z), np.std(rgbde_error_roll), np.std(rgbde_error_pitch), np.std(rgbde_error_yaw)]

# plt.bar(X_axis - 0.2, ekom_average_position_error, 0.4, yerr=ekom_std_position_error,label = 'EKOM', color=color_ekom)
# plt.bar(X_axis + 0.2, rgbde_average_position_error, 0.4, yerr=rgbde_std_position_error, label = 'RGB-D-E',  color=color_rgbde)
  
# plt.xticks(X_axis, X)
# plt.xlabel("Motions")
# plt.ylabel("Mean Position Error [m]")
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

# X = ['Tr X', 'Tr Y', 'Tr Z', 'Roll', 'Pitch', 'Yaw']

# X_axis = np.arange(len(X))
# ekom_average_angle_error = [np.mean(ekom_q_angle_trans_x), np.mean(ekom_q_angle_trans_y),np.mean(ekom_q_angle_trans_z), np.mean(ekom_q_angle_roll), np.mean(ekom_q_angle_pitch), np.mean(ekom_q_angle_yaw)]
# rgbde_average_angle_error = [np.mean(rgbde_q_angle_trans_x), np.mean(rgbde_q_angle_trans_y),np.mean(rgbde_q_angle_trans_z), np.mean(rgbde_q_angle_roll), np.mean(rgbde_q_angle_pitch), np.mean(rgbde_q_angle_yaw)]
  
# ekom_std_angle_error = [np.std(ekom_q_angle_trans_x), np.std(ekom_q_angle_trans_y),np.std(ekom_q_angle_trans_z), np.std(ekom_q_angle_roll), np.std(ekom_q_angle_pitch), np.std(ekom_q_angle_yaw)]
# rgbde_std_angle_error = [np.std(rgbde_q_angle_trans_x), np.std(rgbde_q_angle_trans_y),np.std(rgbde_q_angle_trans_z), np.std(rgbde_q_angle_roll), np.std(rgbde_q_angle_pitch), np.std(rgbde_q_angle_yaw)]

# plt.bar(X_axis - 0.2, ekom_average_angle_error, 0.4, yerr=ekom_std_angle_error, label = 'EKOM', color=color_ekom)
# plt.bar(X_axis + 0.2, rgbde_average_angle_error, 0.4, yerr=rgbde_std_angle_error, label = 'RGB-D-E',  color=color_rgbde)
  
# plt.xticks(X_axis, X)
# plt.xlabel("Motions")
# plt.ylabel("Mean Rotation Error [rad]")
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

# print(ekom_average_position_error, ekom_std_position_error)
# print(rgbde_average_position_error, rgbde_std_position_error)
# print(ekom_average_angle_error, ekom_std_angle_error)
# print(rgbde_average_angle_error, rgbde_std_angle_error)

# X = ['Translations' ,'Rotations']

# X_axis = np.arange(len(X))
# ekom_average_pos_errors = [np.mean(ekom_tr_datasets_position_errors), np.mean(ekom_rot_datasets_position_errors)]
# ekom_std_pos_errors = [np.std(ekom_tr_datasets_position_errors), np.std(ekom_rot_datasets_position_errors)]

# rgbde_average_pos_errors = [np.mean(rgbde_tr_datasets_position_errors), np.mean(rgbde_rot_datasets_position_errors)]
# rgbde_std_pos_errors = [np.std(rgbde_tr_datasets_position_errors), np.std(rgbde_rot_datasets_position_errors)]

# ekom_average_angle_errors = [np.mean(ekom_tr_datasets_angle_errors), np.mean(ekom_rot_datasets_angle_errors)]
# ekom_std_angle_errors = [np.std(ekom_tr_datasets_angle_errors), np.std(ekom_rot_datasets_angle_errors)]

# rgbde_average_angle_errors = [np.mean(rgbde_tr_datasets_angle_errors), np.mean(rgbde_rot_datasets_angle_errors)]
# rgbde_std_angle_errors = [np.std(rgbde_tr_datasets_angle_errors), np.std(rgbde_rot_datasets_angle_errors)]

# fig12 = plt.plot()
# plt.bar(X_axis - 0.2, ekom_average_pos_errors, 0.4, yerr=ekom_std_pos_errors, label = 'EKOM', color=color_ekom)
# plt.bar(X_axis + 0.2, rgbde_average_pos_errors, 0.4, yerr=rgbde_std_pos_errors, label = 'RGB-D-E', color=color_rgbde)
  
# plt.xticks(X_axis, X)
# # plt.xlabel("Algorithms")
# plt.ylabel('Position Error [m]')
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

# fig13 = plt.plot()
# plt.bar(X_axis - 0.2, ekom_average_angle_errors, 0.4, yerr=ekom_std_angle_errors, label = 'EKOM', color=color_ekom)
# plt.bar(X_axis + 0.2, rgbde_average_angle_errors, 0.4, yerr=rgbde_std_angle_errors, label = 'RGB-D-E', color=color_rgbde)
  
# plt.xticks(X_axis, X)
# # plt.xlabel("Algorithms")
# plt.ylabel('Angle Error [rad]')
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

labels = ['D1', ' ', 'D2', ' ', 'D3',' ', 'D4',' ', 'D5',' ', 'D6', ' ']
ticks=[0,1,2,3,4,5,6,7,8,9,10,11]
medianprops = dict(color='white')

quart_vec_pos=[ekom_error_yaw, rgbde_error_yaw, ekom_error_pitch, rgbde_error_pitch, ekom_error_roll, rgbde_error_roll, ekom_error_trans_z, rgbde_error_trans_z, ekom_error_trans_y, rgbde_error_trans_y, ekom_error_trans_x, rgbde_error_trans_x]
quart_vec_ang=[ekom_q_angle_yaw, rgbde_q_angle_yaw, ekom_q_angle_pitch, rgbde_q_angle_pitch, ekom_q_angle_roll, rgbde_q_angle_roll, ekom_q_angle_trans_z, rgbde_q_angle_trans_z, ekom_q_angle_trans_y, rgbde_q_angle_trans_y, ekom_q_angle_trans_x, rgbde_q_angle_trans_x]

# new_quart_array = np.array(quart_vec_pos).transpose

fig15, ax1 = plt.subplots(1,2)
algs = ['EKOM', 'RGB-D-E']
ax1[0].set_xlabel('Position error [m]', color='k')
ax1[1].set_xlabel('Rotation error [rad]', color='k')
res1 = ax1[0].boxplot(quart_vec_pos, labels=labels, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
res2 = ax1[1].boxplot(quart_vec_ang, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res1[element], color='k')
    plt.setp(res2[element], color='k')
ax1[1].set_yticklabels([])
ax1[1].set_yticks([])
colors=[color_x, color_y, color_x, color_y, color_x, color_y, color_x, color_y, color_x, color_y, color_x, color_y]
# color='white'
# colors = [color, color, color, color, color, color,color, color,color, color,color, color]
# patterns=[0,1,0,1,0,1,0,1,0,1,0,1]
for patch, color in zip(res1['boxes'], colors):
    patch.set_facecolor(color)
    # if pattern == 1:
    #     patch.set(hatch = '/')
for patch, color in zip(res2['boxes'], colors):
    patch.set_facecolor(color)
    # if pattern == 1:
    #     patch.set(hatch = '/')
fig15.subplots_adjust(wspace=0)
plt.show()
