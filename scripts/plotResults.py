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

color_edopt = '#5A8CFF'
color_rgbde = '#FF4782'

color_dragon = '#A80000'
color_gelating = '#040276'
color_mustard = '#A89800'

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

filePath_dataset = '/home/luna/shared/data/6-DOF-Objects/results_icra_2024/dragon/'
gt_trans_x_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_y_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_z_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_roll_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_pitch_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_yaw_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gt_trans_x_old['t'] = (gt_trans_x_old['t']-gt_trans_x_old['t'][0])*10
gt_trans_x_old['x'] = gt_trans_x_old['x']*0.01
gt_trans_x_old['y'] = gt_trans_x_old['y']*0.01
gt_trans_x_old['z'] = gt_trans_x_old['z']*0.01

gt_trans_y_old['t'] = (gt_trans_y_old['t']-gt_trans_y_old['t'][0])*10
gt_trans_y_old['x'] = gt_trans_y_old['x']*0.01
gt_trans_y_old['y'] = gt_trans_y_old['y']*0.01
gt_trans_y_old['z'] = gt_trans_y_old['z']*0.01

gt_trans_z_old['t'] = (gt_trans_z_old['t']-gt_trans_z_old['t'][0])*10
gt_trans_z_old['x'] = gt_trans_z_old['x']*0.01
gt_trans_z_old['y'] = gt_trans_z_old['y']*0.01
gt_trans_z_old['z'] = gt_trans_z_old['z']*0.01

gt_roll_old['t'] = (gt_roll_old['t']-gt_roll_old['t'][0])*10
gt_roll_old['x'] = gt_roll_old['x']*0.01
gt_roll_old['y'] = gt_roll_old['y']*0.01
gt_roll_old['z'] = gt_roll_old['z']*0.01

gt_pitch_old['t'] = (gt_pitch_old['t']-gt_pitch_old['t'][0])*10
gt_pitch_old['x'] = gt_pitch_old['x']*0.01
gt_pitch_old['y'] = gt_pitch_old['y']*0.01
gt_pitch_old['z'] = gt_pitch_old['z']*0.01

gt_yaw_old['t'] = (gt_yaw_old['t']-gt_yaw_old['t'][0])*10
gt_yaw_old['x'] = gt_yaw_old['x']*0.01
gt_yaw_old['y'] = gt_yaw_old['y']*0.01
gt_yaw_old['z'] = gt_yaw_old['z']*0.01

gt_trans_x_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_y_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_trans_z_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_roll_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_pitch_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gt_yaw_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gt_trans_x_new['t'] = (gt_trans_x_new['t']-gt_trans_x_new['t'][0])
gt_trans_x_new['x'] = gt_trans_x_new['x']*0.01
gt_trans_x_new['y'] = gt_trans_x_new['y']*0.01
gt_trans_x_new['z'] = gt_trans_x_new['z']*0.01

gt_trans_y_new['t'] = (gt_trans_y_new['t']-gt_trans_y_new['t'][0])
gt_trans_y_new['x'] = gt_trans_y_new['x']*0.01
gt_trans_y_new['y'] = gt_trans_y_new['y']*0.01
gt_trans_y_new['z'] = gt_trans_y_new['z']*0.01

gt_trans_z_new['t'] = (gt_trans_z_new['t']-gt_trans_z_new['t'][0])
gt_trans_z_new['x'] = gt_trans_z_new['x']*0.01
gt_trans_z_new['y'] = gt_trans_z_new['y']*0.01
gt_trans_z_new['z'] = gt_trans_z_new['z']*0.01

gt_roll_new['t'] = (gt_roll_new['t']-gt_roll_new['t'][0])
gt_roll_new['x'] = gt_roll_new['x']*0.01
gt_roll_new['y'] = gt_roll_new['y']*0.01
gt_roll_new['z'] = gt_roll_new['z']*0.01

gt_pitch_new['t'] = (gt_pitch_new['t']-gt_pitch_new['t'][0])
gt_pitch_new['x'] = gt_pitch_new['x']*0.01
gt_pitch_new['y'] = gt_pitch_new['y']*0.01
gt_pitch_new['z'] = gt_pitch_new['z']*0.01

gt_yaw_new['t'] = (gt_yaw_new['t']-gt_yaw_new['t'][0])
gt_yaw_new['x'] = gt_yaw_new['x']*0.01
gt_yaw_new['y'] = gt_yaw_new['y']*0.01
gt_yaw_new['z'] = gt_yaw_new['z']*0.01

edopt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
edopt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
edopt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
edopt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
edopt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
edopt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

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

edopt_trans_x_resampled_x = resampling_by_interpolate(gt_trans_x_new['t'], edopt_trans_x['t'], edopt_trans_x['x'])
edopt_trans_x_resampled_y = resampling_by_interpolate(gt_trans_x_new['t'], edopt_trans_x['t'], edopt_trans_x['y'])
edopt_trans_x_resampled_z = resampling_by_interpolate(gt_trans_x_new['t'], edopt_trans_x['t'], edopt_trans_x['z'])
edopt_trans_x_resampled_qx = resampling_by_interpolate(gt_trans_x_new['t'], edopt_trans_x['t'], edopt_trans_x['qx'])
edopt_trans_x_resampled_qy = resampling_by_interpolate(gt_trans_x_new['t'], edopt_trans_x['t'], edopt_trans_x['qy'])
edopt_trans_x_resampled_qz = resampling_by_interpolate(gt_trans_x_new['t'], edopt_trans_x['t'], edopt_trans_x['qz'])
edopt_trans_x_resampled_qw = resampling_by_interpolate(gt_trans_x_new['t'], edopt_trans_x['t'], edopt_trans_x['qw'])

rgbde_trans_x_resampled_x = resampling_by_interpolate(gt_trans_x_old['t'], rgbde_time_trans_x, rgbde_trans_x['x'])
rgbde_trans_x_resampled_y = resampling_by_interpolate(gt_trans_x_old['t'], rgbde_time_trans_x, rgbde_trans_x['y'])
rgbde_trans_x_resampled_z = resampling_by_interpolate(gt_trans_x_old['t'], rgbde_time_trans_x, rgbde_trans_x['z'])
rgbde_trans_x_resampled_qx = resampling_by_interpolate(gt_trans_x_old['t'], rgbde_time_trans_x, rgbde_trans_x['qx'])
rgbde_trans_x_resampled_qy = resampling_by_interpolate(gt_trans_x_old['t'], rgbde_time_trans_x, rgbde_trans_x['qy'])
rgbde_trans_x_resampled_qz = resampling_by_interpolate(gt_trans_x_old['t'], rgbde_time_trans_x, rgbde_trans_x['qz'])
rgbde_trans_x_resampled_qw = resampling_by_interpolate(gt_trans_x_old['t'], rgbde_time_trans_x, rgbde_trans_x['qw'])

edopt_trans_y_resampled_x = resampling_by_interpolate(gt_trans_y_new['t'], edopt_trans_y['t'], edopt_trans_y['x'])
edopt_trans_y_resampled_y = resampling_by_interpolate(gt_trans_y_new['t'], edopt_trans_y['t'], edopt_trans_y['y'])
edopt_trans_y_resampled_z = resampling_by_interpolate(gt_trans_y_new['t'], edopt_trans_y['t'], edopt_trans_y['z'])
edopt_trans_y_resampled_qx = resampling_by_interpolate(gt_trans_y_new['t'], edopt_trans_y['t'], edopt_trans_y['qx'])
edopt_trans_y_resampled_qy = resampling_by_interpolate(gt_trans_y_new['t'], edopt_trans_y['t'], edopt_trans_y['qy'])
edopt_trans_y_resampled_qz = resampling_by_interpolate(gt_trans_y_new['t'], edopt_trans_y['t'], edopt_trans_y['qz'])
edopt_trans_y_resampled_qw = resampling_by_interpolate(gt_trans_y_new['t'], edopt_trans_y['t'], edopt_trans_y['qw'])

rgbde_trans_y_resampled_x = resampling_by_interpolate(gt_trans_y_old['t'], rgbde_time_trans_y, rgbde_trans_y['x'])
rgbde_trans_y_resampled_y = resampling_by_interpolate(gt_trans_y_old['t'], rgbde_time_trans_y, rgbde_trans_y['y'])
rgbde_trans_y_resampled_z = resampling_by_interpolate(gt_trans_y_old['t'], rgbde_time_trans_y, rgbde_trans_y['z'])
rgbde_trans_y_resampled_qx = resampling_by_interpolate(gt_trans_y_old['t'], rgbde_time_trans_y, rgbde_trans_y['qx'])
rgbde_trans_y_resampled_qy = resampling_by_interpolate(gt_trans_y_old['t'], rgbde_time_trans_y, rgbde_trans_y['qy'])
rgbde_trans_y_resampled_qz = resampling_by_interpolate(gt_trans_y_old['t'], rgbde_time_trans_y, rgbde_trans_y['qz'])
rgbde_trans_y_resampled_qw = resampling_by_interpolate(gt_trans_y_old['t'], rgbde_time_trans_y, rgbde_trans_y['qw'])

edopt_trans_z_resampled_x = resampling_by_interpolate(gt_trans_z_new['t'], edopt_trans_z['t'], edopt_trans_z['x'])
edopt_trans_z_resampled_y = resampling_by_interpolate(gt_trans_z_new['t'], edopt_trans_z['t'], edopt_trans_z['y'])
edopt_trans_z_resampled_z = resampling_by_interpolate(gt_trans_z_new['t'], edopt_trans_z['t'], edopt_trans_z['z'])
edopt_trans_z_resampled_qx = resampling_by_interpolate(gt_trans_z_new['t'], edopt_trans_z['t'], edopt_trans_z['qx'])
edopt_trans_z_resampled_qy = resampling_by_interpolate(gt_trans_z_new['t'], edopt_trans_z['t'], edopt_trans_z['qy'])
edopt_trans_z_resampled_qz = resampling_by_interpolate(gt_trans_z_new['t'], edopt_trans_z['t'], edopt_trans_z['qz'])
edopt_trans_z_resampled_qw = resampling_by_interpolate(gt_trans_z_new['t'], edopt_trans_z['t'], edopt_trans_z['qw'])

rgbde_trans_z_resampled_x = resampling_by_interpolate(gt_trans_z_old['t'], rgbde_time_trans_z, rgbde_trans_z['x'])
rgbde_trans_z_resampled_y = resampling_by_interpolate(gt_trans_z_old['t'], rgbde_time_trans_z, rgbde_trans_z['y'])
rgbde_trans_z_resampled_z = resampling_by_interpolate(gt_trans_z_old['t'], rgbde_time_trans_z, rgbde_trans_z['z'])
rgbde_trans_z_resampled_qx = resampling_by_interpolate(gt_trans_z_old['t'], rgbde_time_trans_z, rgbde_trans_z['qx'])
rgbde_trans_z_resampled_qy = resampling_by_interpolate(gt_trans_z_old['t'], rgbde_time_trans_z, rgbde_trans_z['qy'])
rgbde_trans_z_resampled_qz = resampling_by_interpolate(gt_trans_z_old['t'], rgbde_time_trans_z, rgbde_trans_z['qz'])
rgbde_trans_z_resampled_qw = resampling_by_interpolate(gt_trans_z_old['t'], rgbde_time_trans_z, rgbde_trans_z['qw'])

edopt_roll_resampled_x = resampling_by_interpolate(gt_roll_new['t'], edopt_roll['t'], edopt_roll['x'])
edopt_roll_resampled_y = resampling_by_interpolate(gt_roll_new['t'], edopt_roll['t'], edopt_roll['y'])
edopt_roll_resampled_z = resampling_by_interpolate(gt_roll_new['t'], edopt_roll['t'], edopt_roll['z'])
edopt_roll_resampled_qx = resampling_by_interpolate(gt_roll_new['t'], edopt_roll['t'], edopt_roll['qx'])
edopt_roll_resampled_qy = resampling_by_interpolate(gt_roll_new['t'], edopt_roll['t'], edopt_roll['qy'])
edopt_roll_resampled_qz = resampling_by_interpolate(gt_roll_new['t'], edopt_roll['t'], edopt_roll['qz'])
edopt_roll_resampled_qw = resampling_by_interpolate(gt_roll_new['t'], edopt_roll['t'], edopt_roll['qw'])

rgbde_roll_resampled_x = resampling_by_interpolate(gt_roll_old['t'], rgbde_time_roll, rgbde_roll['x'])
rgbde_roll_resampled_y = resampling_by_interpolate(gt_roll_old['t'], rgbde_time_roll, rgbde_roll['y'])
rgbde_roll_resampled_z = resampling_by_interpolate(gt_roll_old['t'], rgbde_time_roll, rgbde_roll['z'])
rgbde_roll_resampled_qx = resampling_by_interpolate(gt_roll_old['t'], rgbde_time_roll, rgbde_roll['qx'])
rgbde_roll_resampled_qy = resampling_by_interpolate(gt_roll_old['t'], rgbde_time_roll, rgbde_roll['qy'])
rgbde_roll_resampled_qz = resampling_by_interpolate(gt_roll_old['t'], rgbde_time_roll, rgbde_roll['qz'])
rgbde_roll_resampled_qw = resampling_by_interpolate(gt_roll_old['t'], rgbde_time_roll, rgbde_roll['qw'])

edopt_pitch_resampled_x = resampling_by_interpolate(gt_pitch_new['t'], edopt_pitch['t'], edopt_pitch['x'])
edopt_pitch_resampled_y = resampling_by_interpolate(gt_pitch_new['t'], edopt_pitch['t'], edopt_pitch['y'])
edopt_pitch_resampled_z = resampling_by_interpolate(gt_pitch_new['t'], edopt_pitch['t'], edopt_pitch['z'])
edopt_pitch_resampled_qx = resampling_by_interpolate(gt_pitch_new['t'], edopt_pitch['t'], edopt_pitch['qx'])
edopt_pitch_resampled_qy = resampling_by_interpolate(gt_pitch_new['t'], edopt_pitch['t'], edopt_pitch['qy'])
edopt_pitch_resampled_qz = resampling_by_interpolate(gt_pitch_new['t'], edopt_pitch['t'], edopt_pitch['qz'])
edopt_pitch_resampled_qw = resampling_by_interpolate(gt_pitch_new['t'], edopt_pitch['t'], edopt_pitch['qw'])

rgbde_pitch_resampled_x = resampling_by_interpolate(gt_pitch_old['t'], rgbde_time_pitch, rgbde_pitch['x'])
rgbde_pitch_resampled_y = resampling_by_interpolate(gt_pitch_old['t'], rgbde_time_pitch, rgbde_pitch['y'])
rgbde_pitch_resampled_z = resampling_by_interpolate(gt_pitch_old['t'], rgbde_time_pitch, rgbde_pitch['z'])
rgbde_pitch_resampled_qx = resampling_by_interpolate(gt_pitch_old['t'], rgbde_time_pitch, rgbde_pitch['qx'])
rgbde_pitch_resampled_qy = resampling_by_interpolate(gt_pitch_old['t'], rgbde_time_pitch, rgbde_pitch['qy'])
rgbde_pitch_resampled_qz = resampling_by_interpolate(gt_pitch_old['t'], rgbde_time_pitch, rgbde_pitch['qz'])
rgbde_pitch_resampled_qw = resampling_by_interpolate(gt_pitch_old['t'], rgbde_time_pitch, rgbde_pitch['qw'])

edopt_yaw_resampled_x = resampling_by_interpolate(gt_yaw_new['t'], edopt_yaw['t'], edopt_yaw['x'])
edopt_yaw_resampled_y = resampling_by_interpolate(gt_yaw_new['t'], edopt_yaw['t'], edopt_yaw['y'])
edopt_yaw_resampled_z = resampling_by_interpolate(gt_yaw_new['t'], edopt_yaw['t'], edopt_yaw['z'])
edopt_yaw_resampled_qx = resampling_by_interpolate(gt_yaw_new['t'], edopt_yaw['t'], edopt_yaw['qx'])
edopt_yaw_resampled_qy = resampling_by_interpolate(gt_yaw_new['t'], edopt_yaw['t'], edopt_yaw['qy'])
edopt_yaw_resampled_qz = resampling_by_interpolate(gt_yaw_new['t'], edopt_yaw['t'], edopt_yaw['qz'])
edopt_yaw_resampled_qw = resampling_by_interpolate(gt_yaw_new['t'], edopt_yaw['t'], edopt_yaw['qw'])

rgbde_yaw_resampled_x = resampling_by_interpolate(gt_yaw_old['t'], rgbde_time_yaw, rgbde_yaw['x'])
rgbde_yaw_resampled_y = resampling_by_interpolate(gt_yaw_old['t'], rgbde_time_yaw, rgbde_yaw['y'])
rgbde_yaw_resampled_z = resampling_by_interpolate(gt_yaw_old['t'], rgbde_time_yaw, rgbde_yaw['z'])
rgbde_yaw_resampled_qx = resampling_by_interpolate(gt_yaw_old['t'], rgbde_time_yaw, rgbde_yaw['qx'])
rgbde_yaw_resampled_qy = resampling_by_interpolate(gt_yaw_old['t'], rgbde_time_yaw, rgbde_yaw['qy'])
rgbde_yaw_resampled_qz = resampling_by_interpolate(gt_yaw_old['t'], rgbde_time_yaw, rgbde_yaw['qz'])
rgbde_yaw_resampled_qw = resampling_by_interpolate(gt_yaw_old['t'], rgbde_time_yaw, rgbde_yaw['qw'])

edopt_trans_x_alpha,edopt_trans_x_beta,edopt_trans_x_gamma = quaternion_to_euler_angle(edopt_trans_x['qw'], edopt_trans_x['qx'], edopt_trans_x['qy'], edopt_trans_x['qz'])
rgbde_trans_x_alpha,rgbde_trans_x_beta,rgbde_trans_x_gamma = quaternion_to_euler_angle(rgbde_trans_x['qw'], rgbde_trans_x['qx'], rgbde_trans_x['qy'], rgbde_trans_x['qz'])
gt_trans_x_alpha_new,gt_trans_x_beta_new,gt_trans_x_gamma_new = quaternion_to_euler_angle(gt_trans_x_new['qw'], gt_trans_x_new['qx'], gt_trans_x_new['qy'], gt_trans_x_new['qz'])
gt_trans_x_alpha_old,gt_trans_x_beta_old,gt_trans_x_gamma_old = quaternion_to_euler_angle(gt_trans_x_old['qw'], gt_trans_x_old['qx'], gt_trans_x_old['qy'], gt_trans_x_old['qz'])

edopt_trans_x_alpha_cleaned = cleanEuler(edopt_trans_x_alpha,0)
edopt_trans_x_beta_cleaned = cleanEuler(edopt_trans_x_beta,1)
edopt_trans_x_gamma_cleaned = cleanEuler(edopt_trans_x_gamma,2)

rgbde_trans_x_alpha_cleaned = cleanEuler(rgbde_trans_x_alpha,0)
rgbde_trans_x_beta_cleaned = cleanEuler(rgbde_trans_x_beta,1)
rgbde_trans_x_gamma_cleaned = cleanEuler(rgbde_trans_x_gamma,2)

gt_trans_x_alpha_cleaned_new = cleanEuler(gt_trans_x_alpha_new,0)
gt_trans_x_beta_cleaned_new = cleanEuler(gt_trans_x_beta_new,1)
gt_trans_x_gamma_cleaned_new = cleanEuler(gt_trans_x_gamma_new,1)

gt_trans_x_alpha_cleaned_old = cleanEuler(gt_trans_x_alpha_old,0)
gt_trans_x_beta_cleaned_old = cleanEuler(gt_trans_x_beta_old,1)
gt_trans_x_gamma_cleaned_old = cleanEuler(gt_trans_x_gamma_old,1)

edopt_trans_y_alpha,edopt_trans_y_beta,edopt_trans_y_gamma = quaternion_to_euler_angle(edopt_trans_y['qw'], edopt_trans_y['qx'], edopt_trans_y['qy'], edopt_trans_y['qz'])
rgbde_trans_y_alpha,rgbde_trans_y_beta,rgbde_trans_y_gamma = quaternion_to_euler_angle(rgbde_trans_y['qw'], rgbde_trans_y['qx'], rgbde_trans_y['qy'], rgbde_trans_y['qz'])
gt_trans_y_alpha_new,gt_trans_y_beta_new,gt_trans_y_gamma_new = quaternion_to_euler_angle(gt_trans_y_new['qw'], gt_trans_y_new['qx'], gt_trans_y_new['qy'], gt_trans_y_new['qz'])
gt_trans_y_alpha_old,gt_trans_y_beta_old,gt_trans_y_gamma_old = quaternion_to_euler_angle(gt_trans_y_old['qw'], gt_trans_y_old['qx'], gt_trans_y_old['qy'], gt_trans_y_old['qz'])

edopt_trans_y_alpha_cleaned = cleanEuler(edopt_trans_y_alpha,0)
edopt_trans_y_beta_cleaned = cleanEuler(edopt_trans_y_beta,1)
edopt_trans_y_gamma_cleaned = cleanEuler(edopt_trans_y_gamma,2)

rgbde_trans_y_alpha_cleaned = cleanEuler(rgbde_trans_y_alpha,0)
rgbde_trans_y_beta_cleaned = cleanEuler(rgbde_trans_y_beta,1)
rgbde_trans_y_gamma_cleaned = cleanEuler(rgbde_trans_y_gamma,2)

gt_trans_y_alpha_cleaned_new = cleanEuler(gt_trans_y_alpha_new,0)
gt_trans_y_beta_cleaned_new = cleanEuler(gt_trans_y_beta_new,1)
gt_trans_y_gamma_cleaned_new = cleanEuler(gt_trans_y_gamma_new,2)

gt_trans_y_alpha_cleaned_old = cleanEuler(gt_trans_y_alpha_old,0)
gt_trans_y_beta_cleaned_old = cleanEuler(gt_trans_y_beta_old,1)
gt_trans_y_gamma_cleaned_old = cleanEuler(gt_trans_y_gamma_old,2)

edopt_trans_z_alpha,edopt_trans_z_beta,edopt_trans_z_gamma = quaternion_to_euler_angle(edopt_trans_z['qw'], edopt_trans_z['qx'], edopt_trans_z['qy'], edopt_trans_z['qz'])
rgbde_trans_z_alpha,rgbde_trans_z_beta,rgbde_trans_z_gamma = quaternion_to_euler_angle(rgbde_trans_z['qw'], rgbde_trans_z['qx'], rgbde_trans_z['qy'], rgbde_trans_z['qz'])
gt_trans_z_alpha_new,gt_trans_z_beta_new,gt_trans_z_gamma_new = quaternion_to_euler_angle(gt_trans_z_new['qw'], gt_trans_z_new['qx'], gt_trans_z_new['qy'], gt_trans_z_new['qz'])
gt_trans_z_alpha_old,gt_trans_z_beta_old,gt_trans_z_gamma_old = quaternion_to_euler_angle(gt_trans_z_old['qw'], gt_trans_z_old['qx'], gt_trans_z_old['qy'], gt_trans_z_old['qz'])

edopt_trans_z_alpha_cleaned = cleanEuler(edopt_trans_z_alpha,0)
edopt_trans_z_beta_cleaned = cleanEuler(edopt_trans_z_beta,1)
edopt_trans_z_gamma_cleaned = cleanEuler(edopt_trans_z_gamma,2)

rgbde_trans_z_alpha_cleaned = cleanEuler(rgbde_trans_z_alpha,0)
rgbde_trans_z_beta_cleaned = cleanEuler(rgbde_trans_z_beta,1)
rgbde_trans_z_gamma_cleaned = cleanEuler(rgbde_trans_z_gamma,2)

gt_trans_z_alpha_cleaned_new = cleanEuler(gt_trans_z_alpha_new,0)
gt_trans_z_beta_cleaned_new = cleanEuler(gt_trans_z_beta_new,1)
gt_trans_z_gamma_cleaned_new = cleanEuler(gt_trans_z_gamma_new,2)

gt_trans_z_alpha_cleaned_old = cleanEuler(gt_trans_z_alpha_old,0)
gt_trans_z_beta_cleaned_old = cleanEuler(gt_trans_z_beta_old,1)
gt_trans_z_gamma_cleaned_old = cleanEuler(gt_trans_z_gamma_old,2)

edopt_roll_alpha,edopt_roll_beta,edopt_roll_gamma = quaternion_to_euler_angle(edopt_roll['qw'], edopt_roll['qx'], edopt_roll['qy'], edopt_roll['qz'])
rgbde_roll_alpha,rgbde_roll_beta,rgbde_roll_gamma = quaternion_to_euler_angle(rgbde_roll['qw'], rgbde_roll['qx'], rgbde_roll['qy'], rgbde_roll['qz'])
gt_roll_alpha_new,gt_roll_beta_new,gt_roll_gamma_new = quaternion_to_euler_angle(gt_roll_new['qw'], gt_roll_new['qx'], gt_roll_new['qy'], gt_roll_new['qz'])
gt_roll_alpha_old,gt_roll_beta_old,gt_roll_gamma_old = quaternion_to_euler_angle(gt_roll_old['qw'], gt_roll_old['qx'], gt_roll_old['qy'], gt_roll_old['qz'])

edopt_roll_alpha_cleaned = cleanEuler(edopt_roll_alpha,0)
edopt_roll_beta_cleaned = cleanEuler(edopt_roll_beta,1)
edopt_roll_gamma_cleaned = cleanEuler(edopt_roll_gamma,2)

rgbde_roll_alpha_cleaned = cleanEuler(rgbde_roll_alpha,0)
rgbde_roll_beta_cleaned = cleanEuler(rgbde_roll_beta,1)
rgbde_roll_gamma_cleaned = cleanEuler(rgbde_roll_gamma,2)

gt_roll_alpha_cleaned_new = cleanEuler(gt_roll_alpha_new,0)
gt_roll_beta_cleaned_new = cleanEuler(gt_roll_beta_new,1)
gt_roll_gamma_cleaned_new = cleanEuler(gt_roll_gamma_new,2)

gt_roll_alpha_cleaned_old = cleanEuler(gt_roll_alpha_old,0)
gt_roll_beta_cleaned_old = cleanEuler(gt_roll_beta_old,1)
gt_roll_gamma_cleaned_old = cleanEuler(gt_roll_gamma_old,2)

edopt_pitch_alpha,edopt_pitch_beta,edopt_pitch_gamma = quaternion_to_euler_angle(edopt_pitch['qw'], edopt_pitch['qx'], edopt_pitch['qy'], edopt_pitch['qz'])
rgbde_pitch_alpha,rgbde_pitch_beta,rgbde_pitch_gamma = quaternion_to_euler_angle(rgbde_pitch['qw'], rgbde_pitch['qx'], rgbde_pitch['qy'], rgbde_pitch['qz'])
gt_pitch_alpha_new,gt_pitch_beta_new,gt_pitch_gamma_new = quaternion_to_euler_angle(gt_pitch_new['qw'], gt_pitch_new['qx'], gt_pitch_new['qy'], gt_pitch_new['qz'])
gt_pitch_alpha_old,gt_pitch_beta_old,gt_pitch_gamma_old = quaternion_to_euler_angle(gt_pitch_old['qw'], gt_pitch_old['qx'], gt_pitch_old['qy'], gt_pitch_old['qz'])

edopt_pitch_alpha_cleaned = cleanEuler(edopt_pitch_alpha,0)
edopt_pitch_beta_cleaned = cleanEuler(edopt_pitch_beta,1)
edopt_pitch_gamma_cleaned = cleanEuler(edopt_pitch_gamma,2)

rgbde_pitch_alpha_cleaned = cleanEuler(rgbde_pitch_alpha,0)
rgbde_pitch_beta_cleaned = cleanEuler(rgbde_pitch_beta,1)
rgbde_pitch_gamma_cleaned = cleanEuler(rgbde_pitch_gamma,2)

gt_pitch_alpha_cleaned_new = cleanEuler(gt_pitch_alpha_new,0)
gt_pitch_beta_cleaned_new = cleanEuler(gt_pitch_beta_new,1)
gt_pitch_gamma_cleaned_new = cleanEuler(gt_pitch_gamma_new,2)

gt_pitch_alpha_cleaned_old = cleanEuler(gt_pitch_alpha_old,0)
gt_pitch_beta_cleaned_old = cleanEuler(gt_pitch_beta_old,1)
gt_pitch_gamma_cleaned_old = cleanEuler(gt_pitch_gamma_old,2)

edopt_yaw_alpha,edopt_yaw_beta,edopt_yaw_gamma = quaternion_to_euler_angle(edopt_yaw['qw'], edopt_yaw['qx'], edopt_yaw['qy'], edopt_yaw['qz'])
rgbde_yaw_alpha,rgbde_yaw_beta,rgbde_yaw_gamma = quaternion_to_euler_angle(rgbde_yaw['qw'], rgbde_yaw['qx'], rgbde_yaw['qy'], rgbde_yaw['qz'])
gt_yaw_alpha_new,gt_yaw_beta_new,gt_yaw_gamma_new = quaternion_to_euler_angle(gt_yaw_new['qw'], gt_yaw_new['qx'], gt_yaw_new['qy'], gt_yaw_new['qz'])
gt_yaw_alpha_old,gt_yaw_beta_old,gt_yaw_gamma_old = quaternion_to_euler_angle(gt_yaw_old['qw'], gt_yaw_old['qx'], gt_yaw_old['qy'], gt_yaw_old['qz'])

edopt_yaw_alpha_cleaned = cleanEuler(edopt_yaw_alpha,0)
edopt_yaw_beta_cleaned = cleanEuler(edopt_yaw_beta,1)
edopt_yaw_gamma_cleaned = cleanEuler(edopt_yaw_gamma,2)

rgbde_yaw_alpha_cleaned = cleanEuler(rgbde_yaw_alpha,0)
rgbde_yaw_beta_cleaned = cleanEuler(rgbde_yaw_beta,1)
rgbde_yaw_gamma_cleaned = cleanEuler(rgbde_yaw_gamma,2)

gt_yaw_alpha_cleaned_new = cleanEuler(gt_yaw_alpha_new,0)
gt_yaw_beta_cleaned_new = cleanEuler(gt_yaw_beta_new,1)
gt_yaw_gamma_cleaned_new = cleanEuler(gt_yaw_gamma_new,2)

overlapping1 = 1
overlapping2 = 0.3

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18.5, 9.5)
# axs[0,0].plot(edopt_trans_x['t'], edopt_trans_x['x'], color=color_x, lw=4, label='x')
# axs[0,0].plot(edopt_trans_x['t'], edopt_trans_x['y'], color=color_y, lw=4, label='y')
# axs[0,0].plot(edopt_trans_x['t'], edopt_trans_x['z'], color=color_z, lw=4, label='z')
# axs[0,0].plot(gt_trans_x['t'], gt_trans_x['x'], color='k',  lw=2, ls='--')
# axs[0,0].plot(gt_trans_x['t'], gt_trans_x['y'], color='k',  lw=2, ls='--')
# axs[0,0].plot(gt_trans_x['t'], gt_trans_x['z'], color='k',  lw=2, ls='--')
# axs[0,0].plot(rgbde_time_trans_x, rgbde_trans_x['x'], color=color_x,  lw=12, alpha=overlapping2)
# axs[0,0].plot(rgbde_time_trans_x, rgbde_trans_x['y'], color=color_y,  lw=12, alpha=overlapping2)
# axs[0,0].plot(rgbde_time_trans_x, rgbde_trans_x['z'], color=color_z,  lw=12, alpha=overlapping2)
# axs[0,0].spines['top'].set_visible(False)
# axs[0,0].spines['right'].set_visible(False)
# # axs[0,0].spines['bottom'].set_visible(False)
# # axs[0,0].spines['left'].set_visible(False)
# axs[0,0].text(.5,.9,'D1',horizontalalignment='center',transform=axs[0,0].transAxes)
# axs[0,1].plot(edopt_trans_y['t'], edopt_trans_y['x'], color=color_x, lw=4, label='x')
# axs[0,1].plot(edopt_trans_y['t'], edopt_trans_y['y'], color=color_y, lw=4, label='y')
# axs[0,1].plot(edopt_trans_y['t'], edopt_trans_y['z'], color=color_z, lw=4, label='z')
# axs[0,1].plot(gt_trans_y['t'], gt_trans_y['x'], color='k', lw=2, ls='--')
# axs[0,1].plot(gt_trans_y['t'], gt_trans_y['y'], color='k', lw=2, ls='--')
# axs[0,1].plot(gt_trans_y['t'], gt_trans_y['z'], color='k', lw=2, ls='--')
# axs[0,1].plot(rgbde_time_trans_y, rgbde_trans_y['x'], color=color_x, lw=12, alpha=overlapping2)
# axs[0,1].plot(rgbde_time_trans_y, rgbde_trans_y['y'], color=color_y, lw=12, alpha=overlapping2)
# axs[0,1].plot(rgbde_time_trans_y, rgbde_trans_y['z'], color=color_z, lw=12, alpha=overlapping2)
# axs[0,1].spines['top'].set_visible(False)
# axs[0,1].spines['right'].set_visible(False)
# # axs[0,1].spines['bottom'].set_visible(False)
# axs[0,1].spines['left'].set_visible(False)
# axs[0,1].text(.5,.9,'D2',horizontalalignment='center',transform=axs[0,1].transAxes)
# axs[0,2].plot(edopt_trans_z['t'], edopt_trans_z['x'], color=color_x, lw=4, label='x')
# axs[0,2].plot(edopt_trans_z['t'], edopt_trans_z['y'], color=color_y, lw=4, label='y')
# axs[0,2].plot(edopt_trans_z['t'], edopt_trans_z['z'], color=color_z, lw=4, label='z')
# axs[0,2].plot(gt_trans_z['t'], gt_trans_z['x'], color='k',  lw=2, ls='--')
# axs[0,2].plot(gt_trans_z['t'], gt_trans_z['y'], color='k',  lw=2, ls='--')
# axs[0,2].plot(gt_trans_z['t'], gt_trans_z['z'], color='k',  lw=2, ls='--')
# axs[0,2].plot(rgbde_time_trans_z, rgbde_trans_z['x'], color=color_x, lw=12, alpha=overlapping2)
# axs[0,2].plot(rgbde_time_trans_z, rgbde_trans_z['y'], color=color_y, lw=12, alpha=overlapping2)
# axs[0,2].plot(rgbde_time_trans_z, rgbde_trans_z['z'], color=color_z, lw=12, alpha=overlapping2)
# axs[0,2].spines['top'].set_visible(False)
# axs[0,2].spines['right'].set_visible(False)
# # axs[0,2].spines['bottom'].set_visible(False)
# axs[0,2].spines['left'].set_visible(False)
# axs[0,2].text(.5,.9,'D3',horizontalalignment='center',transform=axs[0,2].transAxes)
# axs[2,2].plot(edopt_roll['t'], edopt_roll['x'], color=color_x, lw=4, label='x')
# axs[2,2].plot(edopt_roll['t'], edopt_roll['y'], color=color_y, lw=4, label='y')
# axs[2,2].plot(edopt_roll['t'], edopt_roll['z'], color=color_z, lw=4, label='z')
# axs[2,2].plot(gt_roll['t'], gt_roll['x'], color='k',  lw=2, ls='--')
# axs[2,2].plot(gt_roll['t'], gt_roll['y'], color='k',  lw=2, ls='--')
# axs[2,2].plot(gt_roll['t'], gt_roll['z'], color='k',  lw=2, ls='--')
# axs[2,2].plot(rgbde_time_roll, rgbde_roll['x'], color=color_x,  lw=12, alpha=overlapping2)
# axs[2,2].plot(rgbde_time_roll, rgbde_roll['y'], color=color_y,  lw=12, alpha=overlapping2)
# axs[2,2].plot(rgbde_time_roll, rgbde_roll['z'], color=color_z,  lw=12, alpha=overlapping2)
# axs[2,2].spines['top'].set_visible(False)
# axs[2,2].spines['right'].set_visible(False)
# # axs[2,2].spines['bottom'].set_visible(False)
# axs[2,2].spines['left'].set_visible(False)
# axs[2,2].text(.5,.9,'D6',horizontalalignment='center',transform=axs[2,2].transAxes)
# axs[2,0].plot(edopt_pitch['t'], edopt_pitch['x'], color=color_x, lw=4, label='x')
# axs[2,0].plot(edopt_pitch['t'], edopt_pitch['y'], color=color_y, lw=4, label='y')
# axs[2,0].plot(edopt_pitch['t'], edopt_pitch['z'], color=color_z, lw=4, label='z')
# axs[2,0].plot(gt_pitch['t'], gt_pitch['x'], color='k',  lw=2, ls='--')
# axs[2,0].plot(gt_pitch['t'], gt_pitch['y'], color='k',  lw=2, ls='--')
# axs[2,0].plot(gt_pitch['t'], gt_pitch['z'], color='k',  lw=2, ls='--')
# axs[2,0].plot(rgbde_time_pitch, rgbde_pitch['x'], color=color_x,  lw=12, alpha=overlapping2)
# axs[2,0].plot(rgbde_time_pitch, rgbde_pitch['y'], color=color_y,  lw=12, alpha=overlapping2)
# axs[2,0].plot(rgbde_time_pitch, rgbde_pitch['z'], color=color_z,  lw=12, alpha=overlapping2)
# axs[2,0].spines['top'].set_visible(False)
# axs[2,0].spines['right'].set_visible(False)
# # axs[2,0].spines['bottom'].set_visible(False)
# axs[2,0].text(.5,.9,'D4',horizontalalignment='center',transform=axs[2,0].transAxes)
# axs[2,1].plot(edopt_yaw['t'], edopt_yaw['x'], color=color_x, lw=4, label='x')
# axs[2,1].plot(edopt_yaw['t'], edopt_yaw['y'], color=color_y, lw=4, label='y')
# axs[2,1].plot(edopt_yaw['t'], edopt_yaw['z'], color=color_z, lw=4, label='z')
# axs[2,1].plot(gt_yaw['t'], gt_yaw['x'], color='k',  lw=2, ls='--')
# axs[2,1].plot(gt_yaw['t'], gt_yaw['y'], color='k',  lw=2, ls='--')
# axs[2,1].plot(gt_yaw['t'], gt_yaw['z'], color='k',  lw=2, ls='--')
# axs[2,1].plot(rgbde_time_yaw, rgbde_yaw['x'], color=color_x,  lw=12, alpha=overlapping2)
# axs[2,1].plot(rgbde_time_yaw, rgbde_yaw['y'], color=color_y,  lw=12, alpha=overlapping2)
# axs[2,1].plot(rgbde_time_yaw, rgbde_yaw['z'], color=color_z,  lw=12, alpha=overlapping2)
# axs[2,1].spines['top'].set_visible(False)
# axs[2,1].spines['right'].set_visible(False)
# # axs[2,1].spines['bottom'].set_visible(False)
# axs[2,1].spines['left'].set_visible(False)
# axs[2,1].text(.5,.9,'D5',horizontalalignment='center',transform=axs[2,1].transAxes)
# axs[1,0].plot(edopt_trans_x['t'], edopt_trans_x_alpha_cleaned, color=color_x, lw=4, label='qx')
# axs[1,0].plot(edopt_trans_x['t'], edopt_trans_x_beta_cleaned, color=color_y, lw=4, label='qy')
# axs[1,0].plot(edopt_trans_x['t'], edopt_trans_x_gamma_cleaned, color=color_z, lw=4, label='qz')
# axs[1,0].plot(gt_trans_x['t'], gt_trans_x_alpha_cleaned, color='k',  lw=1, ls='--')
# axs[1,0].plot(gt_trans_x['t'], gt_trans_x_beta_cleaned, color='k',  lw=1, ls='--')
# axs[1,0].plot(gt_trans_x['t'], gt_trans_x_gamma_cleaned, color='k',  lw=1, ls='--')
# axs[1,0].plot(rgbde_time_trans_x, rgbde_trans_x_alpha_cleaned, color=color_x,  lw=12, alpha=overlapping2)
# axs[1,0].plot(rgbde_time_trans_x, rgbde_trans_x_beta_cleaned, color=color_y,  lw=12, alpha=overlapping2)
# axs[1,0].plot(rgbde_time_trans_x, rgbde_trans_x_gamma_cleaned, color=color_z,  lw=12, alpha=overlapping2)
# axs[1,0].spines['top'].set_visible(False)
# axs[1,0].spines['right'].set_visible(False)
# # axs[1,0].spines['bottom'].set_visible(False)
# axs[1,1].plot(edopt_trans_y['t'], edopt_trans_y_alpha_cleaned, color=color_x, lw=4, label='qx')
# axs[1,1].plot(edopt_trans_y['t'], edopt_trans_y_beta_cleaned, color=color_y, lw=4, label='qy')
# axs[1,1].plot(edopt_trans_y['t'], edopt_trans_y_gamma_cleaned, color=color_z, lw=4, label='qz')
# axs[1,1].plot(gt_trans_y['t'], gt_trans_y_alpha_cleaned, color='k',  lw=2, ls='--')
# axs[1,1].plot(gt_trans_y['t'], gt_trans_y_beta_cleaned, color='k',  lw=2, ls='--')
# axs[1,1].plot(gt_trans_y['t'], gt_trans_y_gamma_cleaned, color='k',  lw=2, ls='--')
# axs[1,1].plot(rgbde_time_trans_y, rgbde_trans_y_alpha_cleaned, color=color_x,  lw=12, alpha=overlapping2)
# axs[1,1].plot(rgbde_time_trans_y, rgbde_trans_y_beta_cleaned, color=color_y,  lw=12, alpha=overlapping2)
# axs[1,1].plot(rgbde_time_trans_y, rgbde_trans_y_gamma_cleaned, color=color_z,  lw=12, alpha=overlapping2)
# axs[1,1].spines['top'].set_visible(False)
# axs[1,1].spines['right'].set_visible(False)
# # axs[1,1].spines['bottom'].set_visible(False)
# axs[1,1].spines['left'].set_visible(False)
# axs[1,2].plot(edopt_trans_z['t'], edopt_trans_z_alpha_cleaned, color=color_x, lw=4, label='qx')
# axs[1,2].plot(edopt_trans_z['t'], edopt_trans_z_beta_cleaned, color=color_y, lw=4, label='qy')
# axs[1,2].plot(edopt_trans_z['t'], edopt_trans_z_gamma_cleaned, color=color_z, lw=4, label='qz')
# axs[1,2].plot(gt_trans_z['t'], gt_trans_z_alpha_cleaned, color='k',  lw=2, ls='--')
# axs[1,2].plot(gt_trans_z['t'], gt_trans_z_beta_cleaned, color='k',  lw=2, ls='--')
# axs[1,2].plot(gt_trans_z['t'], gt_trans_z_gamma_cleaned, color='k',  lw=2, ls='--')
# axs[1,2].plot(rgbde_time_trans_z, rgbde_trans_z_alpha_cleaned, color=color_x,  lw=12, alpha=overlapping2)
# axs[1,2].plot(rgbde_time_trans_z, rgbde_trans_z_beta_cleaned, color=color_y,  lw=12, alpha=overlapping2)
# axs[1,2].plot(rgbde_time_trans_z, rgbde_trans_z_gamma_cleaned, color=color_z,  lw=12, alpha=overlapping2)
# axs[1,2].spines['top'].set_visible(False)
# axs[1,2].spines['right'].set_visible(False)
# # axs[1,2].spines['bottom'].set_visible(False)
# axs[1,2].spines['left'].set_visible(False)
# axs[3,2].plot(edopt_roll['t'], edopt_roll_alpha_cleaned, color=color_x, lw=4, label='qx')
# axs[3,2].plot(edopt_roll['t'], edopt_roll_beta_cleaned, color=color_y, lw=4, label='qy')
# axs[3,2].plot(edopt_roll['t'], edopt_roll_gamma_cleaned, color=color_z, lw=4, label='qz')
# axs[3,2].plot(gt_roll['t'], gt_roll_alpha_cleaned, color='k',  lw=2, ls='--')
# axs[3,2].plot(gt_roll['t'], gt_roll_beta_cleaned, color='k',  lw=2, ls='--')
# axs[3,2].plot(gt_roll['t'], gt_roll_gamma_cleaned, color='k',  lw=2, ls='--')
# axs[3,2].plot(rgbde_time_roll, rgbde_roll_alpha_cleaned, color=color_x,  lw=12, alpha=overlapping2)
# axs[3,2].plot(rgbde_time_roll, rgbde_roll_beta_cleaned, color=color_y,  lw=12, alpha=overlapping2)
# axs[3,2].plot(rgbde_time_roll, rgbde_roll_gamma_cleaned, color=color_z,  lw=12, alpha=overlapping2)
# axs[3,2].spines['top'].set_visible(False)
# axs[3,2].spines['right'].set_visible(False)
# # axs[3,2].spines['bottom'].set_visible(False)
# axs[3,2].spines['left'].set_visible(False)
# axs[3,0].plot(edopt_pitch['t'], edopt_pitch_alpha_cleaned, color=color_x, lw=4)
# axs[3,0].plot(edopt_pitch['t'], edopt_pitch_beta_cleaned, color=color_y, lw=4)
# axs[3,0].plot(edopt_pitch['t'], edopt_pitch_gamma_cleaned, color=color_z, lw=4)
# axs[3,0].plot(gt_pitch['t'], gt_pitch_alpha_cleaned, color='k',  lw=2, ls='--')
# axs[3,0].plot(gt_pitch['t'], gt_pitch_beta_cleaned, color='k',  lw=2, ls='--')
# axs[3,0].plot(gt_pitch['t'], gt_pitch_gamma_cleaned, color='k',  lw=2, ls='--')
# axs[3,0].plot(rgbde_time_pitch, rgbde_pitch_alpha_cleaned, color=color_x,  lw=12, alpha=overlapping2)
# axs[3,0].plot(rgbde_time_pitch, rgbde_pitch_beta_cleaned, color=color_y,  lw=12, alpha=overlapping2)
# axs[3,0].plot(rgbde_time_pitch, rgbde_pitch_gamma_cleaned, color=color_z,  lw=12, alpha=overlapping2)
# axs[3,0].spines['top'].set_visible(False)
# axs[3,0].spines['right'].set_visible(False)
# # axs[3,0].spines['bottom'].set_visible(False)
# axs[3,1].plot(edopt_yaw['t'], edopt_yaw_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)", lw=4)
# axs[3,1].plot(edopt_yaw['t'], edopt_yaw_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw)", lw=4)
# axs[3,1].plot(edopt_yaw['t'], edopt_yaw_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll)", lw=4)
# axs[3,1].plot(gt_yaw['t'], gt_yaw_alpha_cleaned, color='k',  lw=2, ls='--')
# axs[3,1].plot(gt_yaw['t'], gt_yaw_beta_cleaned, color='k',  lw=2, ls='--')
# axs[3,1].plot(gt_yaw['t'], gt_yaw_gamma_cleaned, color='k',  lw=2, ls='--')
# axs[3,1].plot(rgbde_time_yaw, rgbde_yaw_alpha_cleaned, color=color_x,  lw=12, alpha=overlapping2)
# axs[3,1].plot(rgbde_time_yaw, rgbde_yaw_beta_cleaned, color=color_y,  lw=12, alpha=overlapping2)
# axs[3,1].plot(rgbde_time_yaw, rgbde_yaw_gamma_cleaned, color=color_z,  lw=12, alpha=overlapping2)
# axs[3,1].spines['top'].set_visible(False)
# axs[3,1].spines['right'].set_visible(False)
# # axs[3,1].spines['bottom'].set_visible(False)
# axs[3,1].spines['left'].set_visible(False)
# for i in range(0, 4):
#     for j in range(0, 3):
#         axs[i,j].set_xlim([-2, 50])
# axs[0,0].set_ylim(-0.3,  1.2)
# axs[0,1].set_ylim(-0.3,  1.2)
# axs[0,2].set_ylim(-0.3,  1.2)
# axs[2,0].set_ylim(-0.3,  1.2)
# axs[2,1].set_ylim(-0.3,  1.2)
# axs[2,2].set_ylim(-0.3,  1.2)
# axs[1,0].set_ylim(-210,  300)
# axs[1,1].set_ylim(-210,  300)
# axs[1,2].set_ylim(-210,  300)
# axs[3,0].set_ylim(-210,  300)
# axs[3,1].set_ylim(-210,  300)
# axs[3,2].set_ylim(-210,  300)
# axs[0,0].set_xticks([])
# axs[1,0].set_xticks([])
# axs[2,0].set_xticks([])
# axs[0,1].set_xticks([])
# axs[1,1].set_xticks([])
# axs[2,1].set_xticks([])
# axs[0,2].set_xticks([])
# axs[1,2].set_xticks([])
# axs[2,2].set_xticks([])
# axs[0,0].set_xticklabels([])
# axs[1,0].set_xticklabels([])
# axs[2,0].set_xticklabels([])
# axs[0,1].set_xticklabels([])
# axs[1,1].set_xticklabels([])
# axs[2,1].set_xticklabels([])
# axs[0,2].set_xticklabels([])
# axs[1,2].set_xticklabels([])
# axs[2,2].set_xticklabels([])
# axs[0,1].set_yticklabels([])
# axs[0,2].set_yticklabels([])
# axs[1,1].set_yticklabels([])
# axs[1,2].set_yticklabels([])
# axs[2,1].set_yticklabels([])
# axs[2,2].set_yticklabels([])
# axs[3,1].set_yticklabels([])
# axs[3,2].set_yticklabels([])
# axs[0,0].set(ylabel='Pos. [m]')
# axs[1,0].set(ylabel='Rot. [deg]')
# axs[2,0].set(ylabel='Pos. [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Rot.[deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.32),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# line1 = plt.Line2D((.385,.385),(.1,.9), color="k", linewidth=2)
# line2 = plt.Line2D((.65,.65),(.1,.9), color="k", linewidth=2)
# line3 = plt.Line2D((.125,.9),(.5,.5), color="k", linewidth=2)
# fig_summary.add_artist(line1)
# fig_summary.add_artist(line2)
# fig_summary.add_artist(line3)
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

# fig1, (ax1, ax2) = plt.subplots(2)
# ax1.plot(edopt_trans_x['t'], edopt_trans_x['x'], color=color_x, label='x')
# ax1.plot(edopt_trans_x['t'], edopt_trans_x['y'], color=color_y, label='y')
# ax1.plot(edopt_trans_x['t'], edopt_trans_x['z'], color=color_z, label='z')
# ax1.plot(gt_trans_x['t'], gt_trans_x['x'], color=color_x, ls='--')
# ax1.plot(gt_trans_x['t'], gt_trans_x['y'], color=color_y, ls='--')
# ax1.plot(gt_trans_x['t'], gt_trans_x['z'], color=color_z, ls='--')
# ax1.plot(rgbde_time_trans_x, rgbde_trans_x['x'], color=color_x, ls='-.')
# ax1.plot(rgbde_time_trans_x, rgbde_trans_x['y'], color=color_y, ls='-.')
# ax1.plot(rgbde_time_trans_x, rgbde_trans_x['z'], color=color_z, ls='-.')
# ax2.plot(edopt_trans_x['t'], edopt_trans_x_alpha, color=color_x, label='roll')
# ax2.plot(edopt_trans_x['t'], edopt_trans_x_beta, color=color_y, label='pitch')
# ax2.plot(edopt_trans_x['t'], edopt_trans_x_gamma, color=color_z, label='yaw')
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
# ax1.plot(edopt_trans_y['t'], edopt_trans_y['x'], color=color_x, label='x')
# ax1.plot(edopt_trans_y['t'], edopt_trans_y['y'], color=color_y, label='y')
# ax1.plot(edopt_trans_y['t'], edopt_trans_y['z'], color=color_z, label='z')
# ax1.plot(gt_trans_y['t'], gt_trans_y['x'], color=color_x, ls='--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['y'], color=color_y, ls='--')
# ax1.plot(gt_trans_y['t'], gt_trans_y['z'], color=color_z, ls='--')
# ax1.plot(rgbde_time_trans_y, rgbde_trans_y['x'], color=color_x, ls='-.')
# ax1.plot(rgbde_time_trans_y, rgbde_trans_y['y'], color=color_y, ls='-.')
# ax1.plot(rgbde_time_trans_y, rgbde_trans_y['z'], color=color_z, ls='-.')
# ax2.plot(edopt_trans_y['t'], edopt_trans_y_alpha, color=color_x, label='roll')
# ax2.plot(edopt_trans_y['t'], edopt_trans_y_beta, color=color_y, label='pitch')
# ax2.plot(edopt_trans_y['t'], edopt_trans_y_gamma, color=color_z, label='yaw')
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
# ax1.plot(edopt_trans_z['t'], edopt_trans_z['x'], 'r', label='x')
# ax1.plot(edopt_trans_z['t'], edopt_trans_z['y'], 'g', label='y')
# ax1.plot(edopt_trans_z['t'], edopt_trans_z['z'], 'b', label='z')
# ax1.plot(gt_trans_z['t'], gt_trans_z['x'], 'r--')
# ax1.plot(gt_trans_z['t'], gt_trans_z['y'], 'g--')
# ax1.plot(gt_trans_z['t'], gt_trans_z['z'], 'b--')
# ax1.plot(rgbde_time_trans_z, rgbde_trans_z['x'], 'r-.')
# ax1.plot(rgbde_time_trans_z, rgbde_trans_z['y'], 'g-.')
# ax1.plot(rgbde_time_trans_z, rgbde_trans_z['z'], 'b-.')
# ax2.plot(edopt_trans_z['t'], edopt_trans_z['qx'], 'k', label='qx')
# ax2.plot(edopt_trans_z['t'], edopt_trans_z['qy'], 'y', label='qy')
# ax2.plot(edopt_trans_z['t'], edopt_trans_z['qz'], 'm', label='qz')
# ax2.plot(edopt_trans_z['t'], edopt_trans_z['qw'], 'c', label='qw')
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
# ax1.plot(edopt_roll['t'], edopt_roll['x'], 'r', label='x')
# ax1.plot(edopt_roll['t'], edopt_roll['y'], 'g', label='y')
# ax1.plot(edopt_roll['t'], edopt_roll['z'], 'b', label='z')
# ax1.plot(gt_roll['t'], gt_roll['x'], 'r--')
# ax1.plot(gt_roll['t'], gt_roll['y'], 'g--')
# ax1.plot(gt_roll['t'], gt_roll['z'], 'b--')
# ax1.plot(rgbde_time_roll, rgbde_roll['x'], 'r-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['y'], 'g-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['z'], 'b-.')
# ax2.plot(edopt_roll['t'], -edopt_roll['qx'], 'k', label='qx')
# ax2.plot(edopt_roll['t'], -edopt_roll['qy'], 'y', label='qy')
# ax2.plot(edopt_roll['t'], -edopt_roll['qz'], 'm', label='qz')
# ax2.plot(edopt_roll['t'], -edopt_roll['qw'], 'c', label='qw')
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
# # ax1.plot(edopt_roll['t'], edopt_roll['x'], 'r', label='x')
# # ax1.plot(edopt_roll['t'], edopt_roll['y'], 'g', label='y')
# # ax1.plot(edopt_roll['t'], edopt_roll['z'], 'b', label='z')
# # ax1.plot(gt_roll['t'], gt_roll['x'], 'r--')
# # ax1.plot(gt_roll['t'], gt_roll['y'], 'g--')
# # ax1.plot(gt_roll['t'], gt_roll['z'], 'b--')
# ax1.plot(rgbde_time_roll, rgbde_roll['x'], 'r-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['y'], 'g-.')
# ax1.plot(rgbde_time_roll, rgbde_roll['z'], 'b-.')
# # ax2.plot(edopt_roll['t'], -edopt_roll['qx'], 'k', label='qx')
# # ax2.plot(edopt_roll['t'], -edopt_roll['qy'], 'y', label='qy')
# # ax2.plot(edopt_roll['t'], -edopt_roll['qz'], 'm', label='qz')
# # ax2.plot(edopt_roll['t'], -edopt_roll['qw'], 'c', label='qw')
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
# ax1.plot(edopt_pitch['t'], edopt_pitch['x'], 'r', label='x')
# ax1.plot(edopt_pitch['t'], edopt_pitch['y'], 'g', label='y')
# ax1.plot(edopt_pitch['t'], edopt_pitch['z'], 'b', label='z')
# ax1.plot(gt_pitch['t'], gt_pitch['x'], 'r--')
# ax1.plot(gt_pitch['t'], gt_pitch['y'], 'g--')
# ax1.plot(gt_pitch['t'], gt_pitch['z'], 'b--')
# ax1.plot(rgbde_time_pitch, rgbde_pitch['x'], 'r-.')
# ax1.plot(rgbde_time_pitch, rgbde_pitch['y'], 'g-.')
# ax1.plot(rgbde_time_pitch, rgbde_pitch['z'], 'b-.')
# ax2.plot(edopt_pitch['t'], edopt_pitch['qx'], 'k', label='qx')
# ax2.plot(edopt_pitch['t'], edopt_pitch['qy'], 'y', label='qy')
# ax2.plot(edopt_pitch['t'], edopt_pitch['qz'], 'm', label='qz')
# ax2.plot(edopt_pitch['t'], edopt_pitch['qw'], 'c', label='qw')
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
# ax1.plot(edopt_yaw['t'], edopt_yaw['x'], 'r', label='x')
# ax1.plot(edopt_yaw['t'], edopt_yaw['y'], 'g', label='y')
# ax1.plot(edopt_yaw['t'], edopt_yaw['z'], 'b', label='z')
# ax1.plot(gt_yaw['t'], gt_yaw['x'], 'r--')
# ax1.plot(gt_yaw['t'], gt_yaw['y'], 'g--')
# ax1.plot(gt_yaw['t'], gt_yaw['z'], 'b--')
# # ax1.plot(rgbde_time_yaw, rgbde_yaw['x'], 'r-.')
# # ax1.plot(rgbde_time_yaw, rgbde_yaw['y'], 'g-.')
# # ax1.plot(rgbde_time_yaw, rgbde_yaw['z'], 'b-.')
# ax2.plot(edopt_yaw['t'], -edopt_yaw['qx'], 'k', label='qx')
# ax2.plot(edopt_yaw['t'], -edopt_yaw['qy'], 'y', label='qy')
# ax2.plot(edopt_yaw['t'], -edopt_yaw['qz'], 'm', label='qz')
# ax2.plot(edopt_yaw['t'], -edopt_yaw['qw'], 'c', label='qw')
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

# gt_max_x_trans_x = np.max(gt_trans_x['x'])
# gt_min_x_trans_x = np.min(gt_trans_x['x'])
# distance_x_trans_x = gt_max_x_trans_x - gt_min_x_trans_x

# gt_max_trans_y = np.max(gt_trans_y['y'])
# gt_min_trans_y = np.min(gt_trans_y['y'])
# distance_trans_y = gt_max_trans_y - gt_min_trans_y

# gt_max_trans_z = np.max(gt_trans_z['z'])
# gt_min_trans_z = np.min(gt_trans_z['z'])
# distance_trans_z = gt_max_trans_z - gt_min_trans_z

# gt_max_roll = np.max(gt_roll_gamma_cleaned)
# gt_min_roll = np.min(gt_roll_gamma_cleaned)
# total_angle_roll = gt_max_roll - gt_min_roll

# gt_max_pitch = np.max(gt_pitch_alpha_cleaned)
# gt_min_pitch = np.min(gt_pitch_alpha_cleaned)
# total_angle_pitch = gt_max_pitch - gt_min_pitch

# gt_max_yaw = np.max(gt_yaw_beta_cleaned)
# gt_min_yaw = np.min(gt_yaw_beta_cleaned)
# total_angle_yaw = gt_max_yaw - gt_min_yaw

edopt_error_trans_x = computeEuclideanDistance(gt_trans_x_new['x'], gt_trans_x_new['y'], gt_trans_x_new['z'], edopt_trans_x_resampled_x, edopt_trans_x_resampled_y, edopt_trans_x_resampled_z)
edopt_error_trans_y = computeEuclideanDistance(gt_trans_y_new['x'], gt_trans_y_new['y'], gt_trans_y_new['z'], edopt_trans_y_resampled_x, edopt_trans_y_resampled_y, edopt_trans_y_resampled_z)
edopt_error_trans_z = computeEuclideanDistance(gt_trans_z_new['x'], gt_trans_z_new['y'], gt_trans_z_new['z'], edopt_trans_z_resampled_x, edopt_trans_z_resampled_y, edopt_trans_z_resampled_z)
edopt_error_roll = computeEuclideanDistance(gt_roll_new['x'], gt_roll_new['y'], gt_roll_new['z'], edopt_roll_resampled_x, edopt_roll_resampled_y, edopt_roll_resampled_z)
edopt_error_pitch = computeEuclideanDistance(gt_pitch_new['x'], gt_pitch_new['y'], gt_pitch_new['z'], edopt_pitch_resampled_x, edopt_pitch_resampled_y, edopt_pitch_resampled_z)
edopt_error_yaw = computeEuclideanDistance(gt_yaw_new['x'], gt_yaw_new['y'], gt_yaw_new['z'], edopt_yaw_resampled_x, edopt_yaw_resampled_y, edopt_yaw_resampled_z)

edopt_q_angle_trans_x = computeQuaternionError(edopt_trans_x_resampled_qx, edopt_trans_x_resampled_qy, edopt_trans_x_resampled_qz, edopt_trans_x_resampled_qw, gt_trans_x_new['qx'], gt_trans_x_new['qy'], gt_trans_x_new['qz'], gt_trans_x_new['qw'])
edopt_q_angle_trans_y = computeQuaternionError(edopt_trans_y_resampled_qx, edopt_trans_y_resampled_qy, edopt_trans_y_resampled_qz, edopt_trans_y_resampled_qw, gt_trans_y_new['qx'], gt_trans_y_new['qy'], gt_trans_y_new['qz'], gt_trans_y_new['qw'])
edopt_q_angle_trans_z = computeQuaternionError(edopt_trans_z_resampled_qx, edopt_trans_z_resampled_qy, edopt_trans_z_resampled_qz, edopt_trans_z_resampled_qw, gt_trans_z_new['qx'], gt_trans_z_new['qy'], gt_trans_z_new['qz'], gt_trans_z_new['qw'])
edopt_q_angle_roll = computeQuaternionError(edopt_roll_resampled_qx, edopt_roll_resampled_qy, edopt_roll_resampled_qz, edopt_roll_resampled_qw, gt_roll_new['qx'], gt_roll_new['qy'], gt_roll_new['qz'], gt_roll_new['qw'])
edopt_q_angle_pitch = computeQuaternionError(edopt_pitch_resampled_qx, edopt_pitch_resampled_qy, edopt_pitch_resampled_qz, edopt_pitch_resampled_qw, gt_pitch_new['qx'], gt_pitch_new['qy'], gt_pitch_new['qz'], gt_pitch_new['qw'])
edopt_q_angle_yaw = computeQuaternionError(edopt_yaw_resampled_qx, edopt_yaw_resampled_qy, edopt_yaw_resampled_qz, edopt_yaw_resampled_qw, gt_yaw_new['qx'], gt_yaw_new['qy'], gt_yaw_new['qz'], gt_yaw_new['qw'])

rgbde_error_trans_x = computeEuclideanDistance(gt_trans_x_old['x'], gt_trans_x_old['y'], gt_trans_x_old['z'], rgbde_trans_x_resampled_x, rgbde_trans_x_resampled_y, rgbde_trans_x_resampled_z)
rgbde_error_trans_y = computeEuclideanDistance(gt_trans_y_old['x'], gt_trans_y_old['y'], gt_trans_y_old['z'], rgbde_trans_y_resampled_x, rgbde_trans_y_resampled_y, rgbde_trans_y_resampled_z)
rgbde_error_trans_z = computeEuclideanDistance(gt_trans_z_old['x'], gt_trans_z_old['y'], gt_trans_z_old['z'], rgbde_trans_z_resampled_x, rgbde_trans_z_resampled_y, rgbde_trans_z_resampled_z)
rgbde_error_roll = computeEuclideanDistance(gt_roll_old['x'], gt_roll_old['y'], gt_roll_old['z'], rgbde_roll_resampled_x, rgbde_roll_resampled_y, rgbde_roll_resampled_z)
rgbde_error_pitch = computeEuclideanDistance(gt_pitch_old['x'], gt_pitch_old['y'], gt_pitch_old['z'], rgbde_pitch_resampled_x, rgbde_pitch_resampled_y, rgbde_pitch_resampled_z)
rgbde_error_yaw = computeEuclideanDistance(gt_yaw_old['x'], gt_yaw_old['y'], gt_yaw_old['z'], rgbde_yaw_resampled_x, rgbde_yaw_resampled_y, rgbde_yaw_resampled_z)

rgbde_q_angle_trans_x = computeQuaternionError(rgbde_trans_x_resampled_qx, rgbde_trans_x_resampled_qy, rgbde_trans_x_resampled_qz, rgbde_trans_x_resampled_qw, gt_trans_x_old['qx'], gt_trans_x_old['qy'], gt_trans_x_old['qz'], gt_trans_x_old['qw'])
rgbde_q_angle_trans_y = computeQuaternionError(rgbde_trans_y_resampled_qx, rgbde_trans_y_resampled_qy, rgbde_trans_y_resampled_qz, rgbde_trans_y_resampled_qw, gt_trans_y_old['qx'], gt_trans_y_old['qy'], gt_trans_y_old['qz'], gt_trans_y_old['qw'])
rgbde_q_angle_trans_z = computeQuaternionError(rgbde_trans_z_resampled_qx, rgbde_trans_z_resampled_qy, rgbde_trans_z_resampled_qz, rgbde_trans_z_resampled_qw, gt_trans_z_old['qx'], gt_trans_z_old['qy'], gt_trans_z_old['qz'], gt_trans_z_old['qw'])
rgbde_q_angle_roll = computeQuaternionError(rgbde_roll_resampled_qx, rgbde_roll_resampled_qy, rgbde_roll_resampled_qz, rgbde_roll_resampled_qw, gt_roll_old['qx'], gt_roll_old['qy'], gt_roll_old['qz'], gt_roll_old['qw'])
rgbde_q_angle_pitch = computeQuaternionError(rgbde_pitch_resampled_qx, rgbde_pitch_resampled_qy, rgbde_pitch_resampled_qz, rgbde_pitch_resampled_qw, gt_pitch_old['qx'], gt_pitch_old['qy'], gt_pitch_old['qz'], gt_pitch_old['qw'])
rgbde_q_angle_yaw = computeQuaternionError(rgbde_yaw_resampled_qx, rgbde_yaw_resampled_qy, rgbde_yaw_resampled_qz, rgbde_yaw_resampled_qw, gt_yaw_old['qx'], gt_yaw_old['qy'], gt_yaw_old['qz'], gt_yaw_old['qw'])

edopt_tr_datasets_position_errors = np.concatenate((edopt_error_trans_x, edopt_error_trans_y, edopt_error_trans_z))
edopt_rot_datasets_position_errors = np.concatenate((edopt_error_roll, edopt_error_pitch, edopt_error_yaw))
edopt_tr_datasets_angle_errors = np.concatenate((edopt_q_angle_trans_x, edopt_q_angle_trans_y, edopt_q_angle_trans_z))
edopt_rot_datasets_angle_errors = np.concatenate((edopt_q_angle_roll, edopt_q_angle_pitch, edopt_q_angle_yaw))

rgbde_tr_datasets_position_errors = np.concatenate((rgbde_error_trans_x, rgbde_error_trans_y, rgbde_error_trans_z))
rgbde_rot_datasets_position_errors = np.concatenate((rgbde_error_roll, rgbde_error_pitch))
rgbde_tr_datasets_angle_errors = np.concatenate((rgbde_q_angle_trans_x, rgbde_q_angle_trans_y, rgbde_q_angle_trans_z))
rgbde_rot_datasets_angle_errors = np.concatenate((rgbde_q_angle_roll, rgbde_q_angle_pitch))

# print(np.mean(edopt_error_trans_x*100), "cm")
# print(np.mean(rgbde_error_trans_x*100), "cm")

# print(np.mean(edopt_q_angle_trans_x), "rad")
# print(np.mean(rgbde_q_angle_trans_x), "rad")

# print(np.mean(edopt_error_trans_z*100), "cm")
# print(np.mean(rgbde_error_trans_z*100), "cm")

# X = ['Tr X', 'Tr Y', 'Tr Z', 'Roll', 'Pitch', 'Yaw']

# X_axis = np.arange(len(X))
# edopt_average_position_error = [np.mean(edopt_error_trans_x), np.mean(edopt_error_trans_y),np.mean(edopt_error_trans_z), np.mean(edopt_error_roll), np.mean(edopt_error_pitch), np.mean(edopt_error_yaw)]
# rgbde_average_position_error = [np.mean(rgbde_error_trans_x), np.mean(rgbde_error_trans_y),np.mean(rgbde_error_trans_z), np.mean(rgbde_error_roll), np.mean(rgbde_error_pitch), np.mean(rgbde_error_yaw)]
  
# edopt_std_position_error = [np.std(edopt_error_trans_x), np.std(edopt_error_trans_y),np.std(edopt_error_trans_z), np.std(edopt_error_roll), np.std(edopt_error_pitch), np.std(edopt_error_yaw)]
# rgbde_std_position_error = [np.std(rgbde_error_trans_x), np.std(rgbde_error_trans_y),np.std(rgbde_error_trans_z), np.std(rgbde_error_roll), np.std(rgbde_error_pitch), np.std(rgbde_error_yaw)]

# plt.bar(X_axis - 0.2, edopt_average_position_error, 0.4, yerr=edopt_std_position_error,label = 'edopt', color=color_edopt)
# plt.bar(X_axis + 0.2, rgbde_average_position_error, 0.4, yerr=rgbde_std_position_error, label = 'RGB-D-E',  color=color_rgbde)
  
# plt.xticks(X_axis, X)
# plt.xlabel("Motions")
# plt.ylabel("Mean Position Error [m]")
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

# X = ['Tr X', 'Tr Y', 'Tr Z', 'Roll', 'Pitch', 'Yaw']

# X_axis = np.arange(len(X))
# edopt_average_angle_error = [np.mean(edopt_q_angle_trans_x), np.mean(edopt_q_angle_trans_y),np.mean(edopt_q_angle_trans_z), np.mean(edopt_q_angle_roll), np.mean(edopt_q_angle_pitch), np.mean(edopt_q_angle_yaw)]
# rgbde_average_angle_error = [np.mean(rgbde_q_angle_trans_x), np.mean(rgbde_q_angle_trans_y),np.mean(rgbde_q_angle_trans_z), np.mean(rgbde_q_angle_roll), np.mean(rgbde_q_angle_pitch), np.mean(rgbde_q_angle_yaw)]
  
# edopt_std_angle_error = [np.std(edopt_q_angle_trans_x), np.std(edopt_q_angle_trans_y),np.std(edopt_q_angle_trans_z), np.std(edopt_q_angle_roll), np.std(edopt_q_angle_pitch), np.std(edopt_q_angle_yaw)]
# rgbde_std_angle_error = [np.std(rgbde_q_angle_trans_x), np.std(rgbde_q_angle_trans_y),np.std(rgbde_q_angle_trans_z), np.std(rgbde_q_angle_roll), np.std(rgbde_q_angle_pitch), np.std(rgbde_q_angle_yaw)]

# plt.bar(X_axis - 0.2, edopt_average_angle_error, 0.4, yerr=edopt_std_angle_error, label = 'edopt', color=color_edopt)
# plt.bar(X_axis + 0.2, rgbde_average_angle_error, 0.4, yerr=rgbde_std_angle_error, label = 'RGB-D-E',  color=color_rgbde)
  
# plt.xticks(X_axis, X)
# plt.xlabel("Motions")
# plt.ylabel("Mean Rotation Error [rad]")
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

# print(edopt_average_position_error, edopt_std_position_error)
# print(rgbde_average_position_error, rgbde_std_position_error)
# print(edopt_average_angle_error, edopt_std_angle_error)
# print(rgbde_average_angle_error, rgbde_std_angle_error)

# X = ['Translations' ,'Rotations']

# X_axis = np.arange(len(X))
# edopt_average_pos_errors = [np.mean(edopt_tr_datasets_position_errors), np.mean(edopt_rot_datasets_position_errors)]
# edopt_std_pos_errors = [np.std(edopt_tr_datasets_position_errors), np.std(edopt_rot_datasets_position_errors)]

# rgbde_average_pos_errors = [np.mean(rgbde_tr_datasets_position_errors), np.mean(rgbde_rot_datasets_position_errors)]
# rgbde_std_pos_errors = [np.std(rgbde_tr_datasets_position_errors), np.std(rgbde_rot_datasets_position_errors)]

# edopt_average_angle_errors = [np.mean(edopt_tr_datasets_angle_errors), np.mean(edopt_rot_datasets_angle_errors)]
# edopt_std_angle_errors = [np.std(edopt_tr_datasets_angle_errors), np.std(edopt_rot_datasets_angle_errors)]

# rgbde_average_angle_errors = [np.mean(rgbde_tr_datasets_angle_errors), np.mean(rgbde_rot_datasets_angle_errors)]
# rgbde_std_angle_errors = [np.std(rgbde_tr_datasets_angle_errors), np.std(rgbde_rot_datasets_angle_errors)]

# fig12 = plt.plot()
# plt.bar(X_axis - 0.2, edopt_average_pos_errors, 0.4, yerr=edopt_std_pos_errors, label = 'edopt', color=color_edopt)
# plt.bar(X_axis + 0.2, rgbde_average_pos_errors, 0.4, yerr=rgbde_std_pos_errors, label = 'RGB-D-E', color=color_rgbde)
  
# plt.xticks(X_axis, X)
# # plt.xlabel("Algorithms")
# plt.ylabel('Position Error [m]')
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

# fig13 = plt.plot()
# plt.bar(X_axis - 0.2, edopt_average_angle_errors, 0.4, yerr=edopt_std_angle_errors, label = 'edopt', color=color_edopt)
# plt.bar(X_axis + 0.2, rgbde_average_angle_errors, 0.4, yerr=rgbde_std_angle_errors, label = 'RGB-D-E', color=color_rgbde)
  
# plt.xticks(X_axis, X)
# # plt.xlabel("Algorithms")
# plt.ylabel('Angle Error [rad]')
# # plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

rad_to_deg = 180/math.pi

labels = [' ', '\nD6', ' ', '\nD5', ' ', '\nD4',' ', '\nD3',' ', '\nD2',' ', '\nD1']
ticks=[0,1,2,3,4,5,6,7,8,9,10,11]
medianprops = dict(color='white')

quart_vec_pos=[edopt_error_yaw, rgbde_error_yaw, edopt_error_pitch, rgbde_error_pitch, edopt_error_roll, rgbde_error_roll, edopt_error_trans_z, rgbde_error_trans_z, edopt_error_trans_y, rgbde_error_trans_y, edopt_error_trans_x, rgbde_error_trans_x]
quart_vec_ang=[edopt_q_angle_yaw*rad_to_deg, rgbde_q_angle_yaw*rad_to_deg, edopt_q_angle_pitch*rad_to_deg, rgbde_q_angle_pitch*rad_to_deg, edopt_q_angle_roll*rad_to_deg, rgbde_q_angle_roll*rad_to_deg, edopt_q_angle_trans_z*rad_to_deg, rgbde_q_angle_trans_z*rad_to_deg, edopt_q_angle_trans_y*rad_to_deg, rgbde_q_angle_trans_y*rad_to_deg, edopt_q_angle_trans_x*rad_to_deg, rgbde_q_angle_trans_x*rad_to_deg]

# new_quart_array = np.array(quart_vec_pos).transpose

fig15, ax1 = plt.subplots(1,2)
fig15.set_size_inches(8, 6)
ax1[0].set_xlabel('Position error [m]', color='k')
ax1[1].set_xlabel('Rotation error [deg]', color='k')
res1 = ax1[0].boxplot(quart_vec_pos, labels=labels, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
res2 = ax1[1].boxplot(quart_vec_ang, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res1[element], color='k')
    plt.setp(res2[element], color='k')
ax1[1].set_yticklabels([])
ax1[1].set_yticks([])
ax1[0].set_xlim(-0.001,  0.025)
ax1[1].set_xlim(-0.5,  6.5)
ax1[1].xaxis.set_major_locator(plt.MaxNLocator(4))
colors=[color_edopt, color_rgbde, color_edopt, color_rgbde, color_edopt, color_rgbde, color_edopt, color_rgbde, color_edopt, color_rgbde, color_edopt, color_rgbde]
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
ax1[1].legend([res1["boxes"][0], res1["boxes"][1]], ['EDOPT', 'RGB-D-E'], bbox_to_anchor=(-0.7, 1.02, 0, 0.2), loc='lower left', ncol=2)
fig15.subplots_adjust(wspace=0)
plt.show()
