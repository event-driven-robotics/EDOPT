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

color_potted = '#005175' 
color_mustard = '#08FBFB'
color_gelatin = '#08FBD3'
color_tomato = '#5770CB'
color_dragon = color_edopt

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
    
# --------------------------------------------------------------------------- DRAGON ---------------------------------------------------------------------------------------

filePath_dataset = '/home/luna/shared/data/6-DOF-Objects/results_icra_2024/dragon/'
dragon_gt_trans_x_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_y_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_z_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_roll_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_pitch_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_yaw_old = np.genfromtxt(os.path.join(filePath_dataset, 'gt_old_dataset/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

dragon_gt_trans_x_old['t'] = (dragon_gt_trans_x_old['t']-dragon_gt_trans_x_old['t'][0])*10
dragon_gt_trans_x_old['x'] = dragon_gt_trans_x_old['x']*0.01
dragon_gt_trans_x_old['y'] = dragon_gt_trans_x_old['y']*0.01
dragon_gt_trans_x_old['z'] = dragon_gt_trans_x_old['z']*0.01

dragon_gt_trans_y_old['t'] = (dragon_gt_trans_y_old['t']-dragon_gt_trans_y_old['t'][0])*10
dragon_gt_trans_y_old['x'] = dragon_gt_trans_y_old['x']*0.01
dragon_gt_trans_y_old['y'] = dragon_gt_trans_y_old['y']*0.01
dragon_gt_trans_y_old['z'] = dragon_gt_trans_y_old['z']*0.01

dragon_gt_trans_z_old['t'] = (dragon_gt_trans_z_old['t']-dragon_gt_trans_z_old['t'][0])*10
dragon_gt_trans_z_old['x'] = dragon_gt_trans_z_old['x']*0.01
dragon_gt_trans_z_old['y'] = dragon_gt_trans_z_old['y']*0.01
dragon_gt_trans_z_old['z'] = dragon_gt_trans_z_old['z']*0.01

dragon_gt_roll_old['t'] = (dragon_gt_roll_old['t']-dragon_gt_roll_old['t'][0])*10
dragon_gt_roll_old['x'] = dragon_gt_roll_old['x']*0.01
dragon_gt_roll_old['y'] = dragon_gt_roll_old['y']*0.01
dragon_gt_roll_old['z'] = dragon_gt_roll_old['z']*0.01

dragon_gt_pitch_old['t'] = (dragon_gt_pitch_old['t']-dragon_gt_pitch_old['t'][0])*10
dragon_gt_pitch_old['x'] = dragon_gt_pitch_old['x']*0.01
dragon_gt_pitch_old['y'] = dragon_gt_pitch_old['y']*0.01
dragon_gt_pitch_old['z'] = dragon_gt_pitch_old['z']*0.01

dragon_gt_yaw_old['t'] = (dragon_gt_yaw_old['t']-dragon_gt_yaw_old['t'][0])*10
dragon_gt_yaw_old['x'] = dragon_gt_yaw_old['x']*0.01
dragon_gt_yaw_old['y'] = dragon_gt_yaw_old['y']*0.01
dragon_gt_yaw_old['z'] = dragon_gt_yaw_old['z']*0.01

dragon_gt_trans_x_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_y_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_z_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_roll_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_pitch_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_yaw_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt_new_dataset/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

dragon_gt_trans_x_new['t'] = (dragon_gt_trans_x_new['t']-dragon_gt_trans_x_new['t'][0])
dragon_gt_trans_x_new['x'] = dragon_gt_trans_x_new['x']*0.01
dragon_gt_trans_x_new['y'] = dragon_gt_trans_x_new['y']*0.01
dragon_gt_trans_x_new['z'] = dragon_gt_trans_x_new['z']*0.01

dragon_gt_trans_y_new['t'] = (dragon_gt_trans_y_new['t']-dragon_gt_trans_y_new['t'][0])
dragon_gt_trans_y_new['x'] = dragon_gt_trans_y_new['x']*0.01
dragon_gt_trans_y_new['y'] = dragon_gt_trans_y_new['y']*0.01
dragon_gt_trans_y_new['z'] = dragon_gt_trans_y_new['z']*0.01

dragon_gt_trans_z_new['t'] = (dragon_gt_trans_z_new['t']-dragon_gt_trans_z_new['t'][0])
dragon_gt_trans_z_new['x'] = dragon_gt_trans_z_new['x']*0.01
dragon_gt_trans_z_new['y'] = dragon_gt_trans_z_new['y']*0.01
dragon_gt_trans_z_new['z'] = dragon_gt_trans_z_new['z']*0.01

dragon_gt_roll_new['t'] = (dragon_gt_roll_new['t']-dragon_gt_roll_new['t'][0])
dragon_gt_roll_new['x'] = dragon_gt_roll_new['x']*0.01
dragon_gt_roll_new['y'] = dragon_gt_roll_new['y']*0.01
dragon_gt_roll_new['z'] = dragon_gt_roll_new['z']*0.01

dragon_gt_pitch_new['t'] = (dragon_gt_pitch_new['t']-dragon_gt_pitch_new['t'][0])
dragon_gt_pitch_new['x'] = dragon_gt_pitch_new['x']*0.01
dragon_gt_pitch_new['y'] = dragon_gt_pitch_new['y']*0.01
dragon_gt_pitch_new['z'] = dragon_gt_pitch_new['z']*0.01

dragon_gt_yaw_new['t'] = (dragon_gt_yaw_new['t']-dragon_gt_yaw_new['t'][0])
dragon_gt_yaw_new['x'] = dragon_gt_yaw_new['x']*0.01
dragon_gt_yaw_new['y'] = dragon_gt_yaw_new['y']*0.01
dragon_gt_yaw_new['z'] = dragon_gt_yaw_new['z']*0.01

dragon_edopt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_edopt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_edopt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_edopt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_edopt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_edopt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

dragon_edopt_trans_x_resampled_x = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_edopt_trans_x['t'], dragon_edopt_trans_x['x'])
dragon_edopt_trans_x_resampled_y = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_edopt_trans_x['t'], dragon_edopt_trans_x['y'])
dragon_edopt_trans_x_resampled_z = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_edopt_trans_x['t'], dragon_edopt_trans_x['z'])
dragon_edopt_trans_x_resampled_qx = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_edopt_trans_x['t'], dragon_edopt_trans_x['qx'])
dragon_edopt_trans_x_resampled_qy = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_edopt_trans_x['t'], dragon_edopt_trans_x['qy'])
dragon_edopt_trans_x_resampled_qz = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_edopt_trans_x['t'], dragon_edopt_trans_x['qz'])
dragon_edopt_trans_x_resampled_qw = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_edopt_trans_x['t'], dragon_edopt_trans_x['qw'])

dragon_edopt_trans_y_resampled_x = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_edopt_trans_y['t'], dragon_edopt_trans_y['x'])
dragon_edopt_trans_y_resampled_y = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_edopt_trans_y['t'], dragon_edopt_trans_y['y'])
dragon_edopt_trans_y_resampled_z = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_edopt_trans_y['t'], dragon_edopt_trans_y['z'])
dragon_edopt_trans_y_resampled_qx = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_edopt_trans_y['t'], dragon_edopt_trans_y['qx'])
dragon_edopt_trans_y_resampled_qy = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_edopt_trans_y['t'], dragon_edopt_trans_y['qy'])
dragon_edopt_trans_y_resampled_qz = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_edopt_trans_y['t'], dragon_edopt_trans_y['qz'])
dragon_edopt_trans_y_resampled_qw = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_edopt_trans_y['t'], dragon_edopt_trans_y['qw'])

dragon_edopt_trans_z_resampled_x = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_edopt_trans_z['t'], dragon_edopt_trans_z['x'])
dragon_edopt_trans_z_resampled_y = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_edopt_trans_z['t'], dragon_edopt_trans_z['y'])
dragon_edopt_trans_z_resampled_z = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_edopt_trans_z['t'], dragon_edopt_trans_z['z'])
dragon_edopt_trans_z_resampled_qx = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_edopt_trans_z['t'], dragon_edopt_trans_z['qx'])
dragon_edopt_trans_z_resampled_qy = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_edopt_trans_z['t'], dragon_edopt_trans_z['qy'])
dragon_edopt_trans_z_resampled_qz = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_edopt_trans_z['t'], dragon_edopt_trans_z['qz'])
dragon_edopt_trans_z_resampled_qw = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_edopt_trans_z['t'], dragon_edopt_trans_z['qw'])

dragon_edopt_roll_resampled_x = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_edopt_roll['t'], dragon_edopt_roll['x'])
dragon_edopt_roll_resampled_y = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_edopt_roll['t'], dragon_edopt_roll['y'])
dragon_edopt_roll_resampled_z = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_edopt_roll['t'], dragon_edopt_roll['z'])
dragon_edopt_roll_resampled_qx = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_edopt_roll['t'], dragon_edopt_roll['qx'])
dragon_edopt_roll_resampled_qy = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_edopt_roll['t'], dragon_edopt_roll['qy'])
dragon_edopt_roll_resampled_qz = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_edopt_roll['t'], dragon_edopt_roll['qz'])
dragon_edopt_roll_resampled_qw = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_edopt_roll['t'], dragon_edopt_roll['qw'])

dragon_edopt_pitch_resampled_x = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_edopt_pitch['t'], dragon_edopt_pitch['x'])
dragon_edopt_pitch_resampled_y = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_edopt_pitch['t'], dragon_edopt_pitch['y'])
dragon_edopt_pitch_resampled_z = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_edopt_pitch['t'], dragon_edopt_pitch['z'])
dragon_edopt_pitch_resampled_qx = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_edopt_pitch['t'], dragon_edopt_pitch['qx'])
dragon_edopt_pitch_resampled_qy = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_edopt_pitch['t'], dragon_edopt_pitch['qy'])
dragon_edopt_pitch_resampled_qz = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_edopt_pitch['t'], dragon_edopt_pitch['qz'])
dragon_edopt_pitch_resampled_qw = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_edopt_pitch['t'], dragon_edopt_pitch['qw'])

dragon_edopt_yaw_resampled_x = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_edopt_yaw['t'], dragon_edopt_yaw['x'])
dragon_edopt_yaw_resampled_y = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_edopt_yaw['t'], dragon_edopt_yaw['y'])
dragon_edopt_yaw_resampled_z = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_edopt_yaw['t'], dragon_edopt_yaw['z'])
dragon_edopt_yaw_resampled_qx = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_edopt_yaw['t'], dragon_edopt_yaw['qx'])
dragon_edopt_yaw_resampled_qy = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_edopt_yaw['t'], dragon_edopt_yaw['qy'])
dragon_edopt_yaw_resampled_qz = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_edopt_yaw['t'], dragon_edopt_yaw['qz'])
dragon_edopt_yaw_resampled_qw = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_edopt_yaw['t'], dragon_edopt_yaw['qw'])

dragon_edopt_trans_x_alpha,dragon_edopt_trans_x_beta,dragon_edopt_trans_x_gamma = quaternion_to_euler_angle(dragon_edopt_trans_x['qw'], dragon_edopt_trans_x['qx'], dragon_edopt_trans_x['qy'], dragon_edopt_trans_x['qz'])
dragon_gt_trans_x_new_alpha,dragon_gt_trans_x_new_beta,dragon_gt_trans_x_new_gamma = quaternion_to_euler_angle(dragon_gt_trans_x_new['qw'], dragon_gt_trans_x_new['qx'], dragon_gt_trans_x_new['qy'], dragon_gt_trans_x_new['qz'])

dragon_edopt_trans_x_alpha_cleaned = cleanEuler(dragon_edopt_trans_x_alpha,0)
dragon_edopt_trans_x_beta_cleaned = cleanEuler(dragon_edopt_trans_x_beta,1)
dragon_edopt_trans_x_gamma_cleaned = cleanEuler(dragon_edopt_trans_x_gamma,2)

dragon_gt_trans_x_new_alpha_cleaned = cleanEuler(dragon_gt_trans_x_new_alpha,0)
dragon_gt_trans_x_new_beta_cleaned = cleanEuler(dragon_gt_trans_x_new_beta,1)
dragon_gt_trans_x_new_gamma_cleaned = cleanEuler(dragon_gt_trans_x_new_gamma,1)

dragon_edopt_trans_y_alpha,dragon_edopt_trans_y_beta,dragon_edopt_trans_y_gamma = quaternion_to_euler_angle(dragon_edopt_trans_y['qw'], dragon_edopt_trans_y['qx'], dragon_edopt_trans_y['qy'], dragon_edopt_trans_y['qz'])
dragon_gt_trans_y_new_alpha,dragon_gt_trans_y_new_beta,dragon_gt_trans_y_new_gamma = quaternion_to_euler_angle(dragon_gt_trans_y_new['qw'], dragon_gt_trans_y_new['qx'], dragon_gt_trans_y_new['qy'], dragon_gt_trans_y_new['qz'])

dragon_edopt_trans_y_alpha_cleaned = cleanEuler(dragon_edopt_trans_y_alpha,0)
dragon_edopt_trans_y_beta_cleaned = cleanEuler(dragon_edopt_trans_y_beta,1)
dragon_edopt_trans_y_gamma_cleaned = cleanEuler(dragon_edopt_trans_y_gamma,2)

dragon_gt_trans_y_new_alpha_cleaned = cleanEuler(dragon_gt_trans_y_new_alpha,0)
dragon_gt_trans_y_new_beta_cleaned = cleanEuler(dragon_gt_trans_y_new_beta,1)
dragon_gt_trans_y_new_gamma_cleaned = cleanEuler(dragon_gt_trans_y_new_gamma,2)

dragon_edopt_trans_z_alpha,dragon_edopt_trans_z_beta,dragon_edopt_trans_z_gamma = quaternion_to_euler_angle(dragon_edopt_trans_z['qw'], dragon_edopt_trans_z['qx'], dragon_edopt_trans_z['qy'], dragon_edopt_trans_z['qz'])
dragon_gt_trans_z_new_alpha,dragon_gt_trans_z_new_beta,dragon_gt_trans_z_new_gamma = quaternion_to_euler_angle(dragon_gt_trans_z_new['qw'], dragon_gt_trans_z_new['qx'], dragon_gt_trans_z_new['qy'], dragon_gt_trans_z_new['qz'])

dragon_edopt_trans_z_alpha_cleaned = cleanEuler(dragon_edopt_trans_z_alpha,0)
dragon_edopt_trans_z_beta_cleaned = cleanEuler(dragon_edopt_trans_z_beta,1)
dragon_edopt_trans_z_gamma_cleaned = cleanEuler(dragon_edopt_trans_z_gamma,2)

dragon_gt_trans_z_new_alpha_cleaned = cleanEuler(dragon_gt_trans_z_new_alpha,0)
dragon_gt_trans_z_new_beta_cleaned = cleanEuler(dragon_gt_trans_z_new_beta,1)
dragon_gt_trans_z_new_gamma_cleaned = cleanEuler(dragon_gt_trans_z_new_gamma,2)

dragon_edopt_roll_alpha,dragon_edopt_roll_beta,dragon_edopt_roll_gamma = quaternion_to_euler_angle(dragon_edopt_roll['qw'], dragon_edopt_roll['qx'], dragon_edopt_roll['qy'], dragon_edopt_roll['qz'])
dragon_gt_roll_new_alpha,dragon_gt_roll_new_beta,dragon_gt_roll_new_gamma = quaternion_to_euler_angle(dragon_gt_roll_new['qw'], dragon_gt_roll_new['qx'], dragon_gt_roll_new['qy'], dragon_gt_roll_new['qz'])

dragon_edopt_roll_alpha_cleaned = cleanEuler(dragon_edopt_roll_alpha,0)
dragon_edopt_roll_beta_cleaned = cleanEuler(dragon_edopt_roll_beta,1)
dragon_edopt_roll_gamma_cleaned = cleanEuler(dragon_edopt_roll_gamma,2)

dragon_gt_roll_new_alpha_cleaned = cleanEuler(dragon_gt_roll_new_alpha,0)
dragon_gt_roll_new_beta_cleaned = cleanEuler(dragon_gt_roll_new_beta,1)
dragon_gt_roll_new_gamma_cleaned = cleanEuler(dragon_gt_roll_new_gamma,2)

dragon_edopt_pitch_alpha,dragon_edopt_pitch_beta,dragon_edopt_pitch_gamma = quaternion_to_euler_angle(dragon_edopt_pitch['qw'], dragon_edopt_pitch['qx'], dragon_edopt_pitch['qy'], dragon_edopt_pitch['qz'])
dragon_gt_pitch_new_alpha,dragon_gt_pitch_new_beta,dragon_gt_pitch_new_gamma = quaternion_to_euler_angle(dragon_gt_pitch_new['qw'], dragon_gt_pitch_new['qx'], dragon_gt_pitch_new['qy'], dragon_gt_pitch_new['qz'])

dragon_edopt_pitch_alpha_cleaned = cleanEuler(dragon_edopt_pitch_alpha,0)
dragon_edopt_pitch_beta_cleaned = cleanEuler(dragon_edopt_pitch_beta,1)
dragon_edopt_pitch_gamma_cleaned = cleanEuler(dragon_edopt_pitch_gamma,2)

dragon_gt_pitch_new_alpha_cleaned = cleanEuler(dragon_gt_pitch_new_alpha,0)
dragon_gt_pitch_new_beta_cleaned = cleanEuler(dragon_gt_pitch_new_beta,1)
dragon_gt_pitch_new_gamma_cleaned = cleanEuler(dragon_gt_pitch_new_gamma,2)

dragon_edopt_yaw_alpha,dragon_edopt_yaw_beta,dragon_edopt_yaw_gamma = quaternion_to_euler_angle(dragon_edopt_yaw['qw'], dragon_edopt_yaw['qx'], dragon_edopt_yaw['qy'], dragon_edopt_yaw['qz'])
dragon_gt_yaw_new_alpha,dragon_gt_yaw_new_beta,dragon_gt_yaw_new_gamma = quaternion_to_euler_angle(dragon_gt_yaw_new['qw'], dragon_gt_yaw_new['qx'], dragon_gt_yaw_new['qy'], dragon_gt_yaw_new['qz'])

dragon_edopt_yaw_alpha_cleaned = cleanEuler(dragon_edopt_yaw_alpha,0)
dragon_edopt_yaw_beta_cleaned = cleanEuler(dragon_edopt_yaw_beta,1)
dragon_edopt_yaw_gamma_cleaned = cleanEuler(dragon_edopt_yaw_gamma,2)

dragon_gt_yaw_new_alpha_cleaned = cleanEuler(dragon_gt_yaw_new_alpha,0)
dragon_gt_yaw_new_beta_cleaned = cleanEuler(dragon_gt_yaw_new_beta,1)
dragon_gt_yaw_new_gamma_cleaned = cleanEuler(dragon_gt_yaw_new_gamma,2)

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18, 12)
# axs[0,0].plot(dragon_edopt_trans_x['t'], dragon_edopt_trans_x['x'], color=color_x, label='x')
# axs[0,0].plot(dragon_edopt_trans_x['t'], dragon_edopt_trans_x['y'], color=color_y, label='y')
# axs[0,0].plot(dragon_edopt_trans_x['t'], dragon_edopt_trans_x['z'], color=color_z, label='z')
# axs[0,0].plot(dragon_gt_trans_x_new['t'], dragon_gt_trans_x_new['x'], color=color_x, ls='--')
# axs[0,0].plot(dragon_gt_trans_x_new['t'], dragon_gt_trans_x_new['y'], color=color_y, ls='--')
# axs[0,0].plot(dragon_gt_trans_x_new['t'], dragon_gt_trans_x_new['z'], color=color_z, ls='--')
# axs[0,1].plot(dragon_edopt_trans_y['t'], dragon_edopt_trans_y['x'], color=color_x, label='x')
# axs[0,1].plot(dragon_edopt_trans_y['t'], dragon_edopt_trans_y['y'], color=color_y, label='y')
# axs[0,1].plot(dragon_edopt_trans_y['t'], dragon_edopt_trans_y['z'], color=color_z, label='z')
# axs[0,1].plot(dragon_gt_trans_y_new['t'], dragon_gt_trans_y_new['x'], color=color_x, ls='--')
# axs[0,1].plot(dragon_gt_trans_y_new['t'], dragon_gt_trans_y_new['y'], color=color_y, ls='--')
# axs[0,1].plot(dragon_gt_trans_y_new['t'], dragon_gt_trans_y_new['z'], color=color_z, ls='--')
# axs[0,2].plot(dragon_edopt_trans_z['t'], dragon_edopt_trans_z['x'], color=color_x, label='x')
# axs[0,2].plot(dragon_edopt_trans_z['t'], dragon_edopt_trans_z['y'], color=color_y, label='y')
# axs[0,2].plot(dragon_edopt_trans_z['t'], dragon_edopt_trans_z['z'], color=color_z, label='z')
# axs[0,2].plot(dragon_gt_trans_z_new['t'], dragon_gt_trans_z_new['x'], color=color_x, ls='--')
# axs[0,2].plot(dragon_gt_trans_z_new['t'], dragon_gt_trans_z_new['y'], color=color_y, ls='--')
# axs[0,2].plot(dragon_gt_trans_z_new['t'], dragon_gt_trans_z_new['z'], color=color_z, ls='--')
# axs[2,0].plot(dragon_edopt_roll['t'], dragon_edopt_roll['x'], color=color_x, label='x')
# axs[2,0].plot(dragon_edopt_roll['t'], dragon_edopt_roll['y'], color=color_y, label='y')
# axs[2,0].plot(dragon_edopt_roll['t'], dragon_edopt_roll['z'], color=color_z, label='z')
# axs[2,0].plot(dragon_gt_roll_new['t'], dragon_gt_roll_new['x'], color=color_x, ls='--')
# axs[2,0].plot(dragon_gt_roll_new['t'], dragon_gt_roll_new['y'], color=color_y, ls='--')
# axs[2,0].plot(dragon_gt_roll_new['t'], dragon_gt_roll_new['z'], color=color_z, ls='--')
# axs[2,1].plot(dragon_edopt_pitch['t'], dragon_edopt_pitch['x'], color=color_x, label='x')
# axs[2,1].plot(dragon_edopt_pitch['t'], dragon_edopt_pitch['y'], color=color_y, label='y')
# axs[2,1].plot(dragon_edopt_pitch['t'], dragon_edopt_pitch['z'], color=color_z, label='z')
# axs[2,1].plot(dragon_gt_pitch_new['t'], dragon_gt_pitch_new['x'], color=color_x, ls='--')
# axs[2,1].plot(dragon_gt_pitch_new['t'], dragon_gt_pitch_new['y'], color=color_y, ls='--')
# axs[2,1].plot(dragon_gt_pitch_new['t'], dragon_gt_pitch_new['z'], color=color_z, ls='--')
# axs[2,2].plot(dragon_edopt_yaw['t'], dragon_edopt_yaw['x'], color=color_x, label='x')
# axs[2,2].plot(dragon_edopt_yaw['t'], dragon_edopt_yaw['y'], color=color_y, label='y')
# axs[2,2].plot(dragon_edopt_yaw['t'], dragon_edopt_yaw['z'], color=color_z, label='z')
# axs[2,2].plot(dragon_gt_yaw_new['t'], dragon_gt_yaw_new['x'], color=color_x, ls='--')
# axs[2,2].plot(dragon_gt_yaw_new['t'], dragon_gt_yaw_new['y'], color=color_y, ls='--')
# axs[2,2].plot(dragon_gt_yaw_new['t'], dragon_gt_yaw_new['z'], color=color_z, ls='--')
# axs[1,0].plot(dragon_edopt_trans_x['t'], dragon_edopt_trans_x_alpha_cleaned, color=color_x, label='qx')
# axs[1,0].plot(dragon_edopt_trans_x['t'], dragon_edopt_trans_x_beta_cleaned, color=color_y, label='qy')
# axs[1,0].plot(dragon_edopt_trans_x['t'], dragon_edopt_trans_x_gamma_cleaned, color=color_z, label='qz')
# axs[1,0].plot(dragon_gt_trans_x_new['t'], dragon_gt_trans_x_new_alpha_cleaned, color=color_x, ls = '--')
# axs[1,0].plot(dragon_gt_trans_x_new['t'], dragon_gt_trans_x_new_beta_cleaned, color=color_y, ls = '--')
# axs[1,0].plot(dragon_gt_trans_x_new['t'], dragon_gt_trans_x_new_gamma_cleaned, color=color_z, ls = '--')
# axs[1,1].plot(dragon_edopt_trans_y['t'], dragon_edopt_trans_y_alpha_cleaned, color=color_x, label='qx')
# axs[1,1].plot(dragon_edopt_trans_y['t'], dragon_edopt_trans_y_beta_cleaned, color=color_y, label='qy')
# axs[1,1].plot(dragon_edopt_trans_y['t'], dragon_edopt_trans_y_gamma_cleaned, color=color_z, label='qz')
# axs[1,1].plot(dragon_gt_trans_y_new['t'], dragon_gt_trans_y_new_alpha_cleaned, color=color_x, ls = '--')
# axs[1,1].plot(dragon_gt_trans_y_new['t'], dragon_gt_trans_y_new_beta_cleaned, color=color_y, ls = '--')
# axs[1,1].plot(dragon_gt_trans_y_new['t'], dragon_gt_trans_y_new_gamma_cleaned, color=color_z, ls = '--')
# axs[1,2].plot(dragon_edopt_trans_z['t'], dragon_edopt_trans_z_alpha_cleaned, color=color_x, label='qx')
# axs[1,2].plot(dragon_edopt_trans_z['t'], dragon_edopt_trans_z_beta_cleaned, color=color_y, label='qy')
# axs[1,2].plot(dragon_edopt_trans_z['t'], dragon_edopt_trans_z_gamma_cleaned, color=color_z, label='qz')
# axs[1,2].plot(dragon_gt_trans_z_new['t'], dragon_gt_trans_z_new_alpha_cleaned, color=color_x, ls = '--')
# axs[1,2].plot(dragon_gt_trans_z_new['t'], dragon_gt_trans_z_new_beta_cleaned, color=color_y, ls = '--')
# axs[1,2].plot(dragon_gt_trans_z_new['t'], dragon_gt_trans_z_new_gamma_cleaned, color=color_z, ls = '--')
# axs[3,0].plot(dragon_edopt_roll['t'], dragon_edopt_roll_alpha_cleaned, color=color_x, label='qx')
# axs[3,0].plot(dragon_edopt_roll['t'], dragon_edopt_roll_beta_cleaned, color=color_y, label='qy')
# axs[3,0].plot(dragon_edopt_roll['t'], dragon_edopt_roll_gamma_cleaned, color=color_z, label='qz')
# axs[3,0].plot(dragon_gt_roll_new['t'], dragon_gt_roll_new_alpha_cleaned, color=color_x, ls = '--')
# axs[3,0].plot(dragon_gt_roll_new['t'], dragon_gt_roll_new_beta_cleaned, color=color_y, ls = '--')
# axs[3,0].plot(dragon_gt_roll_new['t'], dragon_gt_roll_new_gamma_cleaned, color=color_z, ls = '--')
# axs[3,1].plot(dragon_edopt_pitch['t'], dragon_edopt_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - dragon_edopt")
# axs[3,1].plot(dragon_edopt_pitch['t'], dragon_edopt_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - dragon_edopt")
# axs[3,1].plot(dragon_edopt_pitch['t'], dragon_edopt_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - dragon_edopt")
# axs[3,1].plot(dragon_gt_pitch_new['t'], dragon_gt_pitch_new_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - dragon_gt")
# axs[3,1].plot(dragon_gt_pitch_new['t'], dragon_gt_pitch_new_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - dragon_gt")
# axs[3,1].plot(dragon_gt_pitch_new['t'], dragon_gt_pitch_new_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - dragon_gt")
# axs[3,2].plot(dragon_edopt_yaw['t'], dragon_edopt_yaw_alpha_cleaned, color=color_x, label='x or roll')
# axs[3,2].plot(dragon_edopt_yaw['t'], dragon_edopt_yaw_beta_cleaned, color=color_y, label='y or pitch')
# axs[3,2].plot(dragon_edopt_yaw['t'], dragon_edopt_yaw_gamma_cleaned, color=color_z, label='z or yaw')
# axs[3,2].plot(dragon_gt_yaw_new['t'], dragon_gt_yaw_new_alpha_cleaned, color=color_x, ls = '--')
# axs[3,2].plot(dragon_gt_yaw_new['t'], dragon_gt_yaw_new_beta_cleaned, color=color_y, ls = '--')
# axs[3,2].plot(dragon_gt_yaw_new['t'], dragon_gt_yaw_new_gamma_cleaned, color=color_z, ls = '--')
# # for i in range(0, 4):
# #     for j in range(0, 3):
# #         axs[i,j].set_xlim([-2, 50])
# axs[0,0].set_ylim(-0.3,  1)
# axs[0,1].set_ylim(-0.3,  1)
# axs[0,2].set_ylim(-0.3,  1)
# axs[2,0].set_ylim(-0.1,  0.9)
# axs[2,1].set_ylim(-0.1,  0.9)
# axs[2,2].set_ylim(-0.1,  0.9)
# axs[1,0].set_ylim(-120,  200)
# axs[1,1].set_ylim(-120,  200)
# axs[1,2].set_ylim(-120,  200)
# axs[3,0].set_ylim(-200,  300)
# axs[3,1].set_ylim(-200,  300)
# axs[3,2].set_ylim(-200,  300)
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
# axs[0,0].set(ylabel='Position [m]')
# axs[1,0].set(ylabel='Euler angles [deg]')
# axs[2,0].set(ylabel='Position [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

dragon_edopt_error_trans_x = computeEuclideanDistance(dragon_gt_trans_x_new['x'], dragon_gt_trans_x_new['y'], dragon_gt_trans_x_new['z'], dragon_edopt_trans_x_resampled_x, dragon_edopt_trans_x_resampled_y, dragon_edopt_trans_x_resampled_z)
dragon_edopt_error_trans_y = computeEuclideanDistance(dragon_gt_trans_y_new['x'], dragon_gt_trans_y_new['y'], dragon_gt_trans_y_new['z'], dragon_edopt_trans_y_resampled_x, dragon_edopt_trans_y_resampled_y, dragon_edopt_trans_y_resampled_z)
dragon_edopt_error_trans_z = computeEuclideanDistance(dragon_gt_trans_z_new['x'], dragon_gt_trans_z_new['y'], dragon_gt_trans_z_new['z'], dragon_edopt_trans_z_resampled_x, dragon_edopt_trans_z_resampled_y, dragon_edopt_trans_z_resampled_z)
dragon_edopt_error_roll = computeEuclideanDistance(dragon_gt_roll_new['x'], dragon_gt_roll_new['y'], dragon_gt_roll_new['z'], dragon_edopt_roll_resampled_x, dragon_edopt_roll_resampled_y, dragon_edopt_roll_resampled_z)
dragon_edopt_error_pitch = computeEuclideanDistance(dragon_gt_pitch_new['x'], dragon_gt_pitch_new['y'], dragon_gt_pitch_new['z'], dragon_edopt_pitch_resampled_x, dragon_edopt_pitch_resampled_y, dragon_edopt_pitch_resampled_z)
dragon_edopt_error_yaw = computeEuclideanDistance(dragon_gt_yaw_new['x'], dragon_gt_yaw_new['y'], dragon_gt_yaw_new['z'], dragon_edopt_yaw_resampled_x, dragon_edopt_yaw_resampled_y, dragon_edopt_yaw_resampled_z)

dragon_edopt_q_angle_trans_x = computeQuaternionError(dragon_edopt_trans_x_resampled_qx, dragon_edopt_trans_x_resampled_qy, dragon_edopt_trans_x_resampled_qz, dragon_edopt_trans_x_resampled_qw, dragon_gt_trans_x_new['qx'], dragon_gt_trans_x_new['qy'], dragon_gt_trans_x_new['qz'], dragon_gt_trans_x_new['qw'])
dragon_edopt_q_angle_trans_y = computeQuaternionError(dragon_edopt_trans_y_resampled_qx, dragon_edopt_trans_y_resampled_qy, dragon_edopt_trans_y_resampled_qz, dragon_edopt_trans_y_resampled_qw, dragon_gt_trans_y_new['qx'], dragon_gt_trans_y_new['qy'], dragon_gt_trans_y_new['qz'], dragon_gt_trans_y_new['qw'])
dragon_edopt_q_angle_trans_z = computeQuaternionError(dragon_edopt_trans_z_resampled_qx, dragon_edopt_trans_z_resampled_qy, dragon_edopt_trans_z_resampled_qz, dragon_edopt_trans_z_resampled_qw, dragon_gt_trans_z_new['qx'], dragon_gt_trans_z_new['qy'], dragon_gt_trans_z_new['qz'], dragon_gt_trans_z_new['qw'])
dragon_edopt_q_angle_roll = computeQuaternionError(dragon_edopt_roll_resampled_qx, dragon_edopt_roll_resampled_qy, dragon_edopt_roll_resampled_qz, dragon_edopt_roll_resampled_qw, dragon_gt_roll_new['qx'], dragon_gt_roll_new['qy'], dragon_gt_roll_new['qz'], dragon_gt_roll_new['qw'])
dragon_edopt_q_angle_pitch = computeQuaternionError(dragon_edopt_pitch_resampled_qx, dragon_edopt_pitch_resampled_qy, dragon_edopt_pitch_resampled_qz, dragon_edopt_pitch_resampled_qw, dragon_gt_pitch_new['qx'], dragon_gt_pitch_new['qy'], dragon_gt_pitch_new['qz'], dragon_gt_pitch_new['qw'])
dragon_edopt_q_angle_yaw = computeQuaternionError(dragon_edopt_yaw_resampled_qx, dragon_edopt_yaw_resampled_qy, dragon_edopt_yaw_resampled_qz, dragon_edopt_yaw_resampled_qw, dragon_gt_yaw_new['qx'], dragon_gt_yaw_new['qy'], dragon_gt_yaw_new['qz'], dragon_gt_yaw_new['qw'])

dragon_edopt_position_errors = np.concatenate((dragon_edopt_error_trans_x, dragon_edopt_error_trans_y, dragon_edopt_error_trans_z, dragon_edopt_error_roll, dragon_edopt_error_pitch, dragon_edopt_error_yaw))
dragon_edopt_rotation_errors = np.concatenate((dragon_edopt_q_angle_trans_x, dragon_edopt_q_angle_trans_y, dragon_edopt_q_angle_trans_z, dragon_edopt_q_angle_roll, dragon_edopt_q_angle_pitch, dragon_edopt_q_angle_yaw))




# ---------------------------------------------------------------------------  GELATIN  ---------------------------------------------------------------------------------------

filePath_dataset = '/home/luna/shared/data/6-DOF-Objects/results_icra_2024/gelatin/'
gelatin_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'gt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'gt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'gt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'gt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'gt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'gt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gelatin_gt_trans_x['t'] = (gelatin_gt_trans_x['t']-gelatin_gt_trans_x['t'][0])
gelatin_gt_trans_x['x'] = gelatin_gt_trans_x['x']*0.01
gelatin_gt_trans_x['y'] = gelatin_gt_trans_x['y']*0.01
gelatin_gt_trans_x['z'] = gelatin_gt_trans_x['z']*0.01

gelatin_gt_trans_y['t'] = (gelatin_gt_trans_y['t']-gelatin_gt_trans_y['t'][0])
gelatin_gt_trans_y['x'] = gelatin_gt_trans_y['x']*0.01
gelatin_gt_trans_y['y'] = gelatin_gt_trans_y['y']*0.01
gelatin_gt_trans_y['z'] = gelatin_gt_trans_y['z']*0.01

gelatin_gt_trans_z['t'] = (gelatin_gt_trans_z['t']-gelatin_gt_trans_z['t'][0])
gelatin_gt_trans_z['x'] = gelatin_gt_trans_z['x']*0.01
gelatin_gt_trans_z['y'] = gelatin_gt_trans_z['y']*0.01
gelatin_gt_trans_z['z'] = gelatin_gt_trans_z['z']*0.01

gelatin_gt_roll['t'] = (gelatin_gt_roll['t']-gelatin_gt_roll['t'][0])
gelatin_gt_roll['x'] = gelatin_gt_roll['x']*0.01
gelatin_gt_roll['y'] = gelatin_gt_roll['y']*0.01
gelatin_gt_roll['z'] = gelatin_gt_roll['z']*0.01

gelatin_gt_pitch['t'] = (gelatin_gt_pitch['t']-gelatin_gt_pitch['t'][0])
gelatin_gt_pitch['x'] = gelatin_gt_pitch['x']*0.01
gelatin_gt_pitch['y'] = gelatin_gt_pitch['y']*0.01
gelatin_gt_pitch['z'] = gelatin_gt_pitch['z']*0.01

gelatin_gt_yaw['t'] = (gelatin_gt_yaw['t']-gelatin_gt_yaw['t'][0])
gelatin_gt_yaw['x'] = gelatin_gt_yaw['x']*0.01
gelatin_gt_yaw['y'] = gelatin_gt_yaw['y']*0.01
gelatin_gt_yaw['z'] = gelatin_gt_yaw['z']*0.01

gelatin_edopt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_edopt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_edopt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_edopt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_edopt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_edopt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gelatin_edopt_trans_x_resampled_x = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['x'])
gelatin_edopt_trans_x_resampled_y = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['y'])
gelatin_edopt_trans_x_resampled_z = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['z'])
gelatin_edopt_trans_x_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['qx'])
gelatin_edopt_trans_x_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['qy'])
gelatin_edopt_trans_x_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['qz'])
gelatin_edopt_trans_x_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['qw'])

gelatin_edopt_trans_y_resampled_x = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['x'])
gelatin_edopt_trans_y_resampled_y = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['y'])
gelatin_edopt_trans_y_resampled_z = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['z'])
gelatin_edopt_trans_y_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['qx'])
gelatin_edopt_trans_y_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['qy'])
gelatin_edopt_trans_y_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['qz'])
gelatin_edopt_trans_y_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['qw'])

gelatin_edopt_trans_z_resampled_x = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['x'])
gelatin_edopt_trans_z_resampled_y = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['y'])
gelatin_edopt_trans_z_resampled_z = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['z'])
gelatin_edopt_trans_z_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['qx'])
gelatin_edopt_trans_z_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['qy'])
gelatin_edopt_trans_z_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['qz'])
gelatin_edopt_trans_z_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['qw'])

gelatin_edopt_roll_resampled_x = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_edopt_roll['t'], gelatin_edopt_roll['x'])
gelatin_edopt_roll_resampled_y = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_edopt_roll['t'], gelatin_edopt_roll['y'])
gelatin_edopt_roll_resampled_z = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_edopt_roll['t'], gelatin_edopt_roll['z'])
gelatin_edopt_roll_resampled_qx = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_edopt_roll['t'], gelatin_edopt_roll['qx'])
gelatin_edopt_roll_resampled_qy = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_edopt_roll['t'], gelatin_edopt_roll['qy'])
gelatin_edopt_roll_resampled_qz = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_edopt_roll['t'], gelatin_edopt_roll['qz'])
gelatin_edopt_roll_resampled_qw = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_edopt_roll['t'], gelatin_edopt_roll['qw'])

gelatin_edopt_pitch_resampled_x = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_edopt_pitch['t'], gelatin_edopt_pitch['x'])
gelatin_edopt_pitch_resampled_y = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_edopt_pitch['t'], gelatin_edopt_pitch['y'])
gelatin_edopt_pitch_resampled_z = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_edopt_pitch['t'], gelatin_edopt_pitch['z'])
gelatin_edopt_pitch_resampled_qx = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_edopt_pitch['t'], gelatin_edopt_pitch['qx'])
gelatin_edopt_pitch_resampled_qy = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_edopt_pitch['t'], gelatin_edopt_pitch['qy'])
gelatin_edopt_pitch_resampled_qz = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_edopt_pitch['t'], gelatin_edopt_pitch['qz'])
gelatin_edopt_pitch_resampled_qw = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_edopt_pitch['t'], gelatin_edopt_pitch['qw'])

gelatin_edopt_yaw_resampled_x = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_edopt_yaw['t'], gelatin_edopt_yaw['x'])
gelatin_edopt_yaw_resampled_y = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_edopt_yaw['t'], gelatin_edopt_yaw['y'])
gelatin_edopt_yaw_resampled_z = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_edopt_yaw['t'], gelatin_edopt_yaw['z'])
gelatin_edopt_yaw_resampled_qx = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_edopt_yaw['t'], gelatin_edopt_yaw['qx'])
gelatin_edopt_yaw_resampled_qy = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_edopt_yaw['t'], gelatin_edopt_yaw['qy'])
gelatin_edopt_yaw_resampled_qz = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_edopt_yaw['t'], gelatin_edopt_yaw['qz'])
gelatin_edopt_yaw_resampled_qw = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_edopt_yaw['t'], gelatin_edopt_yaw['qw'])

gelatin_edopt_trans_x_alpha,gelatin_edopt_trans_x_beta,gelatin_edopt_trans_x_gamma = quaternion_to_euler_angle(gelatin_edopt_trans_x['qw'], gelatin_edopt_trans_x['qx'], gelatin_edopt_trans_x['qy'], gelatin_edopt_trans_x['qz'])
gelatin_gt_trans_x_alpha,gelatin_gt_trans_x_beta,gelatin_gt_trans_x_gamma = quaternion_to_euler_angle(gelatin_gt_trans_x['qw'], gelatin_gt_trans_x['qx'], gelatin_gt_trans_x['qy'], gelatin_gt_trans_x['qz'])

gelatin_edopt_trans_x_alpha_cleaned = cleanEuler(gelatin_edopt_trans_x_alpha,0)
gelatin_edopt_trans_x_beta_cleaned = cleanEuler(gelatin_edopt_trans_x_beta,1)
gelatin_edopt_trans_x_gamma_cleaned = cleanEuler(gelatin_edopt_trans_x_gamma,2)

gelatin_gt_trans_x_alpha_cleaned = cleanEuler(gelatin_gt_trans_x_alpha,0)
gelatin_gt_trans_x_beta_cleaned = cleanEuler(gelatin_gt_trans_x_beta,1)
gelatin_gt_trans_x_gamma_cleaned = cleanEuler(gelatin_gt_trans_x_gamma,1)

gelatin_edopt_trans_y_alpha,gelatin_edopt_trans_y_beta,gelatin_edopt_trans_y_gamma = quaternion_to_euler_angle(gelatin_edopt_trans_y['qw'], gelatin_edopt_trans_y['qx'], gelatin_edopt_trans_y['qy'], gelatin_edopt_trans_y['qz'])
gelatin_gt_trans_y_alpha,gelatin_gt_trans_y_beta,gelatin_gt_trans_y_gamma = quaternion_to_euler_angle(gelatin_gt_trans_y['qw'], gelatin_gt_trans_y['qx'], gelatin_gt_trans_y['qy'], gelatin_gt_trans_y['qz'])

gelatin_edopt_trans_y_alpha_cleaned = cleanEuler(gelatin_edopt_trans_y_alpha,0)
gelatin_edopt_trans_y_beta_cleaned = cleanEuler(gelatin_edopt_trans_y_beta,1)
gelatin_edopt_trans_y_gamma_cleaned = cleanEuler(gelatin_edopt_trans_y_gamma,2)

gelatin_gt_trans_y_alpha_cleaned = cleanEuler(gelatin_gt_trans_y_alpha,0)
gelatin_gt_trans_y_beta_cleaned = cleanEuler(gelatin_gt_trans_y_beta,1)
gelatin_gt_trans_y_gamma_cleaned = cleanEuler(gelatin_gt_trans_y_gamma,2)

gelatin_edopt_trans_z_alpha,gelatin_edopt_trans_z_beta,gelatin_edopt_trans_z_gamma = quaternion_to_euler_angle(gelatin_edopt_trans_z['qw'], gelatin_edopt_trans_z['qx'], gelatin_edopt_trans_z['qy'], gelatin_edopt_trans_z['qz'])
gelatin_gt_trans_z_alpha,gelatin_gt_trans_z_beta,gelatin_gt_trans_z_gamma = quaternion_to_euler_angle(gelatin_gt_trans_z['qw'], gelatin_gt_trans_z['qx'], gelatin_gt_trans_z['qy'], gelatin_gt_trans_z['qz'])

gelatin_edopt_trans_z_alpha_cleaned = cleanEuler(gelatin_edopt_trans_z_alpha,0)
gelatin_edopt_trans_z_beta_cleaned = cleanEuler(gelatin_edopt_trans_z_beta,1)
gelatin_edopt_trans_z_gamma_cleaned = cleanEuler(gelatin_edopt_trans_z_gamma,2)

gelatin_gt_trans_z_alpha_cleaned = cleanEuler(gelatin_gt_trans_z_alpha,0)
gelatin_gt_trans_z_beta_cleaned = cleanEuler(gelatin_gt_trans_z_beta,1)
gelatin_gt_trans_z_gamma_cleaned = cleanEuler(gelatin_gt_trans_z_gamma,2)

gelatin_edopt_roll_alpha,gelatin_edopt_roll_beta,gelatin_edopt_roll_gamma = quaternion_to_euler_angle(gelatin_edopt_roll['qw'], gelatin_edopt_roll['qx'], gelatin_edopt_roll['qy'], gelatin_edopt_roll['qz'])
gelatin_gt_roll_alpha,gelatin_gt_roll_beta,gelatin_gt_roll_gamma = quaternion_to_euler_angle(gelatin_gt_roll['qw'], gelatin_gt_roll['qx'], gelatin_gt_roll['qy'], gelatin_gt_roll['qz'])

gelatin_edopt_roll_alpha_cleaned = cleanEuler(gelatin_edopt_roll_alpha,0)
gelatin_edopt_roll_beta_cleaned = cleanEuler(gelatin_edopt_roll_beta,1)
gelatin_edopt_roll_gamma_cleaned = cleanEuler(gelatin_edopt_roll_gamma,2)

gelatin_gt_roll_alpha_cleaned = cleanEuler(gelatin_gt_roll_alpha,0)
gelatin_gt_roll_beta_cleaned = cleanEuler(gelatin_gt_roll_beta,1)
gelatin_gt_roll_gamma_cleaned = cleanEuler(gelatin_gt_roll_gamma,2)

gelatin_edopt_pitch_alpha,gelatin_edopt_pitch_beta,gelatin_edopt_pitch_gamma = quaternion_to_euler_angle(gelatin_edopt_pitch['qw'], gelatin_edopt_pitch['qx'], gelatin_edopt_pitch['qy'], gelatin_edopt_pitch['qz'])
gelatin_gt_pitch_alpha,gelatin_gt_pitch_beta,gelatin_gt_pitch_gamma = quaternion_to_euler_angle(gelatin_gt_pitch['qw'], gelatin_gt_pitch['qx'], gelatin_gt_pitch['qy'], gelatin_gt_pitch['qz'])

gelatin_edopt_pitch_alpha_cleaned = cleanEuler(gelatin_edopt_pitch_alpha,0)
gelatin_edopt_pitch_beta_cleaned = cleanEuler(gelatin_edopt_pitch_beta,1)
gelatin_edopt_pitch_gamma_cleaned = cleanEuler(gelatin_edopt_pitch_gamma,2)

gelatin_gt_pitch_alpha_cleaned = cleanEuler(gelatin_gt_pitch_alpha,0)
gelatin_gt_pitch_beta_cleaned = cleanEuler(gelatin_gt_pitch_beta,1)
gelatin_gt_pitch_gamma_cleaned = cleanEuler(gelatin_gt_pitch_gamma,2)

gelatin_edopt_yaw_alpha,gelatin_edopt_yaw_beta,gelatin_edopt_yaw_gamma = quaternion_to_euler_angle(gelatin_edopt_yaw['qw'], gelatin_edopt_yaw['qx'], gelatin_edopt_yaw['qy'], gelatin_edopt_yaw['qz'])
gelatin_gt_yaw_alpha,gelatin_gt_yaw_beta,gelatin_gt_yaw_gamma = quaternion_to_euler_angle(gelatin_gt_yaw['qw'], gelatin_gt_yaw['qx'], gelatin_gt_yaw['qy'], gelatin_gt_yaw['qz'])

gelatin_edopt_yaw_alpha_cleaned = cleanEuler(gelatin_edopt_yaw_alpha,0)
gelatin_edopt_yaw_beta_cleaned = cleanEuler(gelatin_edopt_yaw_beta,1)
gelatin_edopt_yaw_gamma_cleaned = cleanEuler(gelatin_edopt_yaw_gamma,2)

gelatin_gt_yaw_alpha_cleaned = cleanEuler(gelatin_gt_yaw_alpha,0)
gelatin_gt_yaw_beta_cleaned = cleanEuler(gelatin_gt_yaw_beta,1)
gelatin_gt_yaw_gamma_cleaned = cleanEuler(gelatin_gt_yaw_gamma,2)

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18, 12)
# axs[0,0].plot(gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['x'], color=color_x, label='x')
# axs[0,0].plot(gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['y'], color=color_y, label='y')
# axs[0,0].plot(gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x['z'], color=color_z, label='z')
# axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['x'], color=color_x, ls='--')
# axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['y'], color=color_y, ls='--')
# axs[0,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x['z'], color=color_z, ls='--')
# axs[0,1].plot(gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['x'], color=color_x, label='x')
# axs[0,1].plot(gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['y'], color=color_y, label='y')
# axs[0,1].plot(gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y['z'], color=color_z, label='z')
# axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['x'], color=color_x, ls='--')
# axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['y'], color=color_y, ls='--')
# axs[0,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y['z'], color=color_z, ls='--')
# axs[0,2].plot(gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['x'], color=color_x, label='x')
# axs[0,2].plot(gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['y'], color=color_y, label='y')
# axs[0,2].plot(gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z['z'], color=color_z, label='z')
# axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['x'], color=color_x, ls='--')
# axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['y'], color=color_y, ls='--')
# axs[0,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z['z'], color=color_z, ls='--')
# axs[2,0].plot(gelatin_edopt_roll['t'], gelatin_edopt_roll['x'], color=color_x, label='x')
# axs[2,0].plot(gelatin_edopt_roll['t'], gelatin_edopt_roll['y'], color=color_y, label='y')
# axs[2,0].plot(gelatin_edopt_roll['t'], gelatin_edopt_roll['z'], color=color_z, label='z')
# axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['x'], color=color_x, ls='--')
# axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['y'], color=color_y, ls='--')
# axs[2,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll['z'], color=color_z, ls='--')
# axs[2,1].plot(gelatin_edopt_pitch['t'], gelatin_edopt_pitch['x'], color=color_x, label='x')
# axs[2,1].plot(gelatin_edopt_pitch['t'], gelatin_edopt_pitch['y'], color=color_y, label='y')
# axs[2,1].plot(gelatin_edopt_pitch['t'], gelatin_edopt_pitch['z'], color=color_z, label='z')
# axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['x'], color=color_x, ls='--')
# axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['y'], color=color_y, ls='--')
# axs[2,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch['z'], color=color_z, ls='--')
# axs[2,2].plot(gelatin_edopt_yaw['t'], gelatin_edopt_yaw['x'], color=color_x, label='x')
# axs[2,2].plot(gelatin_edopt_yaw['t'], gelatin_edopt_yaw['y'], color=color_y, label='y')
# axs[2,2].plot(gelatin_edopt_yaw['t'], gelatin_edopt_yaw['z'], color=color_z, label='z')
# axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['x'], color=color_x, ls='--')
# axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['y'], color=color_y, ls='--')
# axs[2,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw['z'], color=color_z, ls='--')
# axs[1,0].plot(gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x_alpha_cleaned, color=color_x, label='qx')
# axs[1,0].plot(gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x_beta_cleaned, color=color_y, label='qy')
# axs[1,0].plot(gelatin_edopt_trans_x['t'], gelatin_edopt_trans_x_gamma_cleaned, color=color_z, label='qz')
# axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
# axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
# axs[1,0].plot(gelatin_gt_trans_x['t'], gelatin_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
# axs[1,1].plot(gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y_alpha_cleaned, color=color_x, label='qx')
# axs[1,1].plot(gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y_beta_cleaned, color=color_y, label='qy')
# axs[1,1].plot(gelatin_edopt_trans_y['t'], gelatin_edopt_trans_y_gamma_cleaned, color=color_z, label='qz')
# axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
# axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
# axs[1,1].plot(gelatin_gt_trans_y['t'], gelatin_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
# axs[1,2].plot(gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z_alpha_cleaned, color=color_x, label='qx')
# axs[1,2].plot(gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z_beta_cleaned, color=color_y, label='qy')
# axs[1,2].plot(gelatin_edopt_trans_z['t'], gelatin_edopt_trans_z_gamma_cleaned, color=color_z, label='qz')
# axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
# axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
# axs[1,2].plot(gelatin_gt_trans_z['t'], gelatin_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
# axs[3,0].plot(gelatin_edopt_roll['t'], gelatin_edopt_roll_alpha_cleaned, color=color_x, label='qx')
# axs[3,0].plot(gelatin_edopt_roll['t'], gelatin_edopt_roll_beta_cleaned, color=color_y, label='qy')
# axs[3,0].plot(gelatin_edopt_roll['t'], gelatin_edopt_roll_gamma_cleaned, color=color_z, label='qz')
# axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_alpha_cleaned, color=color_x, ls = '--')
# axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_beta_cleaned, color=color_y, ls = '--')
# axs[3,0].plot(gelatin_gt_roll['t'], gelatin_gt_roll_gamma_cleaned, color=color_z, ls = '--')
# axs[3,1].plot(gelatin_edopt_pitch['t'], gelatin_edopt_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - gelatin_edopt")
# axs[3,1].plot(gelatin_edopt_pitch['t'], gelatin_edopt_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - gelatin_edopt")
# axs[3,1].plot(gelatin_edopt_pitch['t'], gelatin_edopt_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - gelatin_edopt")
# axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - gelatin_gt")
# axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - gelatin_gt")
# axs[3,1].plot(gelatin_gt_pitch['t'], gelatin_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - gelatin_gt")
# axs[3,2].plot(gelatin_edopt_yaw['t'], gelatin_edopt_yaw_alpha_cleaned, color=color_x, label='x or roll')
# axs[3,2].plot(gelatin_edopt_yaw['t'], gelatin_edopt_yaw_beta_cleaned, color=color_y, label='y or pitch')
# axs[3,2].plot(gelatin_edopt_yaw['t'], gelatin_edopt_yaw_gamma_cleaned, color=color_z, label='z or yaw')
# axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
# axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_beta_cleaned, color=color_y, ls = '--')
# axs[3,2].plot(gelatin_gt_yaw['t'], gelatin_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
# # for i in range(0, 4):
# #     for j in range(0, 3):
# #         axs[i,j].set_xlim([-2, 10])
# axs[0,0].set_ylim(-0.3,  1)
# axs[0,1].set_ylim(-0.3,  1)
# axs[0,2].set_ylim(-0.3,  1)
# axs[2,0].set_ylim(-0.1,  0.9)
# axs[2,1].set_ylim(-0.1,  0.9)
# axs[2,2].set_ylim(-0.1,  0.9)
# axs[1,0].set_ylim(-120,  200)
# axs[1,1].set_ylim(-120,  200)
# axs[1,2].set_ylim(-120,  200)
# axs[3,0].set_ylim(-200,  300)
# axs[3,1].set_ylim(-200,  300)
# axs[3,2].set_ylim(-200,  300)
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
# axs[0,0].set(ylabel='Position [m]')
# axs[1,0].set(ylabel='Euler angles [deg]')
# axs[2,0].set(ylabel='Position [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

gelatin_edopt_error_trans_x = computeEuclideanDistance(gelatin_gt_trans_x['x'], gelatin_gt_trans_x['y'], gelatin_gt_trans_x['z'], gelatin_edopt_trans_x_resampled_x, gelatin_edopt_trans_x_resampled_y, gelatin_edopt_trans_x_resampled_z)
gelatin_edopt_error_trans_y = computeEuclideanDistance(gelatin_gt_trans_y['x'], gelatin_gt_trans_y['y'], gelatin_gt_trans_y['z'], gelatin_edopt_trans_y_resampled_x, gelatin_edopt_trans_y_resampled_y, gelatin_edopt_trans_y_resampled_z)
gelatin_edopt_error_trans_z = computeEuclideanDistance(gelatin_gt_trans_z['x'], gelatin_gt_trans_z['y'], gelatin_gt_trans_z['z'], gelatin_edopt_trans_z_resampled_x, gelatin_edopt_trans_z_resampled_y, gelatin_edopt_trans_z_resampled_z)
gelatin_edopt_error_roll = computeEuclideanDistance(gelatin_gt_roll['x'], gelatin_gt_roll['y'], gelatin_gt_roll['z'], gelatin_edopt_roll_resampled_x, gelatin_edopt_roll_resampled_y, gelatin_edopt_roll_resampled_z)
gelatin_edopt_error_pitch = computeEuclideanDistance(gelatin_gt_pitch['x'], gelatin_gt_pitch['y'], gelatin_gt_pitch['z'], gelatin_edopt_pitch_resampled_x, gelatin_edopt_pitch_resampled_y, gelatin_edopt_pitch_resampled_z)
gelatin_edopt_error_yaw = computeEuclideanDistance(gelatin_gt_yaw['x'], gelatin_gt_yaw['y'], gelatin_gt_yaw['z'], gelatin_edopt_yaw_resampled_x, gelatin_edopt_yaw_resampled_y, gelatin_edopt_yaw_resampled_z)

gelatin_edopt_q_angle_trans_x = computeQuaternionError(gelatin_edopt_trans_x_resampled_qx, gelatin_edopt_trans_x_resampled_qy, gelatin_edopt_trans_x_resampled_qz, gelatin_edopt_trans_x_resampled_qw, gelatin_gt_trans_x['qx'], gelatin_gt_trans_x['qy'], gelatin_gt_trans_x['qz'], gelatin_gt_trans_x['qw'])
gelatin_edopt_q_angle_trans_y = computeQuaternionError(gelatin_edopt_trans_y_resampled_qx, gelatin_edopt_trans_y_resampled_qy, gelatin_edopt_trans_y_resampled_qz, gelatin_edopt_trans_y_resampled_qw, gelatin_gt_trans_y['qx'], gelatin_gt_trans_y['qy'], gelatin_gt_trans_y['qz'], gelatin_gt_trans_y['qw'])
gelatin_edopt_q_angle_trans_z = computeQuaternionError(gelatin_edopt_trans_z_resampled_qx, gelatin_edopt_trans_z_resampled_qy, gelatin_edopt_trans_z_resampled_qz, gelatin_edopt_trans_z_resampled_qw, gelatin_gt_trans_z['qx'], gelatin_gt_trans_z['qy'], gelatin_gt_trans_z['qz'], gelatin_gt_trans_z['qw'])
gelatin_edopt_q_angle_roll = computeQuaternionError(gelatin_edopt_roll_resampled_qx, gelatin_edopt_roll_resampled_qy, gelatin_edopt_roll_resampled_qz, gelatin_edopt_roll_resampled_qw, gelatin_gt_roll['qx'], gelatin_gt_roll['qy'], gelatin_gt_roll['qz'], gelatin_gt_roll['qw'])
gelatin_edopt_q_angle_pitch = computeQuaternionError(gelatin_edopt_pitch_resampled_qx, gelatin_edopt_pitch_resampled_qy, gelatin_edopt_pitch_resampled_qz, gelatin_edopt_pitch_resampled_qw, gelatin_gt_pitch['qx'], gelatin_gt_pitch['qy'], gelatin_gt_pitch['qz'], gelatin_gt_pitch['qw'])
gelatin_edopt_q_angle_yaw = computeQuaternionError(gelatin_edopt_yaw_resampled_qx, gelatin_edopt_yaw_resampled_qy, gelatin_edopt_yaw_resampled_qz, gelatin_edopt_yaw_resampled_qw, gelatin_gt_yaw['qx'], gelatin_gt_yaw['qy'], gelatin_gt_yaw['qz'], gelatin_gt_yaw['qw'])

gelatin_edopt_position_errors = np.concatenate((gelatin_edopt_error_trans_x, gelatin_edopt_error_trans_y, gelatin_edopt_error_trans_z, gelatin_edopt_error_roll, gelatin_edopt_error_pitch, gelatin_edopt_error_yaw))
gelatin_edopt_rotation_errors = np.concatenate((gelatin_edopt_q_angle_trans_x, gelatin_edopt_q_angle_trans_y, gelatin_edopt_q_angle_trans_z, gelatin_edopt_q_angle_roll, gelatin_edopt_q_angle_pitch, gelatin_edopt_q_angle_yaw))


# ---------------------------------------------------------------------------  MUSTARD  ---------------------------------------------------------------------------------------

filePath_dataset = '/home/luna/shared/data/6-DOF-Objects/results_icra_2024/mustard/'
mustard_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'gt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'gt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'gt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'gt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'gt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'gt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

mustard_gt_trans_x['t'] = (mustard_gt_trans_x['t']-mustard_gt_trans_x['t'][0])
mustard_gt_trans_x['x'] = mustard_gt_trans_x['x']*0.01
mustard_gt_trans_x['y'] = mustard_gt_trans_x['y']*0.01
mustard_gt_trans_x['z'] = mustard_gt_trans_x['z']*0.01

mustard_gt_trans_y['t'] = (mustard_gt_trans_y['t']-mustard_gt_trans_y['t'][0])
mustard_gt_trans_y['x'] = mustard_gt_trans_y['x']*0.01
mustard_gt_trans_y['y'] = mustard_gt_trans_y['y']*0.01
mustard_gt_trans_y['z'] = mustard_gt_trans_y['z']*0.01

mustard_gt_trans_z['t'] = (mustard_gt_trans_z['t']-mustard_gt_trans_z['t'][0])
mustard_gt_trans_z['x'] = mustard_gt_trans_z['x']*0.01
mustard_gt_trans_z['y'] = mustard_gt_trans_z['y']*0.01
mustard_gt_trans_z['z'] = mustard_gt_trans_z['z']*0.01

mustard_gt_roll['t'] = (mustard_gt_roll['t']-mustard_gt_roll['t'][0])
mustard_gt_roll['x'] = mustard_gt_roll['x']*0.01
mustard_gt_roll['y'] = mustard_gt_roll['y']*0.01
mustard_gt_roll['z'] = mustard_gt_roll['z']*0.01

mustard_gt_pitch['t'] = (mustard_gt_pitch['t']-mustard_gt_pitch['t'][0])
mustard_gt_pitch['x'] = mustard_gt_pitch['x']*0.01
mustard_gt_pitch['y'] = mustard_gt_pitch['y']*0.01
mustard_gt_pitch['z'] = mustard_gt_pitch['z']*0.01

mustard_gt_yaw['t'] = (mustard_gt_yaw['t']-mustard_gt_yaw['t'][0])
mustard_gt_yaw['x'] = mustard_gt_yaw['x']*0.01
mustard_gt_yaw['y'] = mustard_gt_yaw['y']*0.01
mustard_gt_yaw['z'] = mustard_gt_yaw['z']*0.01

mustard_edopt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_edopt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_edopt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_edopt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_edopt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_edopt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

mustard_edopt_trans_x_resampled_x = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_edopt_trans_x['t'], mustard_edopt_trans_x['x'])
mustard_edopt_trans_x_resampled_y = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_edopt_trans_x['t'], mustard_edopt_trans_x['y'])
mustard_edopt_trans_x_resampled_z = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_edopt_trans_x['t'], mustard_edopt_trans_x['z'])
mustard_edopt_trans_x_resampled_qx = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_edopt_trans_x['t'], mustard_edopt_trans_x['qx'])
mustard_edopt_trans_x_resampled_qy = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_edopt_trans_x['t'], mustard_edopt_trans_x['qy'])
mustard_edopt_trans_x_resampled_qz = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_edopt_trans_x['t'], mustard_edopt_trans_x['qz'])
mustard_edopt_trans_x_resampled_qw = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_edopt_trans_x['t'], mustard_edopt_trans_x['qw'])

mustard_edopt_trans_y_resampled_x = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_edopt_trans_y['t'], mustard_edopt_trans_y['x'])
mustard_edopt_trans_y_resampled_y = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_edopt_trans_y['t'], mustard_edopt_trans_y['y'])
mustard_edopt_trans_y_resampled_z = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_edopt_trans_y['t'], mustard_edopt_trans_y['z'])
mustard_edopt_trans_y_resampled_qx = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_edopt_trans_y['t'], mustard_edopt_trans_y['qx'])
mustard_edopt_trans_y_resampled_qy = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_edopt_trans_y['t'], mustard_edopt_trans_y['qy'])
mustard_edopt_trans_y_resampled_qz = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_edopt_trans_y['t'], mustard_edopt_trans_y['qz'])
mustard_edopt_trans_y_resampled_qw = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_edopt_trans_y['t'], mustard_edopt_trans_y['qw'])

mustard_edopt_trans_z_resampled_x = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_edopt_trans_z['t'], mustard_edopt_trans_z['x'])
mustard_edopt_trans_z_resampled_y = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_edopt_trans_z['t'], mustard_edopt_trans_z['y'])
mustard_edopt_trans_z_resampled_z = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_edopt_trans_z['t'], mustard_edopt_trans_z['z'])
mustard_edopt_trans_z_resampled_qx = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_edopt_trans_z['t'], mustard_edopt_trans_z['qx'])
mustard_edopt_trans_z_resampled_qy = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_edopt_trans_z['t'], mustard_edopt_trans_z['qy'])
mustard_edopt_trans_z_resampled_qz = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_edopt_trans_z['t'], mustard_edopt_trans_z['qz'])
mustard_edopt_trans_z_resampled_qw = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_edopt_trans_z['t'], mustard_edopt_trans_z['qw'])

mustard_edopt_roll_resampled_x = resampling_by_interpolate(mustard_gt_roll['t'], mustard_edopt_roll['t'], mustard_edopt_roll['x'])
mustard_edopt_roll_resampled_y = resampling_by_interpolate(mustard_gt_roll['t'], mustard_edopt_roll['t'], mustard_edopt_roll['y'])
mustard_edopt_roll_resampled_z = resampling_by_interpolate(mustard_gt_roll['t'], mustard_edopt_roll['t'], mustard_edopt_roll['z'])
mustard_edopt_roll_resampled_qx = resampling_by_interpolate(mustard_gt_roll['t'], mustard_edopt_roll['t'], mustard_edopt_roll['qx'])
mustard_edopt_roll_resampled_qy = resampling_by_interpolate(mustard_gt_roll['t'], mustard_edopt_roll['t'], mustard_edopt_roll['qy'])
mustard_edopt_roll_resampled_qz = resampling_by_interpolate(mustard_gt_roll['t'], mustard_edopt_roll['t'], mustard_edopt_roll['qz'])
mustard_edopt_roll_resampled_qw = resampling_by_interpolate(mustard_gt_roll['t'], mustard_edopt_roll['t'], mustard_edopt_roll['qw'])

mustard_edopt_pitch_resampled_x = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_edopt_pitch['t'], mustard_edopt_pitch['x'])
mustard_edopt_pitch_resampled_y = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_edopt_pitch['t'], mustard_edopt_pitch['y'])
mustard_edopt_pitch_resampled_z = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_edopt_pitch['t'], mustard_edopt_pitch['z'])
mustard_edopt_pitch_resampled_qx = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_edopt_pitch['t'], mustard_edopt_pitch['qx'])
mustard_edopt_pitch_resampled_qy = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_edopt_pitch['t'], mustard_edopt_pitch['qy'])
mustard_edopt_pitch_resampled_qz = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_edopt_pitch['t'], mustard_edopt_pitch['qz'])
mustard_edopt_pitch_resampled_qw = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_edopt_pitch['t'], mustard_edopt_pitch['qw'])

mustard_edopt_yaw_resampled_x = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_edopt_yaw['t'], mustard_edopt_yaw['x'])
mustard_edopt_yaw_resampled_y = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_edopt_yaw['t'], mustard_edopt_yaw['y'])
mustard_edopt_yaw_resampled_z = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_edopt_yaw['t'], mustard_edopt_yaw['z'])
mustard_edopt_yaw_resampled_qx = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_edopt_yaw['t'], mustard_edopt_yaw['qx'])
mustard_edopt_yaw_resampled_qy = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_edopt_yaw['t'], mustard_edopt_yaw['qy'])
mustard_edopt_yaw_resampled_qz = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_edopt_yaw['t'], mustard_edopt_yaw['qz'])
mustard_edopt_yaw_resampled_qw = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_edopt_yaw['t'], mustard_edopt_yaw['qw'])

mustard_edopt_trans_x_alpha,mustard_edopt_trans_x_beta,mustard_edopt_trans_x_gamma = quaternion_to_euler_angle(mustard_edopt_trans_x['qw'], mustard_edopt_trans_x['qx'], mustard_edopt_trans_x['qy'], mustard_edopt_trans_x['qz'])
mustard_gt_trans_x_alpha,mustard_gt_trans_x_beta,mustard_gt_trans_x_gamma = quaternion_to_euler_angle(mustard_gt_trans_x['qw'], mustard_gt_trans_x['qx'], mustard_gt_trans_x['qy'], mustard_gt_trans_x['qz'])

mustard_edopt_trans_x_alpha_cleaned = cleanEuler(mustard_edopt_trans_x_alpha,0)
mustard_edopt_trans_x_beta_cleaned = cleanEuler(mustard_edopt_trans_x_beta,1)
mustard_edopt_trans_x_gamma_cleaned = cleanEuler(mustard_edopt_trans_x_gamma,2)

mustard_gt_trans_x_alpha_cleaned = cleanEuler(mustard_gt_trans_x_alpha,0)
mustard_gt_trans_x_beta_cleaned = cleanEuler(mustard_gt_trans_x_beta,1)
mustard_gt_trans_x_gamma_cleaned = cleanEuler(mustard_gt_trans_x_gamma,1)

mustard_edopt_trans_y_alpha,mustard_edopt_trans_y_beta,mustard_edopt_trans_y_gamma = quaternion_to_euler_angle(mustard_edopt_trans_y['qw'], mustard_edopt_trans_y['qx'], mustard_edopt_trans_y['qy'], mustard_edopt_trans_y['qz'])
mustard_gt_trans_y_alpha,mustard_gt_trans_y_beta,mustard_gt_trans_y_gamma = quaternion_to_euler_angle(mustard_gt_trans_y['qw'], mustard_gt_trans_y['qx'], mustard_gt_trans_y['qy'], mustard_gt_trans_y['qz'])

mustard_edopt_trans_y_alpha_cleaned = cleanEuler(mustard_edopt_trans_y_alpha,0)
mustard_edopt_trans_y_beta_cleaned = cleanEuler(mustard_edopt_trans_y_beta,1)
mustard_edopt_trans_y_gamma_cleaned = cleanEuler(mustard_edopt_trans_y_gamma,2)

mustard_gt_trans_y_alpha_cleaned = cleanEuler(mustard_gt_trans_y_alpha,0)
mustard_gt_trans_y_beta_cleaned = cleanEuler(mustard_gt_trans_y_beta,1)
mustard_gt_trans_y_gamma_cleaned = cleanEuler(mustard_gt_trans_y_gamma,2)

mustard_edopt_trans_z_alpha,mustard_edopt_trans_z_beta,mustard_edopt_trans_z_gamma = quaternion_to_euler_angle(mustard_edopt_trans_z['qw'], mustard_edopt_trans_z['qx'], mustard_edopt_trans_z['qy'], mustard_edopt_trans_z['qz'])
mustard_gt_trans_z_alpha,mustard_gt_trans_z_beta,mustard_gt_trans_z_gamma = quaternion_to_euler_angle(mustard_gt_trans_z['qw'], mustard_gt_trans_z['qx'], mustard_gt_trans_z['qy'], mustard_gt_trans_z['qz'])

mustard_edopt_trans_z_alpha_cleaned = cleanEuler(mustard_edopt_trans_z_alpha,0)
mustard_edopt_trans_z_beta_cleaned = cleanEuler(mustard_edopt_trans_z_beta,1)
mustard_edopt_trans_z_gamma_cleaned = cleanEuler(mustard_edopt_trans_z_gamma,2)

mustard_gt_trans_z_alpha_cleaned = cleanEuler(mustard_gt_trans_z_alpha,0)
mustard_gt_trans_z_beta_cleaned = cleanEuler(mustard_gt_trans_z_beta,1)
mustard_gt_trans_z_gamma_cleaned = cleanEuler(mustard_gt_trans_z_gamma,2)

mustard_edopt_roll_alpha,mustard_edopt_roll_beta,mustard_edopt_roll_gamma = quaternion_to_euler_angle(mustard_edopt_roll['qw'], mustard_edopt_roll['qx'], mustard_edopt_roll['qy'], mustard_edopt_roll['qz'])
mustard_gt_roll_alpha,mustard_gt_roll_beta,mustard_gt_roll_gamma = quaternion_to_euler_angle(mustard_gt_roll['qw'], mustard_gt_roll['qx'], mustard_gt_roll['qy'], mustard_gt_roll['qz'])

mustard_edopt_roll_alpha_cleaned = cleanEuler(mustard_edopt_roll_alpha,0)
mustard_edopt_roll_beta_cleaned = cleanEuler(mustard_edopt_roll_beta,1)
mustard_edopt_roll_gamma_cleaned = cleanEuler(mustard_edopt_roll_gamma,2)

mustard_gt_roll_alpha_cleaned = cleanEuler(mustard_gt_roll_alpha,0)
mustard_gt_roll_beta_cleaned = cleanEuler(mustard_gt_roll_beta,1)
mustard_gt_roll_gamma_cleaned = cleanEuler(mustard_gt_roll_gamma,2)

mustard_edopt_pitch_alpha,mustard_edopt_pitch_beta,mustard_edopt_pitch_gamma = quaternion_to_euler_angle(mustard_edopt_pitch['qw'], mustard_edopt_pitch['qx'], mustard_edopt_pitch['qy'], mustard_edopt_pitch['qz'])
mustard_gt_pitch_alpha,mustard_gt_pitch_beta,mustard_gt_pitch_gamma = quaternion_to_euler_angle(mustard_gt_pitch['qw'], mustard_gt_pitch['qx'], mustard_gt_pitch['qy'], mustard_gt_pitch['qz'])

mustard_edopt_pitch_alpha_cleaned = cleanEuler(mustard_edopt_pitch_alpha,0)
mustard_edopt_pitch_beta_cleaned = cleanEuler(mustard_edopt_pitch_beta,1)
mustard_edopt_pitch_gamma_cleaned = cleanEuler(mustard_edopt_pitch_gamma,2)

mustard_gt_pitch_alpha_cleaned = cleanEuler(mustard_gt_pitch_alpha,0)
mustard_gt_pitch_beta_cleaned = cleanEuler(mustard_gt_pitch_beta,1)
mustard_gt_pitch_gamma_cleaned = cleanEuler(mustard_gt_pitch_gamma,2)

mustard_edopt_yaw_alpha,mustard_edopt_yaw_beta,mustard_edopt_yaw_gamma = quaternion_to_euler_angle(mustard_edopt_yaw['qw'], mustard_edopt_yaw['qx'], mustard_edopt_yaw['qy'], mustard_edopt_yaw['qz'])
mustard_gt_yaw_alpha,mustard_gt_yaw_beta,mustard_gt_yaw_gamma = quaternion_to_euler_angle(mustard_gt_yaw['qw'], mustard_gt_yaw['qx'], mustard_gt_yaw['qy'], mustard_gt_yaw['qz'])

mustard_edopt_yaw_alpha_cleaned = cleanEuler(mustard_edopt_yaw_alpha,0)
mustard_edopt_yaw_beta_cleaned = cleanEuler(mustard_edopt_yaw_beta,1)
mustard_edopt_yaw_gamma_cleaned = cleanEuler(mustard_edopt_yaw_gamma,2)

mustard_gt_yaw_alpha_cleaned = cleanEuler(mustard_gt_yaw_alpha,0)
mustard_gt_yaw_beta_cleaned = cleanEuler(mustard_gt_yaw_beta,1)
mustard_gt_yaw_gamma_cleaned = cleanEuler(mustard_gt_yaw_gamma,2)

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18, 12)
# axs[0,0].plot(mustard_edopt_trans_x['t'], mustard_edopt_trans_x['x'], color=color_x, label='x')
# axs[0,0].plot(mustard_edopt_trans_x['t'], mustard_edopt_trans_x['y'], color=color_y, label='y')
# axs[0,0].plot(mustard_edopt_trans_x['t'], mustard_edopt_trans_x['z'], color=color_z, label='z')
# axs[0,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x['x'], color=color_x, ls='--')
# axs[0,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x['y'], color=color_y, ls='--')
# axs[0,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x['z'], color=color_z, ls='--')
# axs[0,1].plot(mustard_edopt_trans_y['t'], mustard_edopt_trans_y['x'], color=color_x, label='x')
# axs[0,1].plot(mustard_edopt_trans_y['t'], mustard_edopt_trans_y['y'], color=color_y, label='y')
# axs[0,1].plot(mustard_edopt_trans_y['t'], mustard_edopt_trans_y['z'], color=color_z, label='z')
# axs[0,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y['x'], color=color_x, ls='--')
# axs[0,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y['y'], color=color_y, ls='--')
# axs[0,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y['z'], color=color_z, ls='--')
# axs[0,2].plot(mustard_edopt_trans_z['t'], mustard_edopt_trans_z['x'], color=color_x, label='x')
# axs[0,2].plot(mustard_edopt_trans_z['t'], mustard_edopt_trans_z['y'], color=color_y, label='y')
# axs[0,2].plot(mustard_edopt_trans_z['t'], mustard_edopt_trans_z['z'], color=color_z, label='z')
# axs[0,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z['x'], color=color_x, ls='--')
# axs[0,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z['y'], color=color_y, ls='--')
# axs[0,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z['z'], color=color_z, ls='--')
# axs[2,0].plot(mustard_edopt_roll['t'], mustard_edopt_roll['x'], color=color_x, label='x')
# axs[2,0].plot(mustard_edopt_roll['t'], mustard_edopt_roll['y'], color=color_y, label='y')
# axs[2,0].plot(mustard_edopt_roll['t'], mustard_edopt_roll['z'], color=color_z, label='z')
# axs[2,0].plot(mustard_gt_roll['t'], mustard_gt_roll['x'], color=color_x, ls='--')
# axs[2,0].plot(mustard_gt_roll['t'], mustard_gt_roll['y'], color=color_y, ls='--')
# axs[2,0].plot(mustard_gt_roll['t'], mustard_gt_roll['z'], color=color_z, ls='--')
# axs[2,1].plot(mustard_edopt_pitch['t'], mustard_edopt_pitch['x'], color=color_x, label='x')
# axs[2,1].plot(mustard_edopt_pitch['t'], mustard_edopt_pitch['y'], color=color_y, label='y')
# axs[2,1].plot(mustard_edopt_pitch['t'], mustard_edopt_pitch['z'], color=color_z, label='z')
# axs[2,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch['x'], color=color_x, ls='--')
# axs[2,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch['y'], color=color_y, ls='--')
# axs[2,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch['z'], color=color_z, ls='--')
# axs[2,2].plot(mustard_edopt_yaw['t'], mustard_edopt_yaw['x'], color=color_x, label='x')
# axs[2,2].plot(mustard_edopt_yaw['t'], mustard_edopt_yaw['y'], color=color_y, label='y')
# axs[2,2].plot(mustard_edopt_yaw['t'], mustard_edopt_yaw['z'], color=color_z, label='z')
# axs[2,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw['x'], color=color_x, ls='--')
# axs[2,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw['y'], color=color_y, ls='--')
# axs[2,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw['z'], color=color_z, ls='--')
# axs[1,0].plot(mustard_edopt_trans_x['t'], mustard_edopt_trans_x_alpha_cleaned, color=color_x, label='qx')
# axs[1,0].plot(mustard_edopt_trans_x['t'], mustard_edopt_trans_x_beta_cleaned, color=color_y, label='qy')
# axs[1,0].plot(mustard_edopt_trans_x['t'], mustard_edopt_trans_x_gamma_cleaned, color=color_z, label='qz')
# axs[1,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
# axs[1,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
# axs[1,0].plot(mustard_gt_trans_x['t'], mustard_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
# axs[1,1].plot(mustard_edopt_trans_y['t'], mustard_edopt_trans_y_alpha_cleaned, color=color_x, label='qx')
# axs[1,1].plot(mustard_edopt_trans_y['t'], mustard_edopt_trans_y_beta_cleaned, color=color_y, label='qy')
# axs[1,1].plot(mustard_edopt_trans_y['t'], mustard_edopt_trans_y_gamma_cleaned, color=color_z, label='qz')
# axs[1,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
# axs[1,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
# axs[1,1].plot(mustard_gt_trans_y['t'], mustard_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
# axs[1,2].plot(mustard_edopt_trans_z['t'], mustard_edopt_trans_z_alpha_cleaned, color=color_x, label='qx')
# axs[1,2].plot(mustard_edopt_trans_z['t'], mustard_edopt_trans_z_beta_cleaned, color=color_y, label='qy')
# axs[1,2].plot(mustard_edopt_trans_z['t'], mustard_edopt_trans_z_gamma_cleaned, color=color_z, label='qz')
# axs[1,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
# axs[1,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
# axs[1,2].plot(mustard_gt_trans_z['t'], mustard_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
# axs[3,0].plot(mustard_edopt_roll['t'], mustard_edopt_roll_alpha_cleaned, color=color_x, label='qx')
# axs[3,0].plot(mustard_edopt_roll['t'], mustard_edopt_roll_beta_cleaned, color=color_y, label='qy')
# axs[3,0].plot(mustard_edopt_roll['t'], mustard_edopt_roll_gamma_cleaned, color=color_z, label='qz')
# axs[3,0].plot(mustard_gt_roll['t'], mustard_gt_roll_alpha_cleaned, color=color_x, ls = '--')
# axs[3,0].plot(mustard_gt_roll['t'], mustard_gt_roll_beta_cleaned, color=color_y, ls = '--')
# axs[3,0].plot(mustard_gt_roll['t'], mustard_gt_roll_gamma_cleaned, color=color_z, ls = '--')
# axs[3,1].plot(mustard_edopt_pitch['t'], mustard_edopt_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - mustard_edopt")
# axs[3,1].plot(mustard_edopt_pitch['t'], mustard_edopt_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - mustard_edopt")
# axs[3,1].plot(mustard_edopt_pitch['t'], mustard_edopt_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - mustard_edopt")
# axs[3,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - mustard_gt")
# axs[3,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - mustard_gt")
# axs[3,1].plot(mustard_gt_pitch['t'], mustard_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - mustard_gt")
# axs[3,2].plot(mustard_edopt_yaw['t'], mustard_edopt_yaw_alpha_cleaned, color=color_x, label='x or roll')
# axs[3,2].plot(mustard_edopt_yaw['t'], mustard_edopt_yaw_beta_cleaned, color=color_y, label='y or pitch')
# axs[3,2].plot(mustard_edopt_yaw['t'], mustard_edopt_yaw_gamma_cleaned, color=color_z, label='z or yaw')
# axs[3,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
# axs[3,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw_beta_cleaned, color=color_y, ls = '--')
# axs[3,2].plot(mustard_gt_yaw['t'], mustard_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
# # for i in range(0, 4):
# #     for j in range(0, 3):
# #         axs[i,j].set_xlim([-2, 50])
# axs[0,0].set_ylim(-0.3,  1)
# axs[0,1].set_ylim(-0.3,  1)
# axs[0,2].set_ylim(-0.3,  1)
# axs[2,0].set_ylim(-0.1,  0.9)
# axs[2,1].set_ylim(-0.1,  0.9)
# axs[2,2].set_ylim(-0.1,  0.9)
# axs[1,0].set_ylim(-120,  200)
# axs[1,1].set_ylim(-120,  200)
# axs[1,2].set_ylim(-120,  200)
# axs[3,0].set_ylim(-200,  300)
# axs[3,1].set_ylim(-200,  300)
# axs[3,2].set_ylim(-200,  300)
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
# axs[0,0].set(ylabel='Position [m]')
# axs[1,0].set(ylabel='Euler angles [deg]')
# axs[2,0].set(ylabel='Position [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

mustard_edopt_error_trans_x = computeEuclideanDistance(mustard_gt_trans_x['x'], mustard_gt_trans_x['y'], mustard_gt_trans_x['z'], mustard_edopt_trans_x_resampled_x, mustard_edopt_trans_x_resampled_y, mustard_edopt_trans_x_resampled_z)
mustard_edopt_error_trans_y = computeEuclideanDistance(mustard_gt_trans_y['x'], mustard_gt_trans_y['y'], mustard_gt_trans_y['z'], mustard_edopt_trans_y_resampled_x, mustard_edopt_trans_y_resampled_y, mustard_edopt_trans_y_resampled_z)
mustard_edopt_error_trans_z = computeEuclideanDistance(mustard_gt_trans_z['x'], mustard_gt_trans_z['y'], mustard_gt_trans_z['z'], mustard_edopt_trans_z_resampled_x, mustard_edopt_trans_z_resampled_y, mustard_edopt_trans_z_resampled_z)
mustard_edopt_error_roll = computeEuclideanDistance(mustard_gt_roll['x'], mustard_gt_roll['y'], mustard_gt_roll['z'], mustard_edopt_roll_resampled_x, mustard_edopt_roll_resampled_y, mustard_edopt_roll_resampled_z)
mustard_edopt_error_pitch = computeEuclideanDistance(mustard_gt_pitch['x'], mustard_gt_pitch['y'], mustard_gt_pitch['z'], mustard_edopt_pitch_resampled_x, mustard_edopt_pitch_resampled_y, mustard_edopt_pitch_resampled_z)
mustard_edopt_error_yaw = computeEuclideanDistance(mustard_gt_yaw['x'], mustard_gt_yaw['y'], mustard_gt_yaw['z'], mustard_edopt_yaw_resampled_x, mustard_edopt_yaw_resampled_y, mustard_edopt_yaw_resampled_z)

mustard_edopt_q_angle_trans_x = computeQuaternionError(mustard_edopt_trans_x_resampled_qx, mustard_edopt_trans_x_resampled_qy, mustard_edopt_trans_x_resampled_qz, mustard_edopt_trans_x_resampled_qw, mustard_gt_trans_x['qx'], mustard_gt_trans_x['qy'], mustard_gt_trans_x['qz'], mustard_gt_trans_x['qw'])
mustard_edopt_q_angle_trans_y = computeQuaternionError(mustard_edopt_trans_y_resampled_qx, mustard_edopt_trans_y_resampled_qy, mustard_edopt_trans_y_resampled_qz, mustard_edopt_trans_y_resampled_qw, mustard_gt_trans_y['qx'], mustard_gt_trans_y['qy'], mustard_gt_trans_y['qz'], mustard_gt_trans_y['qw'])
mustard_edopt_q_angle_trans_z = computeQuaternionError(mustard_edopt_trans_z_resampled_qx, mustard_edopt_trans_z_resampled_qy, mustard_edopt_trans_z_resampled_qz, mustard_edopt_trans_z_resampled_qw, mustard_gt_trans_z['qx'], mustard_gt_trans_z['qy'], mustard_gt_trans_z['qz'], mustard_gt_trans_z['qw'])
mustard_edopt_q_angle_roll = computeQuaternionError(mustard_edopt_roll_resampled_qx, mustard_edopt_roll_resampled_qy, mustard_edopt_roll_resampled_qz, mustard_edopt_roll_resampled_qw, mustard_gt_roll['qx'], mustard_gt_roll['qy'], mustard_gt_roll['qz'], mustard_gt_roll['qw'])
mustard_edopt_q_angle_pitch = computeQuaternionError(mustard_edopt_pitch_resampled_qx, mustard_edopt_pitch_resampled_qy, mustard_edopt_pitch_resampled_qz, mustard_edopt_pitch_resampled_qw, mustard_gt_pitch['qx'], mustard_gt_pitch['qy'], mustard_gt_pitch['qz'], mustard_gt_pitch['qw'])
mustard_edopt_q_angle_yaw = computeQuaternionError(mustard_edopt_yaw_resampled_qx, mustard_edopt_yaw_resampled_qy, mustard_edopt_yaw_resampled_qz, mustard_edopt_yaw_resampled_qw, mustard_gt_yaw['qx'], mustard_gt_yaw['qy'], mustard_gt_yaw['qz'], mustard_gt_yaw['qw'])

mustard_edopt_position_errors = np.concatenate((mustard_edopt_error_trans_x, mustard_edopt_error_trans_y, mustard_edopt_error_trans_z, mustard_edopt_error_roll, mustard_edopt_error_pitch, mustard_edopt_error_yaw))
mustard_edopt_rotation_errors = np.concatenate((mustard_edopt_q_angle_trans_x, mustard_edopt_q_angle_trans_y, mustard_edopt_q_angle_trans_z, mustard_edopt_q_angle_roll, mustard_edopt_q_angle_pitch, mustard_edopt_q_angle_yaw))


# ---------------------------------------------------------------------------  TOMATO  ---------------------------------------------------------------------------------------

filePath_dataset = '/home/luna/shared/data/6-DOF-Objects/results_icra_2024/tomato/'
tomato_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'gt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'gt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'gt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'gt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'gt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'gt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

tomato_gt_trans_x['t'] = (tomato_gt_trans_x['t']-tomato_gt_trans_x['t'][0])
tomato_gt_trans_x['x'] = tomato_gt_trans_x['x']*0.01
tomato_gt_trans_x['y'] = tomato_gt_trans_x['y']*0.01
tomato_gt_trans_x['z'] = tomato_gt_trans_x['z']*0.01

tomato_gt_trans_y['t'] = (tomato_gt_trans_y['t']-tomato_gt_trans_y['t'][0])
tomato_gt_trans_y['x'] = tomato_gt_trans_y['x']*0.01
tomato_gt_trans_y['y'] = tomato_gt_trans_y['y']*0.01
tomato_gt_trans_y['z'] = tomato_gt_trans_y['z']*0.01

tomato_gt_trans_z['t'] = (tomato_gt_trans_z['t']-tomato_gt_trans_z['t'][0])
tomato_gt_trans_z['x'] = tomato_gt_trans_z['x']*0.01
tomato_gt_trans_z['y'] = tomato_gt_trans_z['y']*0.01
tomato_gt_trans_z['z'] = tomato_gt_trans_z['z']*0.01

tomato_gt_roll['t'] = (tomato_gt_roll['t']-tomato_gt_roll['t'][0])
tomato_gt_roll['x'] = tomato_gt_roll['x']*0.01
tomato_gt_roll['y'] = tomato_gt_roll['y']*0.01
tomato_gt_roll['z'] = tomato_gt_roll['z']*0.01

tomato_gt_pitch['t'] = (tomato_gt_pitch['t']-tomato_gt_pitch['t'][0])
tomato_gt_pitch['x'] = tomato_gt_pitch['x']*0.01
tomato_gt_pitch['y'] = tomato_gt_pitch['y']*0.01
tomato_gt_pitch['z'] = tomato_gt_pitch['z']*0.01

tomato_gt_yaw['t'] = (tomato_gt_yaw['t']-tomato_gt_yaw['t'][0])
tomato_gt_yaw['x'] = tomato_gt_yaw['x']*0.01
tomato_gt_yaw['y'] = tomato_gt_yaw['y']*0.01
tomato_gt_yaw['z'] = tomato_gt_yaw['z']*0.01

tomato_edopt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_edopt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_edopt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_edopt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_edopt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_edopt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

tomato_edopt_trans_x_resampled_x = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_edopt_trans_x['t'], tomato_edopt_trans_x['x'])
tomato_edopt_trans_x_resampled_y = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_edopt_trans_x['t'], tomato_edopt_trans_x['y'])
tomato_edopt_trans_x_resampled_z = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_edopt_trans_x['t'], tomato_edopt_trans_x['z'])
tomato_edopt_trans_x_resampled_qx = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_edopt_trans_x['t'], tomato_edopt_trans_x['qx'])
tomato_edopt_trans_x_resampled_qy = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_edopt_trans_x['t'], tomato_edopt_trans_x['qy'])
tomato_edopt_trans_x_resampled_qz = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_edopt_trans_x['t'], tomato_edopt_trans_x['qz'])
tomato_edopt_trans_x_resampled_qw = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_edopt_trans_x['t'], tomato_edopt_trans_x['qw'])

tomato_edopt_trans_y_resampled_x = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_edopt_trans_y['t'], tomato_edopt_trans_y['x'])
tomato_edopt_trans_y_resampled_y = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_edopt_trans_y['t'], tomato_edopt_trans_y['y'])
tomato_edopt_trans_y_resampled_z = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_edopt_trans_y['t'], tomato_edopt_trans_y['z'])
tomato_edopt_trans_y_resampled_qx = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_edopt_trans_y['t'], tomato_edopt_trans_y['qx'])
tomato_edopt_trans_y_resampled_qy = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_edopt_trans_y['t'], tomato_edopt_trans_y['qy'])
tomato_edopt_trans_y_resampled_qz = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_edopt_trans_y['t'], tomato_edopt_trans_y['qz'])
tomato_edopt_trans_y_resampled_qw = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_edopt_trans_y['t'], tomato_edopt_trans_y['qw'])

tomato_edopt_trans_z_resampled_x = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_edopt_trans_z['t'], tomato_edopt_trans_z['x'])
tomato_edopt_trans_z_resampled_y = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_edopt_trans_z['t'], tomato_edopt_trans_z['y'])
tomato_edopt_trans_z_resampled_z = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_edopt_trans_z['t'], tomato_edopt_trans_z['z'])
tomato_edopt_trans_z_resampled_qx = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_edopt_trans_z['t'], tomato_edopt_trans_z['qx'])
tomato_edopt_trans_z_resampled_qy = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_edopt_trans_z['t'], tomato_edopt_trans_z['qy'])
tomato_edopt_trans_z_resampled_qz = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_edopt_trans_z['t'], tomato_edopt_trans_z['qz'])
tomato_edopt_trans_z_resampled_qw = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_edopt_trans_z['t'], tomato_edopt_trans_z['qw'])

tomato_edopt_roll_resampled_x = resampling_by_interpolate(tomato_gt_roll['t'], tomato_edopt_roll['t'], tomato_edopt_roll['x'])
tomato_edopt_roll_resampled_y = resampling_by_interpolate(tomato_gt_roll['t'], tomato_edopt_roll['t'], tomato_edopt_roll['y'])
tomato_edopt_roll_resampled_z = resampling_by_interpolate(tomato_gt_roll['t'], tomato_edopt_roll['t'], tomato_edopt_roll['z'])
tomato_edopt_roll_resampled_qx = resampling_by_interpolate(tomato_gt_roll['t'], tomato_edopt_roll['t'], tomato_edopt_roll['qx'])
tomato_edopt_roll_resampled_qy = resampling_by_interpolate(tomato_gt_roll['t'], tomato_edopt_roll['t'], tomato_edopt_roll['qy'])
tomato_edopt_roll_resampled_qz = resampling_by_interpolate(tomato_gt_roll['t'], tomato_edopt_roll['t'], tomato_edopt_roll['qz'])
tomato_edopt_roll_resampled_qw = resampling_by_interpolate(tomato_gt_roll['t'], tomato_edopt_roll['t'], tomato_edopt_roll['qw'])

tomato_edopt_pitch_resampled_x = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_edopt_pitch['t'], tomato_edopt_pitch['x'])
tomato_edopt_pitch_resampled_y = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_edopt_pitch['t'], tomato_edopt_pitch['y'])
tomato_edopt_pitch_resampled_z = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_edopt_pitch['t'], tomato_edopt_pitch['z'])
tomato_edopt_pitch_resampled_qx = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_edopt_pitch['t'], tomato_edopt_pitch['qx'])
tomato_edopt_pitch_resampled_qy = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_edopt_pitch['t'], tomato_edopt_pitch['qy'])
tomato_edopt_pitch_resampled_qz = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_edopt_pitch['t'], tomato_edopt_pitch['qz'])
tomato_edopt_pitch_resampled_qw = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_edopt_pitch['t'], tomato_edopt_pitch['qw'])

tomato_edopt_yaw_resampled_x = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_edopt_yaw['t'], tomato_edopt_yaw['x'])
tomato_edopt_yaw_resampled_y = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_edopt_yaw['t'], tomato_edopt_yaw['y'])
tomato_edopt_yaw_resampled_z = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_edopt_yaw['t'], tomato_edopt_yaw['z'])
tomato_edopt_yaw_resampled_qx = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_edopt_yaw['t'], tomato_edopt_yaw['qx'])
tomato_edopt_yaw_resampled_qy = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_edopt_yaw['t'], tomato_edopt_yaw['qy'])
tomato_edopt_yaw_resampled_qz = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_edopt_yaw['t'], tomato_edopt_yaw['qz'])
tomato_edopt_yaw_resampled_qw = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_edopt_yaw['t'], tomato_edopt_yaw['qw'])

tomato_edopt_trans_x_alpha,tomato_edopt_trans_x_beta,tomato_edopt_trans_x_gamma = quaternion_to_euler_angle(tomato_edopt_trans_x['qw'], tomato_edopt_trans_x['qx'], tomato_edopt_trans_x['qy'], tomato_edopt_trans_x['qz'])
tomato_gt_trans_x_alpha,tomato_gt_trans_x_beta,tomato_gt_trans_x_gamma = quaternion_to_euler_angle(tomato_gt_trans_x['qw'], tomato_gt_trans_x['qx'], tomato_gt_trans_x['qy'], tomato_gt_trans_x['qz'])

tomato_edopt_trans_x_alpha_cleaned = cleanEuler(tomato_edopt_trans_x_alpha,0)
tomato_edopt_trans_x_beta_cleaned = cleanEuler(tomato_edopt_trans_x_beta,1)
tomato_edopt_trans_x_gamma_cleaned = cleanEuler(tomato_edopt_trans_x_gamma,2)

tomato_gt_trans_x_alpha_cleaned = cleanEuler(tomato_gt_trans_x_alpha,0)
tomato_gt_trans_x_beta_cleaned = cleanEuler(tomato_gt_trans_x_beta,1)
tomato_gt_trans_x_gamma_cleaned = cleanEuler(tomato_gt_trans_x_gamma,1)

tomato_edopt_trans_y_alpha,tomato_edopt_trans_y_beta,tomato_edopt_trans_y_gamma = quaternion_to_euler_angle(tomato_edopt_trans_y['qw'], tomato_edopt_trans_y['qx'], tomato_edopt_trans_y['qy'], tomato_edopt_trans_y['qz'])
tomato_gt_trans_y_alpha,tomato_gt_trans_y_beta,tomato_gt_trans_y_gamma = quaternion_to_euler_angle(tomato_gt_trans_y['qw'], tomato_gt_trans_y['qx'], tomato_gt_trans_y['qy'], tomato_gt_trans_y['qz'])

tomato_edopt_trans_y_alpha_cleaned = cleanEuler(tomato_edopt_trans_y_alpha,0)
tomato_edopt_trans_y_beta_cleaned = cleanEuler(tomato_edopt_trans_y_beta,1)
tomato_edopt_trans_y_gamma_cleaned = cleanEuler(tomato_edopt_trans_y_gamma,2)

tomato_gt_trans_y_alpha_cleaned = cleanEuler(tomato_gt_trans_y_alpha,0)
tomato_gt_trans_y_beta_cleaned = cleanEuler(tomato_gt_trans_y_beta,1)
tomato_gt_trans_y_gamma_cleaned = cleanEuler(tomato_gt_trans_y_gamma,2)

tomato_edopt_trans_z_alpha,tomato_edopt_trans_z_beta,tomato_edopt_trans_z_gamma = quaternion_to_euler_angle(tomato_edopt_trans_z['qw'], tomato_edopt_trans_z['qx'], tomato_edopt_trans_z['qy'], tomato_edopt_trans_z['qz'])
tomato_gt_trans_z_alpha,tomato_gt_trans_z_beta,tomato_gt_trans_z_gamma = quaternion_to_euler_angle(tomato_gt_trans_z['qw'], tomato_gt_trans_z['qx'], tomato_gt_trans_z['qy'], tomato_gt_trans_z['qz'])

tomato_edopt_trans_z_alpha_cleaned = cleanEuler(tomato_edopt_trans_z_alpha,0)
tomato_edopt_trans_z_beta_cleaned = cleanEuler(tomato_edopt_trans_z_beta,1)
tomato_edopt_trans_z_gamma_cleaned = cleanEuler(tomato_edopt_trans_z_gamma,2)

tomato_gt_trans_z_alpha_cleaned = cleanEuler(tomato_gt_trans_z_alpha,0)
tomato_gt_trans_z_beta_cleaned = cleanEuler(tomato_gt_trans_z_beta,1)
tomato_gt_trans_z_gamma_cleaned = cleanEuler(tomato_gt_trans_z_gamma,2)

tomato_edopt_roll_alpha,tomato_edopt_roll_beta,tomato_edopt_roll_gamma = quaternion_to_euler_angle(tomato_edopt_roll['qw'], tomato_edopt_roll['qx'], tomato_edopt_roll['qy'], tomato_edopt_roll['qz'])
tomato_gt_roll_alpha,tomato_gt_roll_beta,tomato_gt_roll_gamma = quaternion_to_euler_angle(tomato_gt_roll['qw'], tomato_gt_roll['qx'], tomato_gt_roll['qy'], tomato_gt_roll['qz'])

tomato_edopt_roll_alpha_cleaned = cleanEuler(tomato_edopt_roll_alpha,0)
tomato_edopt_roll_beta_cleaned = cleanEuler(tomato_edopt_roll_beta,1)
tomato_edopt_roll_gamma_cleaned = cleanEuler(tomato_edopt_roll_gamma,2)

tomato_gt_roll_alpha_cleaned = cleanEuler(tomato_gt_roll_alpha,0)
tomato_gt_roll_beta_cleaned = cleanEuler(tomato_gt_roll_beta,1)
tomato_gt_roll_gamma_cleaned = cleanEuler(tomato_gt_roll_gamma,2)

tomato_edopt_pitch_alpha,tomato_edopt_pitch_beta,tomato_edopt_pitch_gamma = quaternion_to_euler_angle(tomato_edopt_pitch['qw'], tomato_edopt_pitch['qx'], tomato_edopt_pitch['qy'], tomato_edopt_pitch['qz'])
tomato_gt_pitch_alpha,tomato_gt_pitch_beta,tomato_gt_pitch_gamma = quaternion_to_euler_angle(tomato_gt_pitch['qw'], tomato_gt_pitch['qx'], tomato_gt_pitch['qy'], tomato_gt_pitch['qz'])

tomato_edopt_pitch_alpha_cleaned = cleanEuler(tomato_edopt_pitch_alpha,0)
tomato_edopt_pitch_beta_cleaned = cleanEuler(tomato_edopt_pitch_beta,1)
tomato_edopt_pitch_gamma_cleaned = cleanEuler(tomato_edopt_pitch_gamma,2)

tomato_gt_pitch_alpha_cleaned = cleanEuler(tomato_gt_pitch_alpha,0)
tomato_gt_pitch_beta_cleaned = cleanEuler(tomato_gt_pitch_beta,1)
tomato_gt_pitch_gamma_cleaned = cleanEuler(tomato_gt_pitch_gamma,2)

tomato_edopt_yaw_alpha,tomato_edopt_yaw_beta,tomato_edopt_yaw_gamma = quaternion_to_euler_angle(tomato_edopt_yaw['qw'], tomato_edopt_yaw['qx'], tomato_edopt_yaw['qy'], tomato_edopt_yaw['qz'])
tomato_gt_yaw_alpha,tomato_gt_yaw_beta,tomato_gt_yaw_gamma = quaternion_to_euler_angle(tomato_gt_yaw['qw'], tomato_gt_yaw['qx'], tomato_gt_yaw['qy'], tomato_gt_yaw['qz'])

tomato_edopt_yaw_alpha_cleaned = cleanEuler(tomato_edopt_yaw_alpha,0)
tomato_edopt_yaw_beta_cleaned = cleanEuler(tomato_edopt_yaw_beta,1)
tomato_edopt_yaw_gamma_cleaned = cleanEuler(tomato_edopt_yaw_gamma,2)

tomato_gt_yaw_alpha_cleaned = cleanEuler(tomato_gt_yaw_alpha,0)
tomato_gt_yaw_beta_cleaned = cleanEuler(tomato_gt_yaw_beta,1)
tomato_gt_yaw_gamma_cleaned = cleanEuler(tomato_gt_yaw_gamma,2)

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18, 12)
# axs[0,0].plot(tomato_edopt_trans_x['t'], tomato_edopt_trans_x['x'], color=color_x, label='x')
# axs[0,0].plot(tomato_edopt_trans_x['t'], tomato_edopt_trans_x['y'], color=color_y, label='y')
# axs[0,0].plot(tomato_edopt_trans_x['t'], tomato_edopt_trans_x['z'], color=color_z, label='z')
# axs[0,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x['x'], color=color_x, ls='--')
# axs[0,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x['y'], color=color_y, ls='--')
# axs[0,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x['z'], color=color_z, ls='--')
# axs[0,1].plot(tomato_edopt_trans_y['t'], tomato_edopt_trans_y['x'], color=color_x, label='x')
# axs[0,1].plot(tomato_edopt_trans_y['t'], tomato_edopt_trans_y['y'], color=color_y, label='y')
# axs[0,1].plot(tomato_edopt_trans_y['t'], tomato_edopt_trans_y['z'], color=color_z, label='z')
# axs[0,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y['x'], color=color_x, ls='--')
# axs[0,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y['y'], color=color_y, ls='--')
# axs[0,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y['z'], color=color_z, ls='--')
# axs[0,2].plot(tomato_edopt_trans_z['t'], tomato_edopt_trans_z['x'], color=color_x, label='x')
# axs[0,2].plot(tomato_edopt_trans_z['t'], tomato_edopt_trans_z['y'], color=color_y, label='y')
# axs[0,2].plot(tomato_edopt_trans_z['t'], tomato_edopt_trans_z['z'], color=color_z, label='z')
# axs[0,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z['x'], color=color_x, ls='--')
# axs[0,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z['y'], color=color_y, ls='--')
# axs[0,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z['z'], color=color_z, ls='--')
# axs[2,0].plot(tomato_edopt_roll['t'], tomato_edopt_roll['x'], color=color_x, label='x')
# axs[2,0].plot(tomato_edopt_roll['t'], tomato_edopt_roll['y'], color=color_y, label='y')
# axs[2,0].plot(tomato_edopt_roll['t'], tomato_edopt_roll['z'], color=color_z, label='z')
# axs[2,0].plot(tomato_gt_roll['t'], tomato_gt_roll['x'], color=color_x, ls='--')
# axs[2,0].plot(tomato_gt_roll['t'], tomato_gt_roll['y'], color=color_y, ls='--')
# axs[2,0].plot(tomato_gt_roll['t'], tomato_gt_roll['z'], color=color_z, ls='--')
# axs[2,1].plot(tomato_edopt_pitch['t'], tomato_edopt_pitch['x'], color=color_x, label='x')
# axs[2,1].plot(tomato_edopt_pitch['t'], tomato_edopt_pitch['y'], color=color_y, label='y')
# axs[2,1].plot(tomato_edopt_pitch['t'], tomato_edopt_pitch['z'], color=color_z, label='z')
# axs[2,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch['x'], color=color_x, ls='--')
# axs[2,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch['y'], color=color_y, ls='--')
# axs[2,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch['z'], color=color_z, ls='--')
# axs[2,2].plot(tomato_edopt_yaw['t'], tomato_edopt_yaw['x'], color=color_x, label='x')
# axs[2,2].plot(tomato_edopt_yaw['t'], tomato_edopt_yaw['y'], color=color_y, label='y')
# axs[2,2].plot(tomato_edopt_yaw['t'], tomato_edopt_yaw['z'], color=color_z, label='z')
# axs[2,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw['x'], color=color_x, ls='--')
# axs[2,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw['y'], color=color_y, ls='--')
# axs[2,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw['z'], color=color_z, ls='--')
# axs[1,0].plot(tomato_edopt_trans_x['t'], tomato_edopt_trans_x_alpha_cleaned, color=color_x, label='qx')
# axs[1,0].plot(tomato_edopt_trans_x['t'], tomato_edopt_trans_x_beta_cleaned, color=color_y, label='qy')
# axs[1,0].plot(tomato_edopt_trans_x['t'], tomato_edopt_trans_x_gamma_cleaned, color=color_z, label='qz')
# axs[1,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
# axs[1,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
# axs[1,0].plot(tomato_gt_trans_x['t'], tomato_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
# axs[1,1].plot(tomato_edopt_trans_y['t'], tomato_edopt_trans_y_alpha_cleaned, color=color_x, label='qx')
# axs[1,1].plot(tomato_edopt_trans_y['t'], tomato_edopt_trans_y_beta_cleaned, color=color_y, label='qy')
# axs[1,1].plot(tomato_edopt_trans_y['t'], tomato_edopt_trans_y_gamma_cleaned, color=color_z, label='qz')
# axs[1,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
# axs[1,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
# axs[1,1].plot(tomato_gt_trans_y['t'], tomato_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
# axs[1,2].plot(tomato_edopt_trans_z['t'], tomato_edopt_trans_z_alpha_cleaned, color=color_x, label='qx')
# axs[1,2].plot(tomato_edopt_trans_z['t'], tomato_edopt_trans_z_beta_cleaned, color=color_y, label='qy')
# axs[1,2].plot(tomato_edopt_trans_z['t'], tomato_edopt_trans_z_gamma_cleaned, color=color_z, label='qz')
# axs[1,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
# axs[1,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
# axs[1,2].plot(tomato_gt_trans_z['t'], tomato_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
# axs[3,0].plot(tomato_edopt_roll['t'], tomato_edopt_roll_alpha_cleaned, color=color_x, label='qx')
# axs[3,0].plot(tomato_edopt_roll['t'], tomato_edopt_roll_beta_cleaned, color=color_y, label='qy')
# axs[3,0].plot(tomato_edopt_roll['t'], tomato_edopt_roll_gamma_cleaned, color=color_z, label='qz')
# axs[3,0].plot(tomato_gt_roll['t'], tomato_gt_roll_alpha_cleaned, color=color_x, ls = '--')
# axs[3,0].plot(tomato_gt_roll['t'], tomato_gt_roll_beta_cleaned, color=color_y, ls = '--')
# axs[3,0].plot(tomato_gt_roll['t'], tomato_gt_roll_gamma_cleaned, color=color_z, ls = '--')
# axs[3,1].plot(tomato_edopt_pitch['t'], tomato_edopt_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - tomato_edopt")
# axs[3,1].plot(tomato_edopt_pitch['t'], tomato_edopt_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - tomato_edopt")
# axs[3,1].plot(tomato_edopt_pitch['t'], tomato_edopt_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - tomato_edopt")
# axs[3,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - tomato_gt")
# axs[3,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - tomato_gt")
# axs[3,1].plot(tomato_gt_pitch['t'], tomato_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - tomato_gt")
# axs[3,2].plot(tomato_edopt_yaw['t'], tomato_edopt_yaw_alpha_cleaned, color=color_x, label='x or roll')
# axs[3,2].plot(tomato_edopt_yaw['t'], tomato_edopt_yaw_beta_cleaned, color=color_y, label='y or pitch')
# axs[3,2].plot(tomato_edopt_yaw['t'], tomato_edopt_yaw_gamma_cleaned, color=color_z, label='z or yaw')
# axs[3,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
# axs[3,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw_beta_cleaned, color=color_y, ls = '--')
# axs[3,2].plot(tomato_gt_yaw['t'], tomato_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
# # for i in range(0, 4):
# #     for j in range(0, 3):
# #         axs[i,j].set_xlim([-2, 50])
# axs[0,0].set_ylim(-0.3,  1)
# axs[0,1].set_ylim(-0.3,  1)
# axs[0,2].set_ylim(-0.3,  1)
# axs[2,0].set_ylim(-0.1,  0.9)
# axs[2,1].set_ylim(-0.1,  0.9)
# axs[2,2].set_ylim(-0.1,  0.9)
# axs[1,0].set_ylim(-120,  200)
# axs[1,1].set_ylim(-120,  200)
# axs[1,2].set_ylim(-120,  200)
# axs[3,0].set_ylim(-200,  300)
# axs[3,1].set_ylim(-200,  300)
# axs[3,2].set_ylim(-200,  300)
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
# axs[0,0].set(ylabel='Position [m]')
# axs[1,0].set(ylabel='Euler angles [deg]')
# axs[2,0].set(ylabel='Position [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

tomato_edopt_error_trans_x = computeEuclideanDistance(tomato_gt_trans_x['x'], tomato_gt_trans_x['y'], tomato_gt_trans_x['z'], tomato_edopt_trans_x_resampled_x, tomato_edopt_trans_x_resampled_y, tomato_edopt_trans_x_resampled_z)
tomato_edopt_error_trans_y = computeEuclideanDistance(tomato_gt_trans_y['x'], tomato_gt_trans_y['y'], tomato_gt_trans_y['z'], tomato_edopt_trans_y_resampled_x, tomato_edopt_trans_y_resampled_y, tomato_edopt_trans_y_resampled_z)
tomato_edopt_error_trans_z = computeEuclideanDistance(tomato_gt_trans_z['x'], tomato_gt_trans_z['y'], tomato_gt_trans_z['z'], tomato_edopt_trans_z_resampled_x, tomato_edopt_trans_z_resampled_y, tomato_edopt_trans_z_resampled_z)
tomato_edopt_error_roll = computeEuclideanDistance(tomato_gt_roll['x'], tomato_gt_roll['y'], tomato_gt_roll['z'], tomato_edopt_roll_resampled_x, tomato_edopt_roll_resampled_y, tomato_edopt_roll_resampled_z)
tomato_edopt_error_pitch = computeEuclideanDistance(tomato_gt_pitch['x'], tomato_gt_pitch['y'], tomato_gt_pitch['z'], tomato_edopt_pitch_resampled_x, tomato_edopt_pitch_resampled_y, tomato_edopt_pitch_resampled_z)
tomato_edopt_error_yaw = computeEuclideanDistance(tomato_gt_yaw['x'], tomato_gt_yaw['y'], tomato_gt_yaw['z'], tomato_edopt_yaw_resampled_x, tomato_edopt_yaw_resampled_y, tomato_edopt_yaw_resampled_z)

tomato_edopt_q_angle_trans_x = computeQuaternionError(tomato_edopt_trans_x_resampled_qx, tomato_edopt_trans_x_resampled_qy, tomato_edopt_trans_x_resampled_qz, tomato_edopt_trans_x_resampled_qw, tomato_gt_trans_x['qx'], tomato_gt_trans_x['qy'], tomato_gt_trans_x['qz'], tomato_gt_trans_x['qw'])
tomato_edopt_q_angle_trans_y = computeQuaternionError(tomato_edopt_trans_y_resampled_qx, tomato_edopt_trans_y_resampled_qy, tomato_edopt_trans_y_resampled_qz, tomato_edopt_trans_y_resampled_qw, tomato_gt_trans_y['qx'], tomato_gt_trans_y['qy'], tomato_gt_trans_y['qz'], tomato_gt_trans_y['qw'])
tomato_edopt_q_angle_trans_z = computeQuaternionError(tomato_edopt_trans_z_resampled_qx, tomato_edopt_trans_z_resampled_qy, tomato_edopt_trans_z_resampled_qz, tomato_edopt_trans_z_resampled_qw, tomato_gt_trans_z['qx'], tomato_gt_trans_z['qy'], tomato_gt_trans_z['qz'], tomato_gt_trans_z['qw'])
tomato_edopt_q_angle_roll = computeQuaternionError(tomato_edopt_roll_resampled_qx, tomato_edopt_roll_resampled_qy, tomato_edopt_roll_resampled_qz, tomato_edopt_roll_resampled_qw, tomato_gt_roll['qx'], tomato_gt_roll['qy'], tomato_gt_roll['qz'], tomato_gt_roll['qw'])
tomato_edopt_q_angle_pitch = computeQuaternionError(tomato_edopt_pitch_resampled_qx, tomato_edopt_pitch_resampled_qy, tomato_edopt_pitch_resampled_qz, tomato_edopt_pitch_resampled_qw, tomato_gt_pitch['qx'], tomato_gt_pitch['qy'], tomato_gt_pitch['qz'], tomato_gt_pitch['qw'])
tomato_edopt_q_angle_yaw = computeQuaternionError(tomato_edopt_yaw_resampled_qx, tomato_edopt_yaw_resampled_qy, tomato_edopt_yaw_resampled_qz, tomato_edopt_yaw_resampled_qw, tomato_gt_yaw['qx'], tomato_gt_yaw['qy'], tomato_gt_yaw['qz'], tomato_gt_yaw['qw'])

tomato_edopt_position_errors = np.concatenate((tomato_edopt_error_trans_x, tomato_edopt_error_trans_y, tomato_edopt_error_trans_z, tomato_edopt_error_roll, tomato_edopt_error_pitch, tomato_edopt_error_yaw))
tomato_edopt_rotation_errors = np.concatenate((tomato_edopt_q_angle_trans_x, tomato_edopt_q_angle_trans_y, tomato_edopt_q_angle_trans_z, tomato_edopt_q_angle_roll, tomato_edopt_q_angle_pitch, tomato_edopt_q_angle_yaw))

# ---------------------------------------------------------------------------  potted  ---------------------------------------------------------------------------------------

filePath_dataset = '/home/luna/shared/data/6-DOF-Objects/results_icra_2024/potted/'
potted_gt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'gt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'gt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'gt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'gt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'gt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_gt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'gt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

potted_gt_trans_x['t'] = (potted_gt_trans_x['t']-potted_gt_trans_x['t'][0])
potted_gt_trans_x['x'] = potted_gt_trans_x['x']*0.01
potted_gt_trans_x['y'] = potted_gt_trans_x['y']*0.01
potted_gt_trans_x['z'] = potted_gt_trans_x['z']*0.01

potted_gt_trans_y['t'] = (potted_gt_trans_y['t']-potted_gt_trans_y['t'][0])
potted_gt_trans_y['x'] = potted_gt_trans_y['x']*0.01
potted_gt_trans_y['y'] = potted_gt_trans_y['y']*0.01
potted_gt_trans_y['z'] = potted_gt_trans_y['z']*0.01

potted_gt_trans_z['t'] = (potted_gt_trans_z['t']-potted_gt_trans_z['t'][0])
potted_gt_trans_z['x'] = potted_gt_trans_z['x']*0.01
potted_gt_trans_z['y'] = potted_gt_trans_z['y']*0.01
potted_gt_trans_z['z'] = potted_gt_trans_z['z']*0.01

potted_gt_roll['t'] = (potted_gt_roll['t']-potted_gt_roll['t'][0])
potted_gt_roll['x'] = potted_gt_roll['x']*0.01
potted_gt_roll['y'] = potted_gt_roll['y']*0.01
potted_gt_roll['z'] = potted_gt_roll['z']*0.01

potted_gt_pitch['t'] = (potted_gt_pitch['t']-potted_gt_pitch['t'][0])
potted_gt_pitch['x'] = potted_gt_pitch['x']*0.01
potted_gt_pitch['y'] = potted_gt_pitch['y']*0.01
potted_gt_pitch['z'] = potted_gt_pitch['z']*0.01

potted_gt_yaw['t'] = (potted_gt_yaw['t']-potted_gt_yaw['t'][0])
potted_gt_yaw['x'] = potted_gt_yaw['x']*0.01
potted_gt_yaw['y'] = potted_gt_yaw['y']*0.01
potted_gt_yaw['z'] = potted_gt_yaw['z']*0.01

potted_edopt_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_edopt_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_edopt_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_edopt_roll = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_edopt_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_edopt_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'edopt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

potted_edopt_trans_x_resampled_x = resampling_by_interpolate(potted_gt_trans_x['t'], potted_edopt_trans_x['t'], potted_edopt_trans_x['x'])
potted_edopt_trans_x_resampled_y = resampling_by_interpolate(potted_gt_trans_x['t'], potted_edopt_trans_x['t'], potted_edopt_trans_x['y'])
potted_edopt_trans_x_resampled_z = resampling_by_interpolate(potted_gt_trans_x['t'], potted_edopt_trans_x['t'], potted_edopt_trans_x['z'])
potted_edopt_trans_x_resampled_qx = resampling_by_interpolate(potted_gt_trans_x['t'], potted_edopt_trans_x['t'], potted_edopt_trans_x['qx'])
potted_edopt_trans_x_resampled_qy = resampling_by_interpolate(potted_gt_trans_x['t'], potted_edopt_trans_x['t'], potted_edopt_trans_x['qy'])
potted_edopt_trans_x_resampled_qz = resampling_by_interpolate(potted_gt_trans_x['t'], potted_edopt_trans_x['t'], potted_edopt_trans_x['qz'])
potted_edopt_trans_x_resampled_qw = resampling_by_interpolate(potted_gt_trans_x['t'], potted_edopt_trans_x['t'], potted_edopt_trans_x['qw'])

potted_edopt_trans_y_resampled_x = resampling_by_interpolate(potted_gt_trans_y['t'], potted_edopt_trans_y['t'], potted_edopt_trans_y['x'])
potted_edopt_trans_y_resampled_y = resampling_by_interpolate(potted_gt_trans_y['t'], potted_edopt_trans_y['t'], potted_edopt_trans_y['y'])
potted_edopt_trans_y_resampled_z = resampling_by_interpolate(potted_gt_trans_y['t'], potted_edopt_trans_y['t'], potted_edopt_trans_y['z'])
potted_edopt_trans_y_resampled_qx = resampling_by_interpolate(potted_gt_trans_y['t'], potted_edopt_trans_y['t'], potted_edopt_trans_y['qx'])
potted_edopt_trans_y_resampled_qy = resampling_by_interpolate(potted_gt_trans_y['t'], potted_edopt_trans_y['t'], potted_edopt_trans_y['qy'])
potted_edopt_trans_y_resampled_qz = resampling_by_interpolate(potted_gt_trans_y['t'], potted_edopt_trans_y['t'], potted_edopt_trans_y['qz'])
potted_edopt_trans_y_resampled_qw = resampling_by_interpolate(potted_gt_trans_y['t'], potted_edopt_trans_y['t'], potted_edopt_trans_y['qw'])

potted_edopt_trans_z_resampled_x = resampling_by_interpolate(potted_gt_trans_z['t'], potted_edopt_trans_z['t'], potted_edopt_trans_z['x'])
potted_edopt_trans_z_resampled_y = resampling_by_interpolate(potted_gt_trans_z['t'], potted_edopt_trans_z['t'], potted_edopt_trans_z['y'])
potted_edopt_trans_z_resampled_z = resampling_by_interpolate(potted_gt_trans_z['t'], potted_edopt_trans_z['t'], potted_edopt_trans_z['z'])
potted_edopt_trans_z_resampled_qx = resampling_by_interpolate(potted_gt_trans_z['t'], potted_edopt_trans_z['t'], potted_edopt_trans_z['qx'])
potted_edopt_trans_z_resampled_qy = resampling_by_interpolate(potted_gt_trans_z['t'], potted_edopt_trans_z['t'], potted_edopt_trans_z['qy'])
potted_edopt_trans_z_resampled_qz = resampling_by_interpolate(potted_gt_trans_z['t'], potted_edopt_trans_z['t'], potted_edopt_trans_z['qz'])
potted_edopt_trans_z_resampled_qw = resampling_by_interpolate(potted_gt_trans_z['t'], potted_edopt_trans_z['t'], potted_edopt_trans_z['qw'])

potted_edopt_roll_resampled_x = resampling_by_interpolate(potted_gt_roll['t'], potted_edopt_roll['t'], potted_edopt_roll['x'])
potted_edopt_roll_resampled_y = resampling_by_interpolate(potted_gt_roll['t'], potted_edopt_roll['t'], potted_edopt_roll['y'])
potted_edopt_roll_resampled_z = resampling_by_interpolate(potted_gt_roll['t'], potted_edopt_roll['t'], potted_edopt_roll['z'])
potted_edopt_roll_resampled_qx = resampling_by_interpolate(potted_gt_roll['t'], potted_edopt_roll['t'], potted_edopt_roll['qx'])
potted_edopt_roll_resampled_qy = resampling_by_interpolate(potted_gt_roll['t'], potted_edopt_roll['t'], potted_edopt_roll['qy'])
potted_edopt_roll_resampled_qz = resampling_by_interpolate(potted_gt_roll['t'], potted_edopt_roll['t'], potted_edopt_roll['qz'])
potted_edopt_roll_resampled_qw = resampling_by_interpolate(potted_gt_roll['t'], potted_edopt_roll['t'], potted_edopt_roll['qw'])

potted_edopt_pitch_resampled_x = resampling_by_interpolate(potted_gt_pitch['t'], potted_edopt_pitch['t'], potted_edopt_pitch['x'])
potted_edopt_pitch_resampled_y = resampling_by_interpolate(potted_gt_pitch['t'], potted_edopt_pitch['t'], potted_edopt_pitch['y'])
potted_edopt_pitch_resampled_z = resampling_by_interpolate(potted_gt_pitch['t'], potted_edopt_pitch['t'], potted_edopt_pitch['z'])
potted_edopt_pitch_resampled_qx = resampling_by_interpolate(potted_gt_pitch['t'], potted_edopt_pitch['t'], potted_edopt_pitch['qx'])
potted_edopt_pitch_resampled_qy = resampling_by_interpolate(potted_gt_pitch['t'], potted_edopt_pitch['t'], potted_edopt_pitch['qy'])
potted_edopt_pitch_resampled_qz = resampling_by_interpolate(potted_gt_pitch['t'], potted_edopt_pitch['t'], potted_edopt_pitch['qz'])
potted_edopt_pitch_resampled_qw = resampling_by_interpolate(potted_gt_pitch['t'], potted_edopt_pitch['t'], potted_edopt_pitch['qw'])

potted_edopt_yaw_resampled_x = resampling_by_interpolate(potted_gt_yaw['t'], potted_edopt_yaw['t'], potted_edopt_yaw['x'])
potted_edopt_yaw_resampled_y = resampling_by_interpolate(potted_gt_yaw['t'], potted_edopt_yaw['t'], potted_edopt_yaw['y'])
potted_edopt_yaw_resampled_z = resampling_by_interpolate(potted_gt_yaw['t'], potted_edopt_yaw['t'], potted_edopt_yaw['z'])
potted_edopt_yaw_resampled_qx = resampling_by_interpolate(potted_gt_yaw['t'], potted_edopt_yaw['t'], potted_edopt_yaw['qx'])
potted_edopt_yaw_resampled_qy = resampling_by_interpolate(potted_gt_yaw['t'], potted_edopt_yaw['t'], potted_edopt_yaw['qy'])
potted_edopt_yaw_resampled_qz = resampling_by_interpolate(potted_gt_yaw['t'], potted_edopt_yaw['t'], potted_edopt_yaw['qz'])
potted_edopt_yaw_resampled_qw = resampling_by_interpolate(potted_gt_yaw['t'], potted_edopt_yaw['t'], potted_edopt_yaw['qw'])

potted_edopt_trans_x_alpha,potted_edopt_trans_x_beta,potted_edopt_trans_x_gamma = quaternion_to_euler_angle(potted_edopt_trans_x['qw'], potted_edopt_trans_x['qx'], potted_edopt_trans_x['qy'], potted_edopt_trans_x['qz'])
potted_gt_trans_x_alpha,potted_gt_trans_x_beta,potted_gt_trans_x_gamma = quaternion_to_euler_angle(potted_gt_trans_x['qw'], potted_gt_trans_x['qx'], potted_gt_trans_x['qy'], potted_gt_trans_x['qz'])

potted_edopt_trans_x_alpha_cleaned = cleanEuler(potted_edopt_trans_x_alpha,0)
potted_edopt_trans_x_beta_cleaned = cleanEuler(potted_edopt_trans_x_beta,1)
potted_edopt_trans_x_gamma_cleaned = cleanEuler(potted_edopt_trans_x_gamma,2)

potted_gt_trans_x_alpha_cleaned = cleanEuler(potted_gt_trans_x_alpha,0)
potted_gt_trans_x_beta_cleaned = cleanEuler(potted_gt_trans_x_beta,1)
potted_gt_trans_x_gamma_cleaned = cleanEuler(potted_gt_trans_x_gamma,1)

potted_edopt_trans_y_alpha,potted_edopt_trans_y_beta,potted_edopt_trans_y_gamma = quaternion_to_euler_angle(potted_edopt_trans_y['qw'], potted_edopt_trans_y['qx'], potted_edopt_trans_y['qy'], potted_edopt_trans_y['qz'])
potted_gt_trans_y_alpha,potted_gt_trans_y_beta,potted_gt_trans_y_gamma = quaternion_to_euler_angle(potted_gt_trans_y['qw'], potted_gt_trans_y['qx'], potted_gt_trans_y['qy'], potted_gt_trans_y['qz'])

potted_edopt_trans_y_alpha_cleaned = cleanEuler(potted_edopt_trans_y_alpha,0)
potted_edopt_trans_y_beta_cleaned = cleanEuler(potted_edopt_trans_y_beta,1)
potted_edopt_trans_y_gamma_cleaned = cleanEuler(potted_edopt_trans_y_gamma,2)

potted_gt_trans_y_alpha_cleaned = cleanEuler(potted_gt_trans_y_alpha,0)
potted_gt_trans_y_beta_cleaned = cleanEuler(potted_gt_trans_y_beta,1)
potted_gt_trans_y_gamma_cleaned = cleanEuler(potted_gt_trans_y_gamma,2)

potted_edopt_trans_z_alpha,potted_edopt_trans_z_beta,potted_edopt_trans_z_gamma = quaternion_to_euler_angle(potted_edopt_trans_z['qw'], potted_edopt_trans_z['qx'], potted_edopt_trans_z['qy'], potted_edopt_trans_z['qz'])
potted_gt_trans_z_alpha,potted_gt_trans_z_beta,potted_gt_trans_z_gamma = quaternion_to_euler_angle(potted_gt_trans_z['qw'], potted_gt_trans_z['qx'], potted_gt_trans_z['qy'], potted_gt_trans_z['qz'])

potted_edopt_trans_z_alpha_cleaned = cleanEuler(potted_edopt_trans_z_alpha,0)
potted_edopt_trans_z_beta_cleaned = cleanEuler(potted_edopt_trans_z_beta,1)
potted_edopt_trans_z_gamma_cleaned = cleanEuler(potted_edopt_trans_z_gamma,2)

potted_gt_trans_z_alpha_cleaned = cleanEuler(potted_gt_trans_z_alpha,0)
potted_gt_trans_z_beta_cleaned = cleanEuler(potted_gt_trans_z_beta,1)
potted_gt_trans_z_gamma_cleaned = cleanEuler(potted_gt_trans_z_gamma,2)

potted_edopt_roll_alpha,potted_edopt_roll_beta,potted_edopt_roll_gamma = quaternion_to_euler_angle(potted_edopt_roll['qw'], potted_edopt_roll['qx'], potted_edopt_roll['qy'], potted_edopt_roll['qz'])
potted_gt_roll_alpha,potted_gt_roll_beta,potted_gt_roll_gamma = quaternion_to_euler_angle(potted_gt_roll['qw'], potted_gt_roll['qx'], potted_gt_roll['qy'], potted_gt_roll['qz'])

potted_edopt_roll_alpha_cleaned = cleanEuler(potted_edopt_roll_alpha,0)
potted_edopt_roll_beta_cleaned = cleanEuler(potted_edopt_roll_beta,1)
potted_edopt_roll_gamma_cleaned = cleanEuler(potted_edopt_roll_gamma,2)

potted_gt_roll_alpha_cleaned = cleanEuler(potted_gt_roll_alpha,0)
potted_gt_roll_beta_cleaned = cleanEuler(potted_gt_roll_beta,1)
potted_gt_roll_gamma_cleaned = cleanEuler(potted_gt_roll_gamma,2)

potted_edopt_pitch_alpha,potted_edopt_pitch_beta,potted_edopt_pitch_gamma = quaternion_to_euler_angle(potted_edopt_pitch['qw'], potted_edopt_pitch['qx'], potted_edopt_pitch['qy'], potted_edopt_pitch['qz'])
potted_gt_pitch_alpha,potted_gt_pitch_beta,potted_gt_pitch_gamma = quaternion_to_euler_angle(potted_gt_pitch['qw'], potted_gt_pitch['qx'], potted_gt_pitch['qy'], potted_gt_pitch['qz'])

potted_edopt_pitch_alpha_cleaned = cleanEuler(potted_edopt_pitch_alpha,0)
potted_edopt_pitch_beta_cleaned = cleanEuler(potted_edopt_pitch_beta,1)
potted_edopt_pitch_gamma_cleaned = cleanEuler(potted_edopt_pitch_gamma,2)

potted_gt_pitch_alpha_cleaned = cleanEuler(potted_gt_pitch_alpha,0)
potted_gt_pitch_beta_cleaned = cleanEuler(potted_gt_pitch_beta,1)
potted_gt_pitch_gamma_cleaned = cleanEuler(potted_gt_pitch_gamma,2)

potted_edopt_yaw_alpha,potted_edopt_yaw_beta,potted_edopt_yaw_gamma = quaternion_to_euler_angle(potted_edopt_yaw['qw'], potted_edopt_yaw['qx'], potted_edopt_yaw['qy'], potted_edopt_yaw['qz'])
potted_gt_yaw_alpha,potted_gt_yaw_beta,potted_gt_yaw_gamma = quaternion_to_euler_angle(potted_gt_yaw['qw'], potted_gt_yaw['qx'], potted_gt_yaw['qy'], potted_gt_yaw['qz'])

potted_edopt_yaw_alpha_cleaned = cleanEuler(potted_edopt_yaw_alpha,0)
potted_edopt_yaw_beta_cleaned = cleanEuler(potted_edopt_yaw_beta,1)
potted_edopt_yaw_gamma_cleaned = cleanEuler(potted_edopt_yaw_gamma,2)

potted_gt_yaw_alpha_cleaned = cleanEuler(potted_gt_yaw_alpha,0)
potted_gt_yaw_beta_cleaned = cleanEuler(potted_gt_yaw_beta,1)
potted_gt_yaw_gamma_cleaned = cleanEuler(potted_gt_yaw_gamma,2)

# fig_summary, axs = plt.subplots(4,3)
# fig_summary.set_size_inches(18, 12)
# axs[0,0].plot(potted_edopt_trans_x['t'], potted_edopt_trans_x['x'], color=color_x, label='x')
# axs[0,0].plot(potted_edopt_trans_x['t'], potted_edopt_trans_x['y'], color=color_y, label='y')
# axs[0,0].plot(potted_edopt_trans_x['t'], potted_edopt_trans_x['z'], color=color_z, label='z')
# axs[0,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x['x'], color=color_x, ls='--')
# axs[0,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x['y'], color=color_y, ls='--')
# axs[0,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x['z'], color=color_z, ls='--')
# axs[0,1].plot(potted_edopt_trans_y['t'], potted_edopt_trans_y['x'], color=color_x, label='x')
# axs[0,1].plot(potted_edopt_trans_y['t'], potted_edopt_trans_y['y'], color=color_y, label='y')
# axs[0,1].plot(potted_edopt_trans_y['t'], potted_edopt_trans_y['z'], color=color_z, label='z')
# axs[0,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y['x'], color=color_x, ls='--')
# axs[0,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y['y'], color=color_y, ls='--')
# axs[0,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y['z'], color=color_z, ls='--')
# axs[0,2].plot(potted_edopt_trans_z['t'], potted_edopt_trans_z['x'], color=color_x, label='x')
# axs[0,2].plot(potted_edopt_trans_z['t'], potted_edopt_trans_z['y'], color=color_y, label='y')
# axs[0,2].plot(potted_edopt_trans_z['t'], potted_edopt_trans_z['z'], color=color_z, label='z')
# axs[0,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z['x'], color=color_x, ls='--')
# axs[0,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z['y'], color=color_y, ls='--')
# axs[0,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z['z'], color=color_z, ls='--')
# axs[2,0].plot(potted_edopt_roll['t'], potted_edopt_roll['x'], color=color_x, label='x')
# axs[2,0].plot(potted_edopt_roll['t'], potted_edopt_roll['y'], color=color_y, label='y')
# axs[2,0].plot(potted_edopt_roll['t'], potted_edopt_roll['z'], color=color_z, label='z')
# axs[2,0].plot(potted_gt_roll['t'], potted_gt_roll['x'], color=color_x, ls='--')
# axs[2,0].plot(potted_gt_roll['t'], potted_gt_roll['y'], color=color_y, ls='--')
# axs[2,0].plot(potted_gt_roll['t'], potted_gt_roll['z'], color=color_z, ls='--')
# axs[2,1].plot(potted_edopt_pitch['t'], potted_edopt_pitch['x'], color=color_x, label='x')
# axs[2,1].plot(potted_edopt_pitch['t'], potted_edopt_pitch['y'], color=color_y, label='y')
# axs[2,1].plot(potted_edopt_pitch['t'], potted_edopt_pitch['z'], color=color_z, label='z')
# axs[2,1].plot(potted_gt_pitch['t'], potted_gt_pitch['x'], color=color_x, ls='--')
# axs[2,1].plot(potted_gt_pitch['t'], potted_gt_pitch['y'], color=color_y, ls='--')
# axs[2,1].plot(potted_gt_pitch['t'], potted_gt_pitch['z'], color=color_z, ls='--')
# axs[2,2].plot(potted_edopt_yaw['t'], potted_edopt_yaw['x'], color=color_x, label='x')
# axs[2,2].plot(potted_edopt_yaw['t'], potted_edopt_yaw['y'], color=color_y, label='y')
# axs[2,2].plot(potted_edopt_yaw['t'], potted_edopt_yaw['z'], color=color_z, label='z')
# axs[2,2].plot(potted_gt_yaw['t'], potted_gt_yaw['x'], color=color_x, ls='--')
# axs[2,2].plot(potted_gt_yaw['t'], potted_gt_yaw['y'], color=color_y, ls='--')
# axs[2,2].plot(potted_gt_yaw['t'], potted_gt_yaw['z'], color=color_z, ls='--')
# axs[1,0].plot(potted_edopt_trans_x['t'], potted_edopt_trans_x_alpha_cleaned, color=color_x, label='qx')
# axs[1,0].plot(potted_edopt_trans_x['t'], potted_edopt_trans_x_beta_cleaned, color=color_y, label='qy')
# axs[1,0].plot(potted_edopt_trans_x['t'], potted_edopt_trans_x_gamma_cleaned, color=color_z, label='qz')
# axs[1,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x_alpha_cleaned, color=color_x, ls = '--')
# axs[1,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x_beta_cleaned, color=color_y, ls = '--')
# axs[1,0].plot(potted_gt_trans_x['t'], potted_gt_trans_x_gamma_cleaned, color=color_z, ls = '--')
# axs[1,1].plot(potted_edopt_trans_y['t'], potted_edopt_trans_y_alpha_cleaned, color=color_x, label='qx')
# axs[1,1].plot(potted_edopt_trans_y['t'], potted_edopt_trans_y_beta_cleaned, color=color_y, label='qy')
# axs[1,1].plot(potted_edopt_trans_y['t'], potted_edopt_trans_y_gamma_cleaned, color=color_z, label='qz')
# axs[1,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y_alpha_cleaned, color=color_x, ls = '--')
# axs[1,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y_beta_cleaned, color=color_y, ls = '--')
# axs[1,1].plot(potted_gt_trans_y['t'], potted_gt_trans_y_gamma_cleaned, color=color_z, ls = '--')
# axs[1,2].plot(potted_edopt_trans_z['t'], potted_edopt_trans_z_alpha_cleaned, color=color_x, label='qx')
# axs[1,2].plot(potted_edopt_trans_z['t'], potted_edopt_trans_z_beta_cleaned, color=color_y, label='qy')
# axs[1,2].plot(potted_edopt_trans_z['t'], potted_edopt_trans_z_gamma_cleaned, color=color_z, label='qz')
# axs[1,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z_alpha_cleaned, color=color_x, ls = '--')
# axs[1,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z_beta_cleaned, color=color_y, ls = '--')
# axs[1,2].plot(potted_gt_trans_z['t'], potted_gt_trans_z_gamma_cleaned, color=color_z, ls = '--')
# axs[3,0].plot(potted_edopt_roll['t'], potted_edopt_roll_alpha_cleaned, color=color_x, label='qx')
# axs[3,0].plot(potted_edopt_roll['t'], potted_edopt_roll_beta_cleaned, color=color_y, label='qy')
# axs[3,0].plot(potted_edopt_roll['t'], potted_edopt_roll_gamma_cleaned, color=color_z, label='qz')
# axs[3,0].plot(potted_gt_roll['t'], potted_gt_roll_alpha_cleaned, color=color_x, ls = '--')
# axs[3,0].plot(potted_gt_roll['t'], potted_gt_roll_beta_cleaned, color=color_y, ls = '--')
# axs[3,0].plot(potted_gt_roll['t'], potted_gt_roll_gamma_cleaned, color=color_z, ls = '--')
# axs[3,1].plot(potted_edopt_pitch['t'], potted_edopt_pitch_alpha_cleaned, color=color_x, label="x / "+r'$\alpha$'+" (pitch)"+ " - potted_edopt")
# axs[3,1].plot(potted_edopt_pitch['t'], potted_edopt_pitch_beta_cleaned, color=color_y, label="y / "+r'$\beta$'+" (yaw) - potted_edopt")
# axs[3,1].plot(potted_edopt_pitch['t'], potted_edopt_pitch_gamma_cleaned, color=color_z, label="z / "+r'$\gamma$'+" (roll) - potted_edopt")
# axs[3,1].plot(potted_gt_pitch['t'], potted_gt_pitch_alpha_cleaned, color=color_x, ls = '--', label="x / "+r'$\alpha$'+" (pitch)"+ " - potted_gt")
# axs[3,1].plot(potted_gt_pitch['t'], potted_gt_pitch_beta_cleaned, color=color_y, ls = '--',  label="y / "+r'$\beta$'+" (yaw) - potted_gt")
# axs[3,1].plot(potted_gt_pitch['t'], potted_gt_pitch_gamma_cleaned, color=color_z, ls = '--', label="z / "+r'$\gamma$'+" (roll) - potted_gt")
# axs[3,2].plot(potted_edopt_yaw['t'], potted_edopt_yaw_alpha_cleaned, color=color_x, label='x or roll')
# axs[3,2].plot(potted_edopt_yaw['t'], potted_edopt_yaw_beta_cleaned, color=color_y, label='y or pitch')
# axs[3,2].plot(potted_edopt_yaw['t'], potted_edopt_yaw_gamma_cleaned, color=color_z, label='z or yaw')
# axs[3,2].plot(potted_gt_yaw['t'], potted_gt_yaw_alpha_cleaned, color=color_x, ls = '--')
# axs[3,2].plot(potted_gt_yaw['t'], potted_gt_yaw_beta_cleaned, color=color_y, ls = '--')
# axs[3,2].plot(potted_gt_yaw['t'], potted_gt_yaw_gamma_cleaned, color=color_z, ls = '--')
# # for i in range(0, 4):
# #     for j in range(0, 3):
# #         axs[i,j].set_xlim([-2, 50])
# axs[0,0].set_ylim(-0.3,  1)
# axs[0,1].set_ylim(-0.3,  1)
# axs[0,2].set_ylim(-0.3,  1)
# axs[2,0].set_ylim(-0.1,  0.9)
# axs[2,1].set_ylim(-0.1,  0.9)
# axs[2,2].set_ylim(-0.1,  0.9)
# axs[1,0].set_ylim(-120,  200)
# axs[1,1].set_ylim(-120,  200)
# axs[1,2].set_ylim(-120,  200)
# axs[3,0].set_ylim(-200,  300)
# axs[3,1].set_ylim(-200,  300)
# axs[3,2].set_ylim(-200,  300)
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
# axs[0,0].set(ylabel='Position [m]')
# axs[1,0].set(ylabel='Euler angles [deg]')
# axs[2,0].set(ylabel='Position [m]')
# axs[3,0].set(xlabel='Time [s]', ylabel='Euler angles [deg]')
# axs[3,1].set(xlabel='Time [s]')
# axs[3,2].set(xlabel='Time [s]')
# axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, shadow=True, ncol=3)
# fig_summary.align_ylabels(axs[:, 0])
# fig_summary.subplots_adjust(wspace=0.05, hspace=0.06)
# plt.show()

potted_edopt_error_trans_x = computeEuclideanDistance(potted_gt_trans_x['x'], potted_gt_trans_x['y'], potted_gt_trans_x['z'], potted_edopt_trans_x_resampled_x, potted_edopt_trans_x_resampled_y, potted_edopt_trans_x_resampled_z)
potted_edopt_error_trans_y = computeEuclideanDistance(potted_gt_trans_y['x'], potted_gt_trans_y['y'], potted_gt_trans_y['z'], potted_edopt_trans_y_resampled_x, potted_edopt_trans_y_resampled_y, potted_edopt_trans_y_resampled_z)
potted_edopt_error_trans_z = computeEuclideanDistance(potted_gt_trans_z['x'], potted_gt_trans_z['y'], potted_gt_trans_z['z'], potted_edopt_trans_z_resampled_x, potted_edopt_trans_z_resampled_y, potted_edopt_trans_z_resampled_z)
potted_edopt_error_roll = computeEuclideanDistance(potted_gt_roll['x'], potted_gt_roll['y'], potted_gt_roll['z'], potted_edopt_roll_resampled_x, potted_edopt_roll_resampled_y, potted_edopt_roll_resampled_z)
potted_edopt_error_pitch = computeEuclideanDistance(potted_gt_pitch['x'], potted_gt_pitch['y'], potted_gt_pitch['z'], potted_edopt_pitch_resampled_x, potted_edopt_pitch_resampled_y, potted_edopt_pitch_resampled_z)
potted_edopt_error_yaw = computeEuclideanDistance(potted_gt_yaw['x'], potted_gt_yaw['y'], potted_gt_yaw['z'], potted_edopt_yaw_resampled_x, potted_edopt_yaw_resampled_y, potted_edopt_yaw_resampled_z)

potted_edopt_q_angle_trans_x = computeQuaternionError(potted_edopt_trans_x_resampled_qx, potted_edopt_trans_x_resampled_qy, potted_edopt_trans_x_resampled_qz, potted_edopt_trans_x_resampled_qw, potted_gt_trans_x['qx'], potted_gt_trans_x['qy'], potted_gt_trans_x['qz'], potted_gt_trans_x['qw'])
potted_edopt_q_angle_trans_y = computeQuaternionError(potted_edopt_trans_y_resampled_qx, potted_edopt_trans_y_resampled_qy, potted_edopt_trans_y_resampled_qz, potted_edopt_trans_y_resampled_qw, potted_gt_trans_y['qx'], potted_gt_trans_y['qy'], potted_gt_trans_y['qz'], potted_gt_trans_y['qw'])
potted_edopt_q_angle_trans_z = computeQuaternionError(potted_edopt_trans_z_resampled_qx, potted_edopt_trans_z_resampled_qy, potted_edopt_trans_z_resampled_qz, potted_edopt_trans_z_resampled_qw, potted_gt_trans_z['qx'], potted_gt_trans_z['qy'], potted_gt_trans_z['qz'], potted_gt_trans_z['qw'])
potted_edopt_q_angle_roll = computeQuaternionError(potted_edopt_roll_resampled_qx, potted_edopt_roll_resampled_qy, potted_edopt_roll_resampled_qz, potted_edopt_roll_resampled_qw, potted_gt_roll['qx'], potted_gt_roll['qy'], potted_gt_roll['qz'], potted_gt_roll['qw'])
potted_edopt_q_angle_pitch = computeQuaternionError(potted_edopt_pitch_resampled_qx, potted_edopt_pitch_resampled_qy, potted_edopt_pitch_resampled_qz, potted_edopt_pitch_resampled_qw, potted_gt_pitch['qx'], potted_gt_pitch['qy'], potted_gt_pitch['qz'], potted_gt_pitch['qw'])
potted_edopt_q_angle_yaw = computeQuaternionError(potted_edopt_yaw_resampled_qx, potted_edopt_yaw_resampled_qy, potted_edopt_yaw_resampled_qz, potted_edopt_yaw_resampled_qw, potted_gt_yaw['qx'], potted_gt_yaw['qy'], potted_gt_yaw['qz'], potted_gt_yaw['qw'])

potted_edopt_position_errors = np.concatenate((potted_edopt_error_trans_x, potted_edopt_error_trans_y, potted_edopt_error_trans_z, potted_edopt_error_roll, potted_edopt_error_pitch, potted_edopt_error_yaw))
potted_edopt_rotation_errors = np.concatenate((potted_edopt_q_angle_trans_x, potted_edopt_q_angle_trans_y, potted_edopt_q_angle_trans_z, potted_edopt_q_angle_roll, potted_edopt_q_angle_pitch, potted_edopt_q_angle_yaw))


labels = ['O5', 'O4', 'O3', 'O2', 'O1']
ticks=[0, 1, 2, 3, 4]
medianprops = dict(color='white')

rad_to_deg = 180/math.pi

all_objects_position_errors = [potted_edopt_position_errors, tomato_edopt_position_errors, mustard_edopt_position_errors, gelatin_edopt_position_errors, dragon_edopt_position_errors]
all_objects_rotation_errors = [potted_edopt_rotation_errors*rad_to_deg, tomato_edopt_rotation_errors*rad_to_deg, mustard_edopt_rotation_errors*rad_to_deg, gelatin_edopt_rotation_errors*rad_to_deg, dragon_edopt_rotation_errors*rad_to_deg]
# new_quart_array = np.array(quart_vec_pos).transpose

fig15, ax1 = plt.subplots(1,2)
fig15.set_size_inches(8, 6)
ax1[0].set_xlabel('Position error [m]', color='k')
ax1[1].set_xlabel('Rotation error [deg]', color='k')
res1 = ax1[0].boxplot(all_objects_position_errors, labels=labels, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
res2 = ax1[1].boxplot(all_objects_rotation_errors, vert=False, showfliers=False,
                    patch_artist=True, medianprops=medianprops)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res1[element], color='k')
    plt.setp(res2[element], color='k')
ax1[1].set_yticklabels([])
ax1[1].set_yticks([])
ax1[0].set_xlim(-0.001,  0.018)
ax1[1].set_xlim(-1,  17)
ax1[1].xaxis.set_major_locator(plt.MaxNLocator(4))
colors=[color_edopt, color_edopt, color_edopt, color_edopt, color_edopt]
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
fig15.subplots_adjust(wspace=0.1)
plt.show()