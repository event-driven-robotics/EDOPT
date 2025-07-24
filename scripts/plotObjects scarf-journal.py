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

color_eros = '#5A8CFF'
color_rgbde = '#FF4782'

color_potted = '#005175' 
color_mustard = '#08FBFB'
color_gelatin = '#08FBD3'
color_tomato = '#5770CB'
color_dragon = color_eros

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

filePath_dataset = '/data/scarf-journal/dragon/'

dragon_gt_trans_x_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_y_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_trans_z_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_roll_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_pitch_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_gt_yaw_new = np.genfromtxt(os.path.join(filePath_dataset, 'gt/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

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

dragon_gt_trans_x_new_alpha,dragon_gt_trans_x_new_beta,dragon_gt_trans_x_new_gamma = quaternion_to_euler_angle(dragon_gt_trans_x_new['qw'], dragon_gt_trans_x_new['qx'], dragon_gt_trans_x_new['qy'], dragon_gt_trans_x_new['qz'])
dragon_gt_trans_x_new_alpha_cleaned = cleanEuler(dragon_gt_trans_x_new_alpha,0)
dragon_gt_trans_x_new_beta_cleaned = cleanEuler(dragon_gt_trans_x_new_beta,1)
dragon_gt_trans_x_new_gamma_cleaned = cleanEuler(dragon_gt_trans_x_new_gamma,1)

dragon_gt_trans_y_new_alpha,dragon_gt_trans_y_new_beta,dragon_gt_trans_y_new_gamma = quaternion_to_euler_angle(dragon_gt_trans_y_new['qw'], dragon_gt_trans_y_new['qx'], dragon_gt_trans_y_new['qy'], dragon_gt_trans_y_new['qz'])
dragon_gt_trans_y_new_alpha_cleaned = cleanEuler(dragon_gt_trans_y_new_alpha,0)
dragon_gt_trans_y_new_beta_cleaned = cleanEuler(dragon_gt_trans_y_new_beta,1)
dragon_gt_trans_y_new_gamma_cleaned = cleanEuler(dragon_gt_trans_y_new_gamma,2)

dragon_gt_trans_z_new_alpha,dragon_gt_trans_z_new_beta,dragon_gt_trans_z_new_gamma = quaternion_to_euler_angle(dragon_gt_trans_z_new['qw'], dragon_gt_trans_z_new['qx'], dragon_gt_trans_z_new['qy'], dragon_gt_trans_z_new['qz'])
dragon_gt_trans_z_new_alpha_cleaned = cleanEuler(dragon_gt_trans_z_new_alpha,0)
dragon_gt_trans_z_new_beta_cleaned = cleanEuler(dragon_gt_trans_z_new_beta,1)
dragon_gt_trans_z_new_gamma_cleaned = cleanEuler(dragon_gt_trans_z_new_gamma,2)

dragon_gt_roll_new_alpha,dragon_gt_roll_new_beta,dragon_gt_roll_new_gamma = quaternion_to_euler_angle(dragon_gt_roll_new['qw'], dragon_gt_roll_new['qx'], dragon_gt_roll_new['qy'], dragon_gt_roll_new['qz'])
dragon_gt_roll_new_alpha_cleaned = cleanEuler(dragon_gt_roll_new_alpha,0)
dragon_gt_roll_new_beta_cleaned = cleanEuler(dragon_gt_roll_new_beta,1)
dragon_gt_roll_new_gamma_cleaned = cleanEuler(dragon_gt_roll_new_gamma,2)

dragon_gt_pitch_new_alpha,dragon_gt_pitch_new_beta,dragon_gt_pitch_new_gamma = quaternion_to_euler_angle(dragon_gt_pitch_new['qw'], dragon_gt_pitch_new['qx'], dragon_gt_pitch_new['qy'], dragon_gt_pitch_new['qz'])
dragon_gt_pitch_new_alpha_cleaned = cleanEuler(dragon_gt_pitch_new_alpha,0)
dragon_gt_pitch_new_beta_cleaned = cleanEuler(dragon_gt_pitch_new_beta,1)
dragon_gt_pitch_new_gamma_cleaned = cleanEuler(dragon_gt_pitch_new_gamma,2)

dragon_gt_yaw_new_alpha,dragon_gt_yaw_new_beta,dragon_gt_yaw_new_gamma = quaternion_to_euler_angle(dragon_gt_yaw_new['qw'], dragon_gt_yaw_new['qx'], dragon_gt_yaw_new['qy'], dragon_gt_yaw_new['qz'])
dragon_gt_yaw_new_alpha_cleaned = cleanEuler(dragon_gt_yaw_new_alpha,0)
dragon_gt_yaw_new_beta_cleaned = cleanEuler(dragon_gt_yaw_new_beta,1)
dragon_gt_yaw_new_gamma_cleaned = cleanEuler(dragon_gt_yaw_new_gamma,2)

dragon_eros_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_eros_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_eros_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_eros_roll = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_eros_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_eros_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

dragon_eros_trans_x_resampled_x = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_eros_trans_x['t'], dragon_eros_trans_x['x'])
dragon_eros_trans_x_resampled_y = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_eros_trans_x['t'], dragon_eros_trans_x['y'])
dragon_eros_trans_x_resampled_z = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_eros_trans_x['t'], dragon_eros_trans_x['z'])
dragon_eros_trans_x_resampled_qx = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_eros_trans_x['t'], dragon_eros_trans_x['qx'])
dragon_eros_trans_x_resampled_qy = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_eros_trans_x['t'], dragon_eros_trans_x['qy'])
dragon_eros_trans_x_resampled_qz = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_eros_trans_x['t'], dragon_eros_trans_x['qz'])
dragon_eros_trans_x_resampled_qw = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_eros_trans_x['t'], dragon_eros_trans_x['qw'])

dragon_eros_trans_y_resampled_x = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_eros_trans_y['t'], dragon_eros_trans_y['x'])
dragon_eros_trans_y_resampled_y = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_eros_trans_y['t'], dragon_eros_trans_y['y'])
dragon_eros_trans_y_resampled_z = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_eros_trans_y['t'], dragon_eros_trans_y['z'])
dragon_eros_trans_y_resampled_qx = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_eros_trans_y['t'], dragon_eros_trans_y['qx'])
dragon_eros_trans_y_resampled_qy = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_eros_trans_y['t'], dragon_eros_trans_y['qy'])
dragon_eros_trans_y_resampled_qz = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_eros_trans_y['t'], dragon_eros_trans_y['qz'])
dragon_eros_trans_y_resampled_qw = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_eros_trans_y['t'], dragon_eros_trans_y['qw'])

dragon_eros_trans_z_resampled_x = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_eros_trans_z['t'], dragon_eros_trans_z['x'])
dragon_eros_trans_z_resampled_y = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_eros_trans_z['t'], dragon_eros_trans_z['y'])
dragon_eros_trans_z_resampled_z = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_eros_trans_z['t'], dragon_eros_trans_z['z'])
dragon_eros_trans_z_resampled_qx = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_eros_trans_z['t'], dragon_eros_trans_z['qx'])
dragon_eros_trans_z_resampled_qy = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_eros_trans_z['t'], dragon_eros_trans_z['qy'])
dragon_eros_trans_z_resampled_qz = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_eros_trans_z['t'], dragon_eros_trans_z['qz'])
dragon_eros_trans_z_resampled_qw = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_eros_trans_z['t'], dragon_eros_trans_z['qw'])

dragon_eros_roll_resampled_x = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_eros_roll['t'], dragon_eros_roll['x'])
dragon_eros_roll_resampled_y = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_eros_roll['t'], dragon_eros_roll['y'])
dragon_eros_roll_resampled_z = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_eros_roll['t'], dragon_eros_roll['z'])
dragon_eros_roll_resampled_qx = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_eros_roll['t'], dragon_eros_roll['qx'])
dragon_eros_roll_resampled_qy = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_eros_roll['t'], dragon_eros_roll['qy'])
dragon_eros_roll_resampled_qz = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_eros_roll['t'], dragon_eros_roll['qz'])
dragon_eros_roll_resampled_qw = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_eros_roll['t'], dragon_eros_roll['qw'])

dragon_eros_pitch_resampled_x = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_eros_pitch['t'], dragon_eros_pitch['x'])
dragon_eros_pitch_resampled_y = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_eros_pitch['t'], dragon_eros_pitch['y'])
dragon_eros_pitch_resampled_z = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_eros_pitch['t'], dragon_eros_pitch['z'])
dragon_eros_pitch_resampled_qx = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_eros_pitch['t'], dragon_eros_pitch['qx'])
dragon_eros_pitch_resampled_qy = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_eros_pitch['t'], dragon_eros_pitch['qy'])
dragon_eros_pitch_resampled_qz = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_eros_pitch['t'], dragon_eros_pitch['qz'])
dragon_eros_pitch_resampled_qw = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_eros_pitch['t'], dragon_eros_pitch['qw'])

dragon_eros_yaw_resampled_x = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_eros_yaw['t'], dragon_eros_yaw['x'])
dragon_eros_yaw_resampled_y = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_eros_yaw['t'], dragon_eros_yaw['y'])
dragon_eros_yaw_resampled_z = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_eros_yaw['t'], dragon_eros_yaw['z'])
dragon_eros_yaw_resampled_qx = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_eros_yaw['t'], dragon_eros_yaw['qx'])
dragon_eros_yaw_resampled_qy = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_eros_yaw['t'], dragon_eros_yaw['qy'])
dragon_eros_yaw_resampled_qz = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_eros_yaw['t'], dragon_eros_yaw['qz'])
dragon_eros_yaw_resampled_qw = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_eros_yaw['t'], dragon_eros_yaw['qw'])

dragon_eros_trans_x_alpha,dragon_eros_trans_x_beta,dragon_eros_trans_x_gamma = quaternion_to_euler_angle(dragon_eros_trans_x['qw'], dragon_eros_trans_x['qx'], dragon_eros_trans_x['qy'], dragon_eros_trans_x['qz'])
dragon_eros_trans_x_alpha_cleaned = cleanEuler(dragon_eros_trans_x_alpha,0)
dragon_eros_trans_x_beta_cleaned = cleanEuler(dragon_eros_trans_x_beta,1)
dragon_eros_trans_x_gamma_cleaned = cleanEuler(dragon_eros_trans_x_gamma,2)

dragon_eros_trans_y_alpha,dragon_eros_trans_y_beta,dragon_eros_trans_y_gamma = quaternion_to_euler_angle(dragon_eros_trans_y['qw'], dragon_eros_trans_y['qx'], dragon_eros_trans_y['qy'], dragon_eros_trans_y['qz'])
dragon_eros_trans_y_alpha_cleaned = cleanEuler(dragon_eros_trans_y_alpha,0)
dragon_eros_trans_y_beta_cleaned = cleanEuler(dragon_eros_trans_y_beta,1)
dragon_eros_trans_y_gamma_cleaned = cleanEuler(dragon_eros_trans_y_gamma,2)

dragon_eros_trans_z_alpha,dragon_eros_trans_z_beta,dragon_eros_trans_z_gamma = quaternion_to_euler_angle(dragon_eros_trans_z['qw'], dragon_eros_trans_z['qx'], dragon_eros_trans_z['qy'], dragon_eros_trans_z['qz'])
dragon_eros_trans_z_alpha_cleaned = cleanEuler(dragon_eros_trans_z_alpha,0)
dragon_eros_trans_z_beta_cleaned = cleanEuler(dragon_eros_trans_z_beta,1)
dragon_eros_trans_z_gamma_cleaned = cleanEuler(dragon_eros_trans_z_gamma,2)

dragon_eros_roll_alpha,dragon_eros_roll_beta,dragon_eros_roll_gamma = quaternion_to_euler_angle(dragon_eros_roll['qw'], dragon_eros_roll['qx'], dragon_eros_roll['qy'], dragon_eros_roll['qz'])
dragon_eros_roll_alpha_cleaned = cleanEuler(dragon_eros_roll_alpha,0)
dragon_eros_roll_beta_cleaned = cleanEuler(dragon_eros_roll_beta,1)
dragon_eros_roll_gamma_cleaned = cleanEuler(dragon_eros_roll_gamma,2)

dragon_eros_pitch_alpha,dragon_eros_pitch_beta,dragon_eros_pitch_gamma = quaternion_to_euler_angle(dragon_eros_pitch['qw'], dragon_eros_pitch['qx'], dragon_eros_pitch['qy'], dragon_eros_pitch['qz'])
dragon_eros_pitch_alpha_cleaned = cleanEuler(dragon_eros_pitch_alpha,0)
dragon_eros_pitch_beta_cleaned = cleanEuler(dragon_eros_pitch_beta,1)
dragon_eros_pitch_gamma_cleaned = cleanEuler(dragon_eros_pitch_gamma,2)

dragon_eros_yaw_alpha,dragon_eros_yaw_beta,dragon_eros_yaw_gamma = quaternion_to_euler_angle(dragon_eros_yaw['qw'], dragon_eros_yaw['qx'], dragon_eros_yaw['qy'], dragon_eros_yaw['qz'])
dragon_eros_yaw_alpha_cleaned = cleanEuler(dragon_eros_yaw_alpha,0)
dragon_eros_yaw_beta_cleaned = cleanEuler(dragon_eros_yaw_beta,1)
dragon_eros_yaw_gamma_cleaned = cleanEuler(dragon_eros_yaw_gamma,2)

dragon_eros_error_trans_x = computeEuclideanDistance(dragon_gt_trans_x_new['x'], dragon_gt_trans_x_new['y'], dragon_gt_trans_x_new['z'], dragon_eros_trans_x_resampled_x, dragon_eros_trans_x_resampled_y, dragon_eros_trans_x_resampled_z)
dragon_eros_error_trans_y = computeEuclideanDistance(dragon_gt_trans_y_new['x'], dragon_gt_trans_y_new['y'], dragon_gt_trans_y_new['z'], dragon_eros_trans_y_resampled_x, dragon_eros_trans_y_resampled_y, dragon_eros_trans_y_resampled_z)
dragon_eros_error_trans_z = computeEuclideanDistance(dragon_gt_trans_z_new['x'], dragon_gt_trans_z_new['y'], dragon_gt_trans_z_new['z'], dragon_eros_trans_z_resampled_x, dragon_eros_trans_z_resampled_y, dragon_eros_trans_z_resampled_z)
dragon_eros_error_roll = computeEuclideanDistance(dragon_gt_roll_new['x'], dragon_gt_roll_new['y'], dragon_gt_roll_new['z'], dragon_eros_roll_resampled_x, dragon_eros_roll_resampled_y, dragon_eros_roll_resampled_z)
dragon_eros_error_pitch = computeEuclideanDistance(dragon_gt_pitch_new['x'], dragon_gt_pitch_new['y'], dragon_gt_pitch_new['z'], dragon_eros_pitch_resampled_x, dragon_eros_pitch_resampled_y, dragon_eros_pitch_resampled_z)
dragon_eros_error_yaw = computeEuclideanDistance(dragon_gt_yaw_new['x'], dragon_gt_yaw_new['y'], dragon_gt_yaw_new['z'], dragon_eros_yaw_resampled_x, dragon_eros_yaw_resampled_y, dragon_eros_yaw_resampled_z)

dragon_eros_q_angle_trans_x = computeQuaternionError(dragon_eros_trans_x_resampled_qx, dragon_eros_trans_x_resampled_qy, dragon_eros_trans_x_resampled_qz, dragon_eros_trans_x_resampled_qw, dragon_gt_trans_x_new['qx'], dragon_gt_trans_x_new['qy'], dragon_gt_trans_x_new['qz'], dragon_gt_trans_x_new['qw'])
dragon_eros_q_angle_trans_y = computeQuaternionError(dragon_eros_trans_y_resampled_qx, dragon_eros_trans_y_resampled_qy, dragon_eros_trans_y_resampled_qz, dragon_eros_trans_y_resampled_qw, dragon_gt_trans_y_new['qx'], dragon_gt_trans_y_new['qy'], dragon_gt_trans_y_new['qz'], dragon_gt_trans_y_new['qw'])
dragon_eros_q_angle_trans_z = computeQuaternionError(dragon_eros_trans_z_resampled_qx, dragon_eros_trans_z_resampled_qy, dragon_eros_trans_z_resampled_qz, dragon_eros_trans_z_resampled_qw, dragon_gt_trans_z_new['qx'], dragon_gt_trans_z_new['qy'], dragon_gt_trans_z_new['qz'], dragon_gt_trans_z_new['qw'])
dragon_eros_q_angle_roll = computeQuaternionError(dragon_eros_roll_resampled_qx, dragon_eros_roll_resampled_qy, dragon_eros_roll_resampled_qz, dragon_eros_roll_resampled_qw, dragon_gt_roll_new['qx'], dragon_gt_roll_new['qy'], dragon_gt_roll_new['qz'], dragon_gt_roll_new['qw'])
dragon_eros_q_angle_pitch = computeQuaternionError(dragon_eros_pitch_resampled_qx, dragon_eros_pitch_resampled_qy, dragon_eros_pitch_resampled_qz, dragon_eros_pitch_resampled_qw, dragon_gt_pitch_new['qx'], dragon_gt_pitch_new['qy'], dragon_gt_pitch_new['qz'], dragon_gt_pitch_new['qw'])
dragon_eros_q_angle_yaw = computeQuaternionError(dragon_eros_yaw_resampled_qx, dragon_eros_yaw_resampled_qy, dragon_eros_yaw_resampled_qz, dragon_eros_yaw_resampled_qw, dragon_gt_yaw_new['qx'], dragon_gt_yaw_new['qy'], dragon_gt_yaw_new['qz'], dragon_gt_yaw_new['qw'])

dragon_eros_position_errors = np.concatenate((dragon_eros_error_trans_x, dragon_eros_error_trans_y, dragon_eros_error_trans_z, dragon_eros_error_roll, dragon_eros_error_pitch, dragon_eros_error_yaw))
dragon_eros_rotation_errors = np.concatenate((dragon_eros_q_angle_trans_x, dragon_eros_q_angle_trans_y, dragon_eros_q_angle_trans_z, dragon_eros_q_angle_roll, dragon_eros_q_angle_pitch, dragon_eros_q_angle_yaw))

dragon_scarf_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_scarf_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_scarf_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_scarf_roll = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_scarf_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
dragon_scarf_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

dragon_scarf_trans_x_resampled_x = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_scarf_trans_x['t'], dragon_scarf_trans_x['x'])
dragon_scarf_trans_x_resampled_y = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_scarf_trans_x['t'], dragon_scarf_trans_x['y'])
dragon_scarf_trans_x_resampled_z = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_scarf_trans_x['t'], dragon_scarf_trans_x['z'])
dragon_scarf_trans_x_resampled_qx = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_scarf_trans_x['t'], dragon_scarf_trans_x['qx'])
dragon_scarf_trans_x_resampled_qy = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_scarf_trans_x['t'], dragon_scarf_trans_x['qy'])
dragon_scarf_trans_x_resampled_qz = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_scarf_trans_x['t'], dragon_scarf_trans_x['qz'])
dragon_scarf_trans_x_resampled_qw = resampling_by_interpolate(dragon_gt_trans_x_new['t'], dragon_scarf_trans_x['t'], dragon_scarf_trans_x['qw'])

dragon_scarf_trans_y_resampled_x = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_scarf_trans_y['t'], dragon_scarf_trans_y['x'])
dragon_scarf_trans_y_resampled_y = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_scarf_trans_y['t'], dragon_scarf_trans_y['y'])
dragon_scarf_trans_y_resampled_z = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_scarf_trans_y['t'], dragon_scarf_trans_y['z'])
dragon_scarf_trans_y_resampled_qx = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_scarf_trans_y['t'], dragon_scarf_trans_y['qx'])
dragon_scarf_trans_y_resampled_qy = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_scarf_trans_y['t'], dragon_scarf_trans_y['qy'])
dragon_scarf_trans_y_resampled_qz = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_scarf_trans_y['t'], dragon_scarf_trans_y['qz'])
dragon_scarf_trans_y_resampled_qw = resampling_by_interpolate(dragon_gt_trans_y_new['t'], dragon_scarf_trans_y['t'], dragon_scarf_trans_y['qw'])

dragon_scarf_trans_z_resampled_x = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_scarf_trans_z['t'], dragon_scarf_trans_z['x'])
dragon_scarf_trans_z_resampled_y = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_scarf_trans_z['t'], dragon_scarf_trans_z['y'])
dragon_scarf_trans_z_resampled_z = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_scarf_trans_z['t'], dragon_scarf_trans_z['z'])
dragon_scarf_trans_z_resampled_qx = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_scarf_trans_z['t'], dragon_scarf_trans_z['qx'])
dragon_scarf_trans_z_resampled_qy = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_scarf_trans_z['t'], dragon_scarf_trans_z['qy'])
dragon_scarf_trans_z_resampled_qz = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_scarf_trans_z['t'], dragon_scarf_trans_z['qz'])
dragon_scarf_trans_z_resampled_qw = resampling_by_interpolate(dragon_gt_trans_z_new['t'], dragon_scarf_trans_z['t'], dragon_scarf_trans_z['qw'])

dragon_scarf_roll_resampled_x = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_scarf_roll['t'], dragon_scarf_roll['x'])
dragon_scarf_roll_resampled_y = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_scarf_roll['t'], dragon_scarf_roll['y'])
dragon_scarf_roll_resampled_z = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_scarf_roll['t'], dragon_scarf_roll['z'])
dragon_scarf_roll_resampled_qx = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_scarf_roll['t'], dragon_scarf_roll['qx'])
dragon_scarf_roll_resampled_qy = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_scarf_roll['t'], dragon_scarf_roll['qy'])
dragon_scarf_roll_resampled_qz = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_scarf_roll['t'], dragon_scarf_roll['qz'])
dragon_scarf_roll_resampled_qw = resampling_by_interpolate(dragon_gt_roll_new['t'], dragon_scarf_roll['t'], dragon_scarf_roll['qw'])

dragon_scarf_pitch_resampled_x = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_scarf_pitch['t'], dragon_scarf_pitch['x'])
dragon_scarf_pitch_resampled_y = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_scarf_pitch['t'], dragon_scarf_pitch['y'])
dragon_scarf_pitch_resampled_z = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_scarf_pitch['t'], dragon_scarf_pitch['z'])
dragon_scarf_pitch_resampled_qx = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_scarf_pitch['t'], dragon_scarf_pitch['qx'])
dragon_scarf_pitch_resampled_qy = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_scarf_pitch['t'], dragon_scarf_pitch['qy'])
dragon_scarf_pitch_resampled_qz = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_scarf_pitch['t'], dragon_scarf_pitch['qz'])
dragon_scarf_pitch_resampled_qw = resampling_by_interpolate(dragon_gt_pitch_new['t'], dragon_scarf_pitch['t'], dragon_scarf_pitch['qw'])

dragon_scarf_yaw_resampled_x = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_scarf_yaw['t'], dragon_scarf_yaw['x'])
dragon_scarf_yaw_resampled_y = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_scarf_yaw['t'], dragon_scarf_yaw['y'])
dragon_scarf_yaw_resampled_z = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_scarf_yaw['t'], dragon_scarf_yaw['z'])
dragon_scarf_yaw_resampled_qx = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_scarf_yaw['t'], dragon_scarf_yaw['qx'])
dragon_scarf_yaw_resampled_qy = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_scarf_yaw['t'], dragon_scarf_yaw['qy'])
dragon_scarf_yaw_resampled_qz = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_scarf_yaw['t'], dragon_scarf_yaw['qz'])
dragon_scarf_yaw_resampled_qw = resampling_by_interpolate(dragon_gt_yaw_new['t'], dragon_scarf_yaw['t'], dragon_scarf_yaw['qw'])

dragon_scarf_trans_x_alpha,dragon_scarf_trans_x_beta,dragon_scarf_trans_x_gamma = quaternion_to_euler_angle(dragon_scarf_trans_x['qw'], dragon_scarf_trans_x['qx'], dragon_scarf_trans_x['qy'], dragon_scarf_trans_x['qz'])
dragon_scarf_trans_x_alpha_cleaned = cleanEuler(dragon_scarf_trans_x_alpha,0)
dragon_scarf_trans_x_beta_cleaned = cleanEuler(dragon_scarf_trans_x_beta,1)
dragon_scarf_trans_x_gamma_cleaned = cleanEuler(dragon_scarf_trans_x_gamma,2)

dragon_scarf_trans_y_alpha,dragon_scarf_trans_y_beta,dragon_scarf_trans_y_gamma = quaternion_to_euler_angle(dragon_scarf_trans_y['qw'], dragon_scarf_trans_y['qx'], dragon_scarf_trans_y['qy'], dragon_scarf_trans_y['qz'])
dragon_scarf_trans_y_alpha_cleaned = cleanEuler(dragon_scarf_trans_y_alpha,0)
dragon_scarf_trans_y_beta_cleaned = cleanEuler(dragon_scarf_trans_y_beta,1)
dragon_scarf_trans_y_gamma_cleaned = cleanEuler(dragon_scarf_trans_y_gamma,2)

dragon_scarf_trans_z_alpha,dragon_scarf_trans_z_beta,dragon_scarf_trans_z_gamma = quaternion_to_euler_angle(dragon_scarf_trans_z['qw'], dragon_scarf_trans_z['qx'], dragon_scarf_trans_z['qy'], dragon_scarf_trans_z['qz'])
dragon_scarf_trans_z_alpha_cleaned = cleanEuler(dragon_scarf_trans_z_alpha,0)
dragon_scarf_trans_z_beta_cleaned = cleanEuler(dragon_scarf_trans_z_beta,1)
dragon_scarf_trans_z_gamma_cleaned = cleanEuler(dragon_scarf_trans_z_gamma,2)

dragon_scarf_roll_alpha,dragon_scarf_roll_beta,dragon_scarf_roll_gamma = quaternion_to_euler_angle(dragon_scarf_roll['qw'], dragon_scarf_roll['qx'], dragon_scarf_roll['qy'], dragon_scarf_roll['qz'])
dragon_scarf_roll_alpha_cleaned = cleanEuler(dragon_scarf_roll_alpha,0)
dragon_scarf_roll_beta_cleaned = cleanEuler(dragon_scarf_roll_beta,1)
dragon_scarf_roll_gamma_cleaned = cleanEuler(dragon_scarf_roll_gamma,2)

dragon_scarf_pitch_alpha,dragon_scarf_pitch_beta,dragon_scarf_pitch_gamma = quaternion_to_euler_angle(dragon_scarf_pitch['qw'], dragon_scarf_pitch['qx'], dragon_scarf_pitch['qy'], dragon_scarf_pitch['qz'])
dragon_scarf_pitch_alpha_cleaned = cleanEuler(dragon_scarf_pitch_alpha,0)
dragon_scarf_pitch_beta_cleaned = cleanEuler(dragon_scarf_pitch_beta,1)
dragon_scarf_pitch_gamma_cleaned = cleanEuler(dragon_scarf_pitch_gamma,2)

dragon_scarf_yaw_alpha,dragon_scarf_yaw_beta,dragon_scarf_yaw_gamma = quaternion_to_euler_angle(dragon_scarf_yaw['qw'], dragon_scarf_yaw['qx'], dragon_scarf_yaw['qy'], dragon_scarf_yaw['qz'])
dragon_scarf_yaw_alpha_cleaned = cleanEuler(dragon_scarf_yaw_alpha,0)
dragon_scarf_yaw_beta_cleaned = cleanEuler(dragon_scarf_yaw_beta,1)
dragon_scarf_yaw_gamma_cleaned = cleanEuler(dragon_scarf_yaw_gamma,2)

dragon_scarf_error_trans_x = computeEuclideanDistance(dragon_gt_trans_x_new['x'], dragon_gt_trans_x_new['y'], dragon_gt_trans_x_new['z'], dragon_scarf_trans_x_resampled_x, dragon_scarf_trans_x_resampled_y, dragon_scarf_trans_x_resampled_z)
dragon_scarf_error_trans_y = computeEuclideanDistance(dragon_gt_trans_y_new['x'], dragon_gt_trans_y_new['y'], dragon_gt_trans_y_new['z'], dragon_scarf_trans_y_resampled_x, dragon_scarf_trans_y_resampled_y, dragon_scarf_trans_y_resampled_z)
dragon_scarf_error_trans_z = computeEuclideanDistance(dragon_gt_trans_z_new['x'], dragon_gt_trans_z_new['y'], dragon_gt_trans_z_new['z'], dragon_scarf_trans_z_resampled_x, dragon_scarf_trans_z_resampled_y, dragon_scarf_trans_z_resampled_z)
dragon_scarf_error_roll = computeEuclideanDistance(dragon_gt_roll_new['x'], dragon_gt_roll_new['y'], dragon_gt_roll_new['z'], dragon_scarf_roll_resampled_x, dragon_scarf_roll_resampled_y, dragon_scarf_roll_resampled_z)
dragon_scarf_error_pitch = computeEuclideanDistance(dragon_gt_pitch_new['x'], dragon_gt_pitch_new['y'], dragon_gt_pitch_new['z'], dragon_scarf_pitch_resampled_x, dragon_scarf_pitch_resampled_y, dragon_scarf_pitch_resampled_z)
dragon_scarf_error_yaw = computeEuclideanDistance(dragon_gt_yaw_new['x'], dragon_gt_yaw_new['y'], dragon_gt_yaw_new['z'], dragon_scarf_yaw_resampled_x, dragon_scarf_yaw_resampled_y, dragon_scarf_yaw_resampled_z)

dragon_scarf_q_angle_trans_x = computeQuaternionError(dragon_scarf_trans_x_resampled_qx, dragon_scarf_trans_x_resampled_qy, dragon_scarf_trans_x_resampled_qz, dragon_scarf_trans_x_resampled_qw, dragon_gt_trans_x_new['qx'], dragon_gt_trans_x_new['qy'], dragon_gt_trans_x_new['qz'], dragon_gt_trans_x_new['qw'])
dragon_scarf_q_angle_trans_y = computeQuaternionError(dragon_scarf_trans_y_resampled_qx, dragon_scarf_trans_y_resampled_qy, dragon_scarf_trans_y_resampled_qz, dragon_scarf_trans_y_resampled_qw, dragon_gt_trans_y_new['qx'], dragon_gt_trans_y_new['qy'], dragon_gt_trans_y_new['qz'], dragon_gt_trans_y_new['qw'])
dragon_scarf_q_angle_trans_z = computeQuaternionError(dragon_scarf_trans_z_resampled_qx, dragon_scarf_trans_z_resampled_qy, dragon_scarf_trans_z_resampled_qz, dragon_scarf_trans_z_resampled_qw, dragon_gt_trans_z_new['qx'], dragon_gt_trans_z_new['qy'], dragon_gt_trans_z_new['qz'], dragon_gt_trans_z_new['qw'])
dragon_scarf_q_angle_roll = computeQuaternionError(dragon_scarf_roll_resampled_qx, dragon_scarf_roll_resampled_qy, dragon_scarf_roll_resampled_qz, dragon_scarf_roll_resampled_qw, dragon_gt_roll_new['qx'], dragon_gt_roll_new['qy'], dragon_gt_roll_new['qz'], dragon_gt_roll_new['qw'])
dragon_scarf_q_angle_pitch = computeQuaternionError(dragon_scarf_pitch_resampled_qx, dragon_scarf_pitch_resampled_qy, dragon_scarf_pitch_resampled_qz, dragon_scarf_pitch_resampled_qw, dragon_gt_pitch_new['qx'], dragon_gt_pitch_new['qy'], dragon_gt_pitch_new['qz'], dragon_gt_pitch_new['qw'])
dragon_scarf_q_angle_yaw = computeQuaternionError(dragon_scarf_yaw_resampled_qx, dragon_scarf_yaw_resampled_qy, dragon_scarf_yaw_resampled_qz, dragon_scarf_yaw_resampled_qw, dragon_gt_yaw_new['qx'], dragon_gt_yaw_new['qy'], dragon_gt_yaw_new['qz'], dragon_gt_yaw_new['qw'])

dragon_scarf_position_errors = np.concatenate((dragon_scarf_error_trans_x, dragon_scarf_error_trans_y, dragon_scarf_error_trans_z, dragon_scarf_error_roll, dragon_scarf_error_pitch, dragon_scarf_error_yaw))
dragon_scarf_rotation_errors = np.concatenate((dragon_scarf_q_angle_trans_x, dragon_scarf_q_angle_trans_y, dragon_scarf_q_angle_trans_z, dragon_scarf_q_angle_roll, dragon_scarf_q_angle_pitch, dragon_scarf_q_angle_yaw))


# ---------------------------------------------------------------------------  GELATIN  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/scarf-journal/dragon/'
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

gelatin_gt_trans_x_alpha,gelatin_gt_trans_x_beta,gelatin_gt_trans_x_gamma = quaternion_to_euler_angle(gelatin_gt_trans_x['qw'], gelatin_gt_trans_x['qx'], gelatin_gt_trans_x['qy'], gelatin_gt_trans_x['qz'])
gelatin_gt_trans_x_alpha_cleaned = cleanEuler(gelatin_gt_trans_x_alpha,0)
gelatin_gt_trans_x_beta_cleaned = cleanEuler(gelatin_gt_trans_x_beta,1)
gelatin_gt_trans_x_gamma_cleaned = cleanEuler(gelatin_gt_trans_x_gamma,1)

gelatin_gt_trans_y_alpha,gelatin_gt_trans_y_beta,gelatin_gt_trans_y_gamma = quaternion_to_euler_angle(gelatin_gt_trans_y['qw'], gelatin_gt_trans_y['qx'], gelatin_gt_trans_y['qy'], gelatin_gt_trans_y['qz'])
gelatin_gt_trans_y_alpha_cleaned = cleanEuler(gelatin_gt_trans_y_alpha,0)
gelatin_gt_trans_y_beta_cleaned = cleanEuler(gelatin_gt_trans_y_beta,1)
gelatin_gt_trans_y_gamma_cleaned = cleanEuler(gelatin_gt_trans_y_gamma,2)

gelatin_gt_trans_z_alpha,gelatin_gt_trans_z_beta,gelatin_gt_trans_z_gamma = quaternion_to_euler_angle(gelatin_gt_trans_z['qw'], gelatin_gt_trans_z['qx'], gelatin_gt_trans_z['qy'], gelatin_gt_trans_z['qz'])
gelatin_gt_trans_z_alpha_cleaned = cleanEuler(gelatin_gt_trans_z_alpha,0)
gelatin_gt_trans_z_beta_cleaned = cleanEuler(gelatin_gt_trans_z_beta,1)
gelatin_gt_trans_z_gamma_cleaned = cleanEuler(gelatin_gt_trans_z_gamma,2)

gelatin_gt_roll_alpha,gelatin_gt_roll_beta,gelatin_gt_roll_gamma = quaternion_to_euler_angle(gelatin_gt_roll['qw'], gelatin_gt_roll['qx'], gelatin_gt_roll['qy'], gelatin_gt_roll['qz'])
gelatin_gt_roll_alpha_cleaned = cleanEuler(gelatin_gt_roll_alpha,0)
gelatin_gt_roll_beta_cleaned = cleanEuler(gelatin_gt_roll_beta,1)
gelatin_gt_roll_gamma_cleaned = cleanEuler(gelatin_gt_roll_gamma,2)

gelatin_gt_pitch_alpha,gelatin_gt_pitch_beta,gelatin_gt_pitch_gamma = quaternion_to_euler_angle(gelatin_gt_pitch['qw'], gelatin_gt_pitch['qx'], gelatin_gt_pitch['qy'], gelatin_gt_pitch['qz'])
gelatin_gt_pitch_alpha_cleaned = cleanEuler(gelatin_gt_pitch_alpha,0)
gelatin_gt_pitch_beta_cleaned = cleanEuler(gelatin_gt_pitch_beta,1)
gelatin_gt_pitch_gamma_cleaned = cleanEuler(gelatin_gt_pitch_gamma,2)

gelatin_gt_yaw_alpha,gelatin_gt_yaw_beta,gelatin_gt_yaw_gamma = quaternion_to_euler_angle(gelatin_gt_yaw['qw'], gelatin_gt_yaw['qx'], gelatin_gt_yaw['qy'], gelatin_gt_yaw['qz'])
gelatin_gt_yaw_alpha_cleaned = cleanEuler(gelatin_gt_yaw_alpha,0)
gelatin_gt_yaw_beta_cleaned = cleanEuler(gelatin_gt_yaw_beta,1)
gelatin_gt_yaw_gamma_cleaned = cleanEuler(gelatin_gt_yaw_gamma,2)

gelatin_eros_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_eros_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_eros_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_eros_roll = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_eros_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_eros_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gelatin_eros_trans_x_resampled_x = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_eros_trans_x['t'], gelatin_eros_trans_x['x'])
gelatin_eros_trans_x_resampled_y = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_eros_trans_x['t'], gelatin_eros_trans_x['y'])
gelatin_eros_trans_x_resampled_z = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_eros_trans_x['t'], gelatin_eros_trans_x['z'])
gelatin_eros_trans_x_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_eros_trans_x['t'], gelatin_eros_trans_x['qx'])
gelatin_eros_trans_x_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_eros_trans_x['t'], gelatin_eros_trans_x['qy'])
gelatin_eros_trans_x_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_eros_trans_x['t'], gelatin_eros_trans_x['qz'])
gelatin_eros_trans_x_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_eros_trans_x['t'], gelatin_eros_trans_x['qw'])

gelatin_eros_trans_y_resampled_x = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_eros_trans_y['t'], gelatin_eros_trans_y['x'])
gelatin_eros_trans_y_resampled_y = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_eros_trans_y['t'], gelatin_eros_trans_y['y'])
gelatin_eros_trans_y_resampled_z = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_eros_trans_y['t'], gelatin_eros_trans_y['z'])
gelatin_eros_trans_y_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_eros_trans_y['t'], gelatin_eros_trans_y['qx'])
gelatin_eros_trans_y_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_eros_trans_y['t'], gelatin_eros_trans_y['qy'])
gelatin_eros_trans_y_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_eros_trans_y['t'], gelatin_eros_trans_y['qz'])
gelatin_eros_trans_y_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_eros_trans_y['t'], gelatin_eros_trans_y['qw'])

gelatin_eros_trans_z_resampled_x = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_eros_trans_z['t'], gelatin_eros_trans_z['x'])
gelatin_eros_trans_z_resampled_y = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_eros_trans_z['t'], gelatin_eros_trans_z['y'])
gelatin_eros_trans_z_resampled_z = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_eros_trans_z['t'], gelatin_eros_trans_z['z'])
gelatin_eros_trans_z_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_eros_trans_z['t'], gelatin_eros_trans_z['qx'])
gelatin_eros_trans_z_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_eros_trans_z['t'], gelatin_eros_trans_z['qy'])
gelatin_eros_trans_z_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_eros_trans_z['t'], gelatin_eros_trans_z['qz'])
gelatin_eros_trans_z_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_eros_trans_z['t'], gelatin_eros_trans_z['qw'])

gelatin_eros_roll_resampled_x = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_eros_roll['t'], gelatin_eros_roll['x'])
gelatin_eros_roll_resampled_y = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_eros_roll['t'], gelatin_eros_roll['y'])
gelatin_eros_roll_resampled_z = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_eros_roll['t'], gelatin_eros_roll['z'])
gelatin_eros_roll_resampled_qx = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_eros_roll['t'], gelatin_eros_roll['qx'])
gelatin_eros_roll_resampled_qy = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_eros_roll['t'], gelatin_eros_roll['qy'])
gelatin_eros_roll_resampled_qz = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_eros_roll['t'], gelatin_eros_roll['qz'])
gelatin_eros_roll_resampled_qw = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_eros_roll['t'], gelatin_eros_roll['qw'])

gelatin_eros_pitch_resampled_x = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_eros_pitch['t'], gelatin_eros_pitch['x'])
gelatin_eros_pitch_resampled_y = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_eros_pitch['t'], gelatin_eros_pitch['y'])
gelatin_eros_pitch_resampled_z = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_eros_pitch['t'], gelatin_eros_pitch['z'])
gelatin_eros_pitch_resampled_qx = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_eros_pitch['t'], gelatin_eros_pitch['qx'])
gelatin_eros_pitch_resampled_qy = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_eros_pitch['t'], gelatin_eros_pitch['qy'])
gelatin_eros_pitch_resampled_qz = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_eros_pitch['t'], gelatin_eros_pitch['qz'])
gelatin_eros_pitch_resampled_qw = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_eros_pitch['t'], gelatin_eros_pitch['qw'])

gelatin_eros_yaw_resampled_x = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_eros_yaw['t'], gelatin_eros_yaw['x'])
gelatin_eros_yaw_resampled_y = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_eros_yaw['t'], gelatin_eros_yaw['y'])
gelatin_eros_yaw_resampled_z = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_eros_yaw['t'], gelatin_eros_yaw['z'])
gelatin_eros_yaw_resampled_qx = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_eros_yaw['t'], gelatin_eros_yaw['qx'])
gelatin_eros_yaw_resampled_qy = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_eros_yaw['t'], gelatin_eros_yaw['qy'])
gelatin_eros_yaw_resampled_qz = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_eros_yaw['t'], gelatin_eros_yaw['qz'])
gelatin_eros_yaw_resampled_qw = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_eros_yaw['t'], gelatin_eros_yaw['qw'])

gelatin_eros_trans_x_alpha,gelatin_eros_trans_x_beta,gelatin_eros_trans_x_gamma = quaternion_to_euler_angle(gelatin_eros_trans_x['qw'], gelatin_eros_trans_x['qx'], gelatin_eros_trans_x['qy'], gelatin_eros_trans_x['qz'])
gelatin_eros_trans_x_alpha_cleaned = cleanEuler(gelatin_eros_trans_x_alpha,0)
gelatin_eros_trans_x_beta_cleaned = cleanEuler(gelatin_eros_trans_x_beta,1)
gelatin_eros_trans_x_gamma_cleaned = cleanEuler(gelatin_eros_trans_x_gamma,2)

gelatin_eros_trans_y_alpha,gelatin_eros_trans_y_beta,gelatin_eros_trans_y_gamma = quaternion_to_euler_angle(gelatin_eros_trans_y['qw'], gelatin_eros_trans_y['qx'], gelatin_eros_trans_y['qy'], gelatin_eros_trans_y['qz'])
gelatin_eros_trans_y_alpha_cleaned = cleanEuler(gelatin_eros_trans_y_alpha,0)
gelatin_eros_trans_y_beta_cleaned = cleanEuler(gelatin_eros_trans_y_beta,1)
gelatin_eros_trans_y_gamma_cleaned = cleanEuler(gelatin_eros_trans_y_gamma,2)

gelatin_eros_trans_z_alpha,gelatin_eros_trans_z_beta,gelatin_eros_trans_z_gamma = quaternion_to_euler_angle(gelatin_eros_trans_z['qw'], gelatin_eros_trans_z['qx'], gelatin_eros_trans_z['qy'], gelatin_eros_trans_z['qz'])
gelatin_eros_trans_z_alpha_cleaned = cleanEuler(gelatin_eros_trans_z_alpha,0)
gelatin_eros_trans_z_beta_cleaned = cleanEuler(gelatin_eros_trans_z_beta,1)
gelatin_eros_trans_z_gamma_cleaned = cleanEuler(gelatin_eros_trans_z_gamma,2)

gelatin_eros_roll_alpha,gelatin_eros_roll_beta,gelatin_eros_roll_gamma = quaternion_to_euler_angle(gelatin_eros_roll['qw'], gelatin_eros_roll['qx'], gelatin_eros_roll['qy'], gelatin_eros_roll['qz'])
gelatin_eros_roll_alpha_cleaned = cleanEuler(gelatin_eros_roll_alpha,0)
gelatin_eros_roll_beta_cleaned = cleanEuler(gelatin_eros_roll_beta,1)
gelatin_eros_roll_gamma_cleaned = cleanEuler(gelatin_eros_roll_gamma,2)

gelatin_eros_pitch_alpha,gelatin_eros_pitch_beta,gelatin_eros_pitch_gamma = quaternion_to_euler_angle(gelatin_eros_pitch['qw'], gelatin_eros_pitch['qx'], gelatin_eros_pitch['qy'], gelatin_eros_pitch['qz'])
gelatin_eros_pitch_alpha_cleaned = cleanEuler(gelatin_eros_pitch_alpha,0)
gelatin_eros_pitch_beta_cleaned = cleanEuler(gelatin_eros_pitch_beta,1)
gelatin_eros_pitch_gamma_cleaned = cleanEuler(gelatin_eros_pitch_gamma,2)

gelatin_eros_yaw_alpha,gelatin_eros_yaw_beta,gelatin_eros_yaw_gamma = quaternion_to_euler_angle(gelatin_eros_yaw['qw'], gelatin_eros_yaw['qx'], gelatin_eros_yaw['qy'], gelatin_eros_yaw['qz'])
gelatin_eros_yaw_alpha_cleaned = cleanEuler(gelatin_eros_yaw_alpha,0)
gelatin_eros_yaw_beta_cleaned = cleanEuler(gelatin_eros_yaw_beta,1)
gelatin_eros_yaw_gamma_cleaned = cleanEuler(gelatin_eros_yaw_gamma,2)

gelatin_eros_error_trans_x = computeEuclideanDistance(gelatin_gt_trans_x['x'], gelatin_gt_trans_x['y'], gelatin_gt_trans_x['z'], gelatin_eros_trans_x_resampled_x, gelatin_eros_trans_x_resampled_y, gelatin_eros_trans_x_resampled_z)
gelatin_eros_error_trans_y = computeEuclideanDistance(gelatin_gt_trans_y['x'], gelatin_gt_trans_y['y'], gelatin_gt_trans_y['z'], gelatin_eros_trans_y_resampled_x, gelatin_eros_trans_y_resampled_y, gelatin_eros_trans_y_resampled_z)
gelatin_eros_error_trans_z = computeEuclideanDistance(gelatin_gt_trans_z['x'], gelatin_gt_trans_z['y'], gelatin_gt_trans_z['z'], gelatin_eros_trans_z_resampled_x, gelatin_eros_trans_z_resampled_y, gelatin_eros_trans_z_resampled_z)
gelatin_eros_error_roll = computeEuclideanDistance(gelatin_gt_roll['x'], gelatin_gt_roll['y'], gelatin_gt_roll['z'], gelatin_eros_roll_resampled_x, gelatin_eros_roll_resampled_y, gelatin_eros_roll_resampled_z)
gelatin_eros_error_pitch = computeEuclideanDistance(gelatin_gt_pitch['x'], gelatin_gt_pitch['y'], gelatin_gt_pitch['z'], gelatin_eros_pitch_resampled_x, gelatin_eros_pitch_resampled_y, gelatin_eros_pitch_resampled_z)
gelatin_eros_error_yaw = computeEuclideanDistance(gelatin_gt_yaw['x'], gelatin_gt_yaw['y'], gelatin_gt_yaw['z'], gelatin_eros_yaw_resampled_x, gelatin_eros_yaw_resampled_y, gelatin_eros_yaw_resampled_z)

gelatin_eros_q_angle_trans_x = computeQuaternionError(gelatin_eros_trans_x_resampled_qx, gelatin_eros_trans_x_resampled_qy, gelatin_eros_trans_x_resampled_qz, gelatin_eros_trans_x_resampled_qw, gelatin_gt_trans_x['qx'], gelatin_gt_trans_x['qy'], gelatin_gt_trans_x['qz'], gelatin_gt_trans_x['qw'])
gelatin_eros_q_angle_trans_y = computeQuaternionError(gelatin_eros_trans_y_resampled_qx, gelatin_eros_trans_y_resampled_qy, gelatin_eros_trans_y_resampled_qz, gelatin_eros_trans_y_resampled_qw, gelatin_gt_trans_y['qx'], gelatin_gt_trans_y['qy'], gelatin_gt_trans_y['qz'], gelatin_gt_trans_y['qw'])
gelatin_eros_q_angle_trans_z = computeQuaternionError(gelatin_eros_trans_z_resampled_qx, gelatin_eros_trans_z_resampled_qy, gelatin_eros_trans_z_resampled_qz, gelatin_eros_trans_z_resampled_qw, gelatin_gt_trans_z['qx'], gelatin_gt_trans_z['qy'], gelatin_gt_trans_z['qz'], gelatin_gt_trans_z['qw'])
gelatin_eros_q_angle_roll = computeQuaternionError(gelatin_eros_roll_resampled_qx, gelatin_eros_roll_resampled_qy, gelatin_eros_roll_resampled_qz, gelatin_eros_roll_resampled_qw, gelatin_gt_roll['qx'], gelatin_gt_roll['qy'], gelatin_gt_roll['qz'], gelatin_gt_roll['qw'])
gelatin_eros_q_angle_pitch = computeQuaternionError(gelatin_eros_pitch_resampled_qx, gelatin_eros_pitch_resampled_qy, gelatin_eros_pitch_resampled_qz, gelatin_eros_pitch_resampled_qw, gelatin_gt_pitch['qx'], gelatin_gt_pitch['qy'], gelatin_gt_pitch['qz'], gelatin_gt_pitch['qw'])
gelatin_eros_q_angle_yaw = computeQuaternionError(gelatin_eros_yaw_resampled_qx, gelatin_eros_yaw_resampled_qy, gelatin_eros_yaw_resampled_qz, gelatin_eros_yaw_resampled_qw, gelatin_gt_yaw['qx'], gelatin_gt_yaw['qy'], gelatin_gt_yaw['qz'], gelatin_gt_yaw['qw'])

gelatin_eros_position_errors = np.concatenate((gelatin_eros_error_trans_x, gelatin_eros_error_trans_y, gelatin_eros_error_trans_z, gelatin_eros_error_roll, gelatin_eros_error_pitch, gelatin_eros_error_yaw))
gelatin_eros_rotation_errors = np.concatenate((gelatin_eros_q_angle_trans_x, gelatin_eros_q_angle_trans_y, gelatin_eros_q_angle_trans_z, gelatin_eros_q_angle_roll, gelatin_eros_q_angle_pitch, gelatin_eros_q_angle_yaw))

gelatin_scarf_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_scarf_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_scarf_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_scarf_roll = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_scarf_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
gelatin_scarf_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

gelatin_scarf_trans_x_resampled_x = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_scarf_trans_x['t'], gelatin_scarf_trans_x['x'])
gelatin_scarf_trans_x_resampled_y = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_scarf_trans_x['t'], gelatin_scarf_trans_x['y'])
gelatin_scarf_trans_x_resampled_z = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_scarf_trans_x['t'], gelatin_scarf_trans_x['z'])
gelatin_scarf_trans_x_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_scarf_trans_x['t'], gelatin_scarf_trans_x['qx'])
gelatin_scarf_trans_x_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_scarf_trans_x['t'], gelatin_scarf_trans_x['qy'])
gelatin_scarf_trans_x_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_scarf_trans_x['t'], gelatin_scarf_trans_x['qz'])
gelatin_scarf_trans_x_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_x['t'], gelatin_scarf_trans_x['t'], gelatin_scarf_trans_x['qw'])

gelatin_scarf_trans_y_resampled_x = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_scarf_trans_y['t'], gelatin_scarf_trans_y['x'])
gelatin_scarf_trans_y_resampled_y = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_scarf_trans_y['t'], gelatin_scarf_trans_y['y'])
gelatin_scarf_trans_y_resampled_z = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_scarf_trans_y['t'], gelatin_scarf_trans_y['z'])
gelatin_scarf_trans_y_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_scarf_trans_y['t'], gelatin_scarf_trans_y['qx'])
gelatin_scarf_trans_y_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_scarf_trans_y['t'], gelatin_scarf_trans_y['qy'])
gelatin_scarf_trans_y_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_scarf_trans_y['t'], gelatin_scarf_trans_y['qz'])
gelatin_scarf_trans_y_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_y['t'], gelatin_scarf_trans_y['t'], gelatin_scarf_trans_y['qw'])

gelatin_scarf_trans_z_resampled_x = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_scarf_trans_z['t'], gelatin_scarf_trans_z['x'])
gelatin_scarf_trans_z_resampled_y = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_scarf_trans_z['t'], gelatin_scarf_trans_z['y'])
gelatin_scarf_trans_z_resampled_z = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_scarf_trans_z['t'], gelatin_scarf_trans_z['z'])
gelatin_scarf_trans_z_resampled_qx = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_scarf_trans_z['t'], gelatin_scarf_trans_z['qx'])
gelatin_scarf_trans_z_resampled_qy = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_scarf_trans_z['t'], gelatin_scarf_trans_z['qy'])
gelatin_scarf_trans_z_resampled_qz = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_scarf_trans_z['t'], gelatin_scarf_trans_z['qz'])
gelatin_scarf_trans_z_resampled_qw = resampling_by_interpolate(gelatin_gt_trans_z['t'], gelatin_scarf_trans_z['t'], gelatin_scarf_trans_z['qw'])

gelatin_scarf_roll_resampled_x = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_scarf_roll['t'], gelatin_scarf_roll['x'])
gelatin_scarf_roll_resampled_y = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_scarf_roll['t'], gelatin_scarf_roll['y'])
gelatin_scarf_roll_resampled_z = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_scarf_roll['t'], gelatin_scarf_roll['z'])
gelatin_scarf_roll_resampled_qx = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_scarf_roll['t'], gelatin_scarf_roll['qx'])
gelatin_scarf_roll_resampled_qy = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_scarf_roll['t'], gelatin_scarf_roll['qy'])
gelatin_scarf_roll_resampled_qz = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_scarf_roll['t'], gelatin_scarf_roll['qz'])
gelatin_scarf_roll_resampled_qw = resampling_by_interpolate(gelatin_gt_roll['t'], gelatin_scarf_roll['t'], gelatin_scarf_roll['qw'])

gelatin_scarf_pitch_resampled_x = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_scarf_pitch['t'], gelatin_scarf_pitch['x'])
gelatin_scarf_pitch_resampled_y = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_scarf_pitch['t'], gelatin_scarf_pitch['y'])
gelatin_scarf_pitch_resampled_z = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_scarf_pitch['t'], gelatin_scarf_pitch['z'])
gelatin_scarf_pitch_resampled_qx = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_scarf_pitch['t'], gelatin_scarf_pitch['qx'])
gelatin_scarf_pitch_resampled_qy = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_scarf_pitch['t'], gelatin_scarf_pitch['qy'])
gelatin_scarf_pitch_resampled_qz = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_scarf_pitch['t'], gelatin_scarf_pitch['qz'])
gelatin_scarf_pitch_resampled_qw = resampling_by_interpolate(gelatin_gt_pitch['t'], gelatin_scarf_pitch['t'], gelatin_scarf_pitch['qw'])

gelatin_scarf_yaw_resampled_x = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_scarf_yaw['t'], gelatin_scarf_yaw['x'])
gelatin_scarf_yaw_resampled_y = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_scarf_yaw['t'], gelatin_scarf_yaw['y'])
gelatin_scarf_yaw_resampled_z = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_scarf_yaw['t'], gelatin_scarf_yaw['z'])
gelatin_scarf_yaw_resampled_qx = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_scarf_yaw['t'], gelatin_scarf_yaw['qx'])
gelatin_scarf_yaw_resampled_qy = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_scarf_yaw['t'], gelatin_scarf_yaw['qy'])
gelatin_scarf_yaw_resampled_qz = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_scarf_yaw['t'], gelatin_scarf_yaw['qz'])
gelatin_scarf_yaw_resampled_qw = resampling_by_interpolate(gelatin_gt_yaw['t'], gelatin_scarf_yaw['t'], gelatin_scarf_yaw['qw'])

gelatin_scarf_trans_x_alpha,gelatin_scarf_trans_x_beta,gelatin_scarf_trans_x_gamma = quaternion_to_euler_angle(gelatin_scarf_trans_x['qw'], gelatin_scarf_trans_x['qx'], gelatin_scarf_trans_x['qy'], gelatin_scarf_trans_x['qz'])
gelatin_scarf_trans_x_alpha_cleaned = cleanEuler(gelatin_scarf_trans_x_alpha,0)
gelatin_scarf_trans_x_beta_cleaned = cleanEuler(gelatin_scarf_trans_x_beta,1)
gelatin_scarf_trans_x_gamma_cleaned = cleanEuler(gelatin_scarf_trans_x_gamma,2)

gelatin_scarf_trans_y_alpha,gelatin_scarf_trans_y_beta,gelatin_scarf_trans_y_gamma = quaternion_to_euler_angle(gelatin_scarf_trans_y['qw'], gelatin_scarf_trans_y['qx'], gelatin_scarf_trans_y['qy'], gelatin_scarf_trans_y['qz'])
gelatin_scarf_trans_y_alpha_cleaned = cleanEuler(gelatin_scarf_trans_y_alpha,0)
gelatin_scarf_trans_y_beta_cleaned = cleanEuler(gelatin_scarf_trans_y_beta,1)
gelatin_scarf_trans_y_gamma_cleaned = cleanEuler(gelatin_scarf_trans_y_gamma,2)

gelatin_scarf_trans_z_alpha,gelatin_scarf_trans_z_beta,gelatin_scarf_trans_z_gamma = quaternion_to_euler_angle(gelatin_scarf_trans_z['qw'], gelatin_scarf_trans_z['qx'], gelatin_scarf_trans_z['qy'], gelatin_scarf_trans_z['qz'])
gelatin_scarf_trans_z_alpha_cleaned = cleanEuler(gelatin_scarf_trans_z_alpha,0)
gelatin_scarf_trans_z_beta_cleaned = cleanEuler(gelatin_scarf_trans_z_beta,1)
gelatin_scarf_trans_z_gamma_cleaned = cleanEuler(gelatin_scarf_trans_z_gamma,2)

gelatin_scarf_roll_alpha,gelatin_scarf_roll_beta,gelatin_scarf_roll_gamma = quaternion_to_euler_angle(gelatin_scarf_roll['qw'], gelatin_scarf_roll['qx'], gelatin_scarf_roll['qy'], gelatin_scarf_roll['qz'])
gelatin_scarf_roll_alpha_cleaned = cleanEuler(gelatin_scarf_roll_alpha,0)
gelatin_scarf_roll_beta_cleaned = cleanEuler(gelatin_scarf_roll_beta,1)
gelatin_scarf_roll_gamma_cleaned = cleanEuler(gelatin_scarf_roll_gamma,2)

gelatin_scarf_pitch_alpha,gelatin_scarf_pitch_beta,gelatin_scarf_pitch_gamma = quaternion_to_euler_angle(gelatin_scarf_pitch['qw'], gelatin_scarf_pitch['qx'], gelatin_scarf_pitch['qy'], gelatin_scarf_pitch['qz'])
gelatin_scarf_pitch_alpha_cleaned = cleanEuler(gelatin_scarf_pitch_alpha,0)
gelatin_scarf_pitch_beta_cleaned = cleanEuler(gelatin_scarf_pitch_beta,1)
gelatin_scarf_pitch_gamma_cleaned = cleanEuler(gelatin_scarf_pitch_gamma,2)

gelatin_scarf_yaw_alpha,gelatin_scarf_yaw_beta,gelatin_scarf_yaw_gamma = quaternion_to_euler_angle(gelatin_scarf_yaw['qw'], gelatin_scarf_yaw['qx'], gelatin_scarf_yaw['qy'], gelatin_scarf_yaw['qz'])
gelatin_scarf_yaw_alpha_cleaned = cleanEuler(gelatin_scarf_yaw_alpha,0)
gelatin_scarf_yaw_beta_cleaned = cleanEuler(gelatin_scarf_yaw_beta,1)
gelatin_scarf_yaw_gamma_cleaned = cleanEuler(gelatin_scarf_yaw_gamma,2)

gelatin_scarf_error_trans_x = computeEuclideanDistance(gelatin_gt_trans_x['x'], gelatin_gt_trans_x['y'], gelatin_gt_trans_x['z'], gelatin_scarf_trans_x_resampled_x, gelatin_scarf_trans_x_resampled_y, gelatin_scarf_trans_x_resampled_z)
gelatin_scarf_error_trans_y = computeEuclideanDistance(gelatin_gt_trans_y['x'], gelatin_gt_trans_y['y'], gelatin_gt_trans_y['z'], gelatin_scarf_trans_y_resampled_x, gelatin_scarf_trans_y_resampled_y, gelatin_scarf_trans_y_resampled_z)
gelatin_scarf_error_trans_z = computeEuclideanDistance(gelatin_gt_trans_z['x'], gelatin_gt_trans_z['y'], gelatin_gt_trans_z['z'], gelatin_scarf_trans_z_resampled_x, gelatin_scarf_trans_z_resampled_y, gelatin_scarf_trans_z_resampled_z)
gelatin_scarf_error_roll = computeEuclideanDistance(gelatin_gt_roll['x'], gelatin_gt_roll['y'], gelatin_gt_roll['z'], gelatin_scarf_roll_resampled_x, gelatin_scarf_roll_resampled_y, gelatin_scarf_roll_resampled_z)
gelatin_scarf_error_pitch = computeEuclideanDistance(gelatin_gt_pitch['x'], gelatin_gt_pitch['y'], gelatin_gt_pitch['z'], gelatin_scarf_pitch_resampled_x, gelatin_scarf_pitch_resampled_y, gelatin_scarf_pitch_resampled_z)
gelatin_scarf_error_yaw = computeEuclideanDistance(gelatin_gt_yaw['x'], gelatin_gt_yaw['y'], gelatin_gt_yaw['z'], gelatin_scarf_yaw_resampled_x, gelatin_scarf_yaw_resampled_y, gelatin_scarf_yaw_resampled_z)

gelatin_scarf_q_angle_trans_x = computeQuaternionError(gelatin_scarf_trans_x_resampled_qx, gelatin_scarf_trans_x_resampled_qy, gelatin_scarf_trans_x_resampled_qz, gelatin_scarf_trans_x_resampled_qw, gelatin_gt_trans_x['qx'], gelatin_gt_trans_x['qy'], gelatin_gt_trans_x['qz'], gelatin_gt_trans_x['qw'])
gelatin_scarf_q_angle_trans_y = computeQuaternionError(gelatin_scarf_trans_y_resampled_qx, gelatin_scarf_trans_y_resampled_qy, gelatin_scarf_trans_y_resampled_qz, gelatin_scarf_trans_y_resampled_qw, gelatin_gt_trans_y['qx'], gelatin_gt_trans_y['qy'], gelatin_gt_trans_y['qz'], gelatin_gt_trans_y['qw'])
gelatin_scarf_q_angle_trans_z = computeQuaternionError(gelatin_scarf_trans_z_resampled_qx, gelatin_scarf_trans_z_resampled_qy, gelatin_scarf_trans_z_resampled_qz, gelatin_scarf_trans_z_resampled_qw, gelatin_gt_trans_z['qx'], gelatin_gt_trans_z['qy'], gelatin_gt_trans_z['qz'], gelatin_gt_trans_z['qw'])
gelatin_scarf_q_angle_roll = computeQuaternionError(gelatin_scarf_roll_resampled_qx, gelatin_scarf_roll_resampled_qy, gelatin_scarf_roll_resampled_qz, gelatin_scarf_roll_resampled_qw, gelatin_gt_roll['qx'], gelatin_gt_roll['qy'], gelatin_gt_roll['qz'], gelatin_gt_roll['qw'])
gelatin_scarf_q_angle_pitch = computeQuaternionError(gelatin_scarf_pitch_resampled_qx, gelatin_scarf_pitch_resampled_qy, gelatin_scarf_pitch_resampled_qz, gelatin_scarf_pitch_resampled_qw, gelatin_gt_pitch['qx'], gelatin_gt_pitch['qy'], gelatin_gt_pitch['qz'], gelatin_gt_pitch['qw'])
gelatin_scarf_q_angle_yaw = computeQuaternionError(gelatin_scarf_yaw_resampled_qx, gelatin_scarf_yaw_resampled_qy, gelatin_scarf_yaw_resampled_qz, gelatin_scarf_yaw_resampled_qw, gelatin_gt_yaw['qx'], gelatin_gt_yaw['qy'], gelatin_gt_yaw['qz'], gelatin_gt_yaw['qw'])

gelatin_scarf_position_errors = np.concatenate((gelatin_scarf_error_trans_x, gelatin_scarf_error_trans_y, gelatin_scarf_error_trans_z, gelatin_scarf_error_roll, gelatin_scarf_error_pitch, gelatin_scarf_error_yaw))
gelatin_scarf_rotation_errors = np.concatenate((gelatin_scarf_q_angle_trans_x, gelatin_scarf_q_angle_trans_y, gelatin_scarf_q_angle_trans_z, gelatin_scarf_q_angle_roll, gelatin_scarf_q_angle_pitch, gelatin_scarf_q_angle_yaw))


# ---------------------------------------------------------------------------  MUSTARD  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/scarf-journal/mustard/'
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

mustard_gt_trans_x_alpha,mustard_gt_trans_x_beta,mustard_gt_trans_x_gamma = quaternion_to_euler_angle(mustard_gt_trans_x['qw'], mustard_gt_trans_x['qx'], mustard_gt_trans_x['qy'], mustard_gt_trans_x['qz'])
mustard_gt_trans_x_alpha_cleaned = cleanEuler(mustard_gt_trans_x_alpha,0)
mustard_gt_trans_x_beta_cleaned = cleanEuler(mustard_gt_trans_x_beta,1)
mustard_gt_trans_x_gamma_cleaned = cleanEuler(mustard_gt_trans_x_gamma,1)

mustard_gt_trans_y_alpha,mustard_gt_trans_y_beta,mustard_gt_trans_y_gamma = quaternion_to_euler_angle(mustard_gt_trans_y['qw'], mustard_gt_trans_y['qx'], mustard_gt_trans_y['qy'], mustard_gt_trans_y['qz'])
mustard_gt_trans_y_alpha_cleaned = cleanEuler(mustard_gt_trans_y_alpha,0)
mustard_gt_trans_y_beta_cleaned = cleanEuler(mustard_gt_trans_y_beta,1)
mustard_gt_trans_y_gamma_cleaned = cleanEuler(mustard_gt_trans_y_gamma,2)

mustard_gt_trans_z_alpha,mustard_gt_trans_z_beta,mustard_gt_trans_z_gamma = quaternion_to_euler_angle(mustard_gt_trans_z['qw'], mustard_gt_trans_z['qx'], mustard_gt_trans_z['qy'], mustard_gt_trans_z['qz'])
mustard_gt_trans_z_alpha_cleaned = cleanEuler(mustard_gt_trans_z_alpha,0)
mustard_gt_trans_z_beta_cleaned = cleanEuler(mustard_gt_trans_z_beta,1)
mustard_gt_trans_z_gamma_cleaned = cleanEuler(mustard_gt_trans_z_gamma,2)

mustard_gt_roll_alpha,mustard_gt_roll_beta,mustard_gt_roll_gamma = quaternion_to_euler_angle(mustard_gt_roll['qw'], mustard_gt_roll['qx'], mustard_gt_roll['qy'], mustard_gt_roll['qz'])
mustard_gt_roll_alpha_cleaned = cleanEuler(mustard_gt_roll_alpha,0)
mustard_gt_roll_beta_cleaned = cleanEuler(mustard_gt_roll_beta,1)
mustard_gt_roll_gamma_cleaned = cleanEuler(mustard_gt_roll_gamma,2)

mustard_gt_pitch_alpha,mustard_gt_pitch_beta,mustard_gt_pitch_gamma = quaternion_to_euler_angle(mustard_gt_pitch['qw'], mustard_gt_pitch['qx'], mustard_gt_pitch['qy'], mustard_gt_pitch['qz'])
mustard_gt_pitch_alpha_cleaned = cleanEuler(mustard_gt_pitch_alpha,0)
mustard_gt_pitch_beta_cleaned = cleanEuler(mustard_gt_pitch_beta,1)
mustard_gt_pitch_gamma_cleaned = cleanEuler(mustard_gt_pitch_gamma,2)

mustard_gt_yaw_alpha,mustard_gt_yaw_beta,mustard_gt_yaw_gamma = quaternion_to_euler_angle(mustard_gt_yaw['qw'], mustard_gt_yaw['qx'], mustard_gt_yaw['qy'], mustard_gt_yaw['qz'])
mustard_gt_yaw_alpha_cleaned = cleanEuler(mustard_gt_yaw_alpha,0)
mustard_gt_yaw_beta_cleaned = cleanEuler(mustard_gt_yaw_beta,1)
mustard_gt_yaw_gamma_cleaned = cleanEuler(mustard_gt_yaw_gamma,2)

mustard_eros_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_eros_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_eros_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_eros_roll = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_eros_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_eros_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

mustard_eros_trans_x_resampled_x = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_eros_trans_x['t'], mustard_eros_trans_x['x'])
mustard_eros_trans_x_resampled_y = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_eros_trans_x['t'], mustard_eros_trans_x['y'])
mustard_eros_trans_x_resampled_z = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_eros_trans_x['t'], mustard_eros_trans_x['z'])
mustard_eros_trans_x_resampled_qx = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_eros_trans_x['t'], mustard_eros_trans_x['qx'])
mustard_eros_trans_x_resampled_qy = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_eros_trans_x['t'], mustard_eros_trans_x['qy'])
mustard_eros_trans_x_resampled_qz = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_eros_trans_x['t'], mustard_eros_trans_x['qz'])
mustard_eros_trans_x_resampled_qw = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_eros_trans_x['t'], mustard_eros_trans_x['qw'])

mustard_eros_trans_y_resampled_x = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_eros_trans_y['t'], mustard_eros_trans_y['x'])
mustard_eros_trans_y_resampled_y = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_eros_trans_y['t'], mustard_eros_trans_y['y'])
mustard_eros_trans_y_resampled_z = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_eros_trans_y['t'], mustard_eros_trans_y['z'])
mustard_eros_trans_y_resampled_qx = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_eros_trans_y['t'], mustard_eros_trans_y['qx'])
mustard_eros_trans_y_resampled_qy = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_eros_trans_y['t'], mustard_eros_trans_y['qy'])
mustard_eros_trans_y_resampled_qz = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_eros_trans_y['t'], mustard_eros_trans_y['qz'])
mustard_eros_trans_y_resampled_qw = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_eros_trans_y['t'], mustard_eros_trans_y['qw'])

mustard_eros_trans_z_resampled_x = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_eros_trans_z['t'], mustard_eros_trans_z['x'])
mustard_eros_trans_z_resampled_y = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_eros_trans_z['t'], mustard_eros_trans_z['y'])
mustard_eros_trans_z_resampled_z = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_eros_trans_z['t'], mustard_eros_trans_z['z'])
mustard_eros_trans_z_resampled_qx = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_eros_trans_z['t'], mustard_eros_trans_z['qx'])
mustard_eros_trans_z_resampled_qy = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_eros_trans_z['t'], mustard_eros_trans_z['qy'])
mustard_eros_trans_z_resampled_qz = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_eros_trans_z['t'], mustard_eros_trans_z['qz'])
mustard_eros_trans_z_resampled_qw = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_eros_trans_z['t'], mustard_eros_trans_z['qw'])

mustard_eros_roll_resampled_x = resampling_by_interpolate(mustard_gt_roll['t'], mustard_eros_roll['t'], mustard_eros_roll['x'])
mustard_eros_roll_resampled_y = resampling_by_interpolate(mustard_gt_roll['t'], mustard_eros_roll['t'], mustard_eros_roll['y'])
mustard_eros_roll_resampled_z = resampling_by_interpolate(mustard_gt_roll['t'], mustard_eros_roll['t'], mustard_eros_roll['z'])
mustard_eros_roll_resampled_qx = resampling_by_interpolate(mustard_gt_roll['t'], mustard_eros_roll['t'], mustard_eros_roll['qx'])
mustard_eros_roll_resampled_qy = resampling_by_interpolate(mustard_gt_roll['t'], mustard_eros_roll['t'], mustard_eros_roll['qy'])
mustard_eros_roll_resampled_qz = resampling_by_interpolate(mustard_gt_roll['t'], mustard_eros_roll['t'], mustard_eros_roll['qz'])
mustard_eros_roll_resampled_qw = resampling_by_interpolate(mustard_gt_roll['t'], mustard_eros_roll['t'], mustard_eros_roll['qw'])

mustard_eros_pitch_resampled_x = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_eros_pitch['t'], mustard_eros_pitch['x'])
mustard_eros_pitch_resampled_y = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_eros_pitch['t'], mustard_eros_pitch['y'])
mustard_eros_pitch_resampled_z = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_eros_pitch['t'], mustard_eros_pitch['z'])
mustard_eros_pitch_resampled_qx = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_eros_pitch['t'], mustard_eros_pitch['qx'])
mustard_eros_pitch_resampled_qy = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_eros_pitch['t'], mustard_eros_pitch['qy'])
mustard_eros_pitch_resampled_qz = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_eros_pitch['t'], mustard_eros_pitch['qz'])
mustard_eros_pitch_resampled_qw = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_eros_pitch['t'], mustard_eros_pitch['qw'])

mustard_eros_yaw_resampled_x = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_eros_yaw['t'], mustard_eros_yaw['x'])
mustard_eros_yaw_resampled_y = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_eros_yaw['t'], mustard_eros_yaw['y'])
mustard_eros_yaw_resampled_z = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_eros_yaw['t'], mustard_eros_yaw['z'])
mustard_eros_yaw_resampled_qx = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_eros_yaw['t'], mustard_eros_yaw['qx'])
mustard_eros_yaw_resampled_qy = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_eros_yaw['t'], mustard_eros_yaw['qy'])
mustard_eros_yaw_resampled_qz = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_eros_yaw['t'], mustard_eros_yaw['qz'])
mustard_eros_yaw_resampled_qw = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_eros_yaw['t'], mustard_eros_yaw['qw'])

mustard_eros_trans_x_alpha,mustard_eros_trans_x_beta,mustard_eros_trans_x_gamma = quaternion_to_euler_angle(mustard_eros_trans_x['qw'], mustard_eros_trans_x['qx'], mustard_eros_trans_x['qy'], mustard_eros_trans_x['qz'])
mustard_eros_trans_x_alpha_cleaned = cleanEuler(mustard_eros_trans_x_alpha,0)
mustard_eros_trans_x_beta_cleaned = cleanEuler(mustard_eros_trans_x_beta,1)
mustard_eros_trans_x_gamma_cleaned = cleanEuler(mustard_eros_trans_x_gamma,2)

mustard_eros_trans_y_alpha,mustard_eros_trans_y_beta,mustard_eros_trans_y_gamma = quaternion_to_euler_angle(mustard_eros_trans_y['qw'], mustard_eros_trans_y['qx'], mustard_eros_trans_y['qy'], mustard_eros_trans_y['qz'])
mustard_eros_trans_y_alpha_cleaned = cleanEuler(mustard_eros_trans_y_alpha,0)
mustard_eros_trans_y_beta_cleaned = cleanEuler(mustard_eros_trans_y_beta,1)
mustard_eros_trans_y_gamma_cleaned = cleanEuler(mustard_eros_trans_y_gamma,2)

mustard_eros_trans_z_alpha,mustard_eros_trans_z_beta,mustard_eros_trans_z_gamma = quaternion_to_euler_angle(mustard_eros_trans_z['qw'], mustard_eros_trans_z['qx'], mustard_eros_trans_z['qy'], mustard_eros_trans_z['qz'])
mustard_eros_trans_z_alpha_cleaned = cleanEuler(mustard_eros_trans_z_alpha,0)
mustard_eros_trans_z_beta_cleaned = cleanEuler(mustard_eros_trans_z_beta,1)
mustard_eros_trans_z_gamma_cleaned = cleanEuler(mustard_eros_trans_z_gamma,2)

mustard_eros_roll_alpha,mustard_eros_roll_beta,mustard_eros_roll_gamma = quaternion_to_euler_angle(mustard_eros_roll['qw'], mustard_eros_roll['qx'], mustard_eros_roll['qy'], mustard_eros_roll['qz'])
mustard_eros_roll_alpha_cleaned = cleanEuler(mustard_eros_roll_alpha,0)
mustard_eros_roll_beta_cleaned = cleanEuler(mustard_eros_roll_beta,1)
mustard_eros_roll_gamma_cleaned = cleanEuler(mustard_eros_roll_gamma,2)

mustard_eros_pitch_alpha,mustard_eros_pitch_beta,mustard_eros_pitch_gamma = quaternion_to_euler_angle(mustard_eros_pitch['qw'], mustard_eros_pitch['qx'], mustard_eros_pitch['qy'], mustard_eros_pitch['qz'])
mustard_eros_pitch_alpha_cleaned = cleanEuler(mustard_eros_pitch_alpha,0)
mustard_eros_pitch_beta_cleaned = cleanEuler(mustard_eros_pitch_beta,1)
mustard_eros_pitch_gamma_cleaned = cleanEuler(mustard_eros_pitch_gamma,2)

mustard_eros_yaw_alpha,mustard_eros_yaw_beta,mustard_eros_yaw_gamma = quaternion_to_euler_angle(mustard_eros_yaw['qw'], mustard_eros_yaw['qx'], mustard_eros_yaw['qy'], mustard_eros_yaw['qz'])
mustard_eros_yaw_alpha_cleaned = cleanEuler(mustard_eros_yaw_alpha,0)
mustard_eros_yaw_beta_cleaned = cleanEuler(mustard_eros_yaw_beta,1)
mustard_eros_yaw_gamma_cleaned = cleanEuler(mustard_eros_yaw_gamma,2)

mustard_eros_error_trans_x = computeEuclideanDistance(mustard_gt_trans_x['x'], mustard_gt_trans_x['y'], mustard_gt_trans_x['z'], mustard_eros_trans_x_resampled_x, mustard_eros_trans_x_resampled_y, mustard_eros_trans_x_resampled_z)
mustard_eros_error_trans_y = computeEuclideanDistance(mustard_gt_trans_y['x'], mustard_gt_trans_y['y'], mustard_gt_trans_y['z'], mustard_eros_trans_y_resampled_x, mustard_eros_trans_y_resampled_y, mustard_eros_trans_y_resampled_z)
mustard_eros_error_trans_z = computeEuclideanDistance(mustard_gt_trans_z['x'], mustard_gt_trans_z['y'], mustard_gt_trans_z['z'], mustard_eros_trans_z_resampled_x, mustard_eros_trans_z_resampled_y, mustard_eros_trans_z_resampled_z)
mustard_eros_error_roll = computeEuclideanDistance(mustard_gt_roll['x'], mustard_gt_roll['y'], mustard_gt_roll['z'], mustard_eros_roll_resampled_x, mustard_eros_roll_resampled_y, mustard_eros_roll_resampled_z)
mustard_eros_error_pitch = computeEuclideanDistance(mustard_gt_pitch['x'], mustard_gt_pitch['y'], mustard_gt_pitch['z'], mustard_eros_pitch_resampled_x, mustard_eros_pitch_resampled_y, mustard_eros_pitch_resampled_z)
mustard_eros_error_yaw = computeEuclideanDistance(mustard_gt_yaw['x'], mustard_gt_yaw['y'], mustard_gt_yaw['z'], mustard_eros_yaw_resampled_x, mustard_eros_yaw_resampled_y, mustard_eros_yaw_resampled_z)

mustard_eros_q_angle_trans_x = computeQuaternionError(mustard_eros_trans_x_resampled_qx, mustard_eros_trans_x_resampled_qy, mustard_eros_trans_x_resampled_qz, mustard_eros_trans_x_resampled_qw, mustard_gt_trans_x['qx'], mustard_gt_trans_x['qy'], mustard_gt_trans_x['qz'], mustard_gt_trans_x['qw'])
mustard_eros_q_angle_trans_y = computeQuaternionError(mustard_eros_trans_y_resampled_qx, mustard_eros_trans_y_resampled_qy, mustard_eros_trans_y_resampled_qz, mustard_eros_trans_y_resampled_qw, mustard_gt_trans_y['qx'], mustard_gt_trans_y['qy'], mustard_gt_trans_y['qz'], mustard_gt_trans_y['qw'])
mustard_eros_q_angle_trans_z = computeQuaternionError(mustard_eros_trans_z_resampled_qx, mustard_eros_trans_z_resampled_qy, mustard_eros_trans_z_resampled_qz, mustard_eros_trans_z_resampled_qw, mustard_gt_trans_z['qx'], mustard_gt_trans_z['qy'], mustard_gt_trans_z['qz'], mustard_gt_trans_z['qw'])
mustard_eros_q_angle_roll = computeQuaternionError(mustard_eros_roll_resampled_qx, mustard_eros_roll_resampled_qy, mustard_eros_roll_resampled_qz, mustard_eros_roll_resampled_qw, mustard_gt_roll['qx'], mustard_gt_roll['qy'], mustard_gt_roll['qz'], mustard_gt_roll['qw'])
mustard_eros_q_angle_pitch = computeQuaternionError(mustard_eros_pitch_resampled_qx, mustard_eros_pitch_resampled_qy, mustard_eros_pitch_resampled_qz, mustard_eros_pitch_resampled_qw, mustard_gt_pitch['qx'], mustard_gt_pitch['qy'], mustard_gt_pitch['qz'], mustard_gt_pitch['qw'])
mustard_eros_q_angle_yaw = computeQuaternionError(mustard_eros_yaw_resampled_qx, mustard_eros_yaw_resampled_qy, mustard_eros_yaw_resampled_qz, mustard_eros_yaw_resampled_qw, mustard_gt_yaw['qx'], mustard_gt_yaw['qy'], mustard_gt_yaw['qz'], mustard_gt_yaw['qw'])

mustard_eros_position_errors = np.concatenate((mustard_eros_error_trans_x, mustard_eros_error_trans_y, mustard_eros_error_trans_z, mustard_eros_error_roll, mustard_eros_error_pitch, mustard_eros_error_yaw))
mustard_eros_rotation_errors = np.concatenate((mustard_eros_q_angle_trans_x, mustard_eros_q_angle_trans_y, mustard_eros_q_angle_trans_z, mustard_eros_q_angle_roll, mustard_eros_q_angle_pitch, mustard_eros_q_angle_yaw))

mustard_scarf_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_scarf_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_scarf_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_scarf_roll = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_scarf_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
mustard_scarf_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

mustard_scarf_trans_x_resampled_x = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_scarf_trans_x['t'], mustard_scarf_trans_x['x'])
mustard_scarf_trans_x_resampled_y = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_scarf_trans_x['t'], mustard_scarf_trans_x['y'])
mustard_scarf_trans_x_resampled_z = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_scarf_trans_x['t'], mustard_scarf_trans_x['z'])
mustard_scarf_trans_x_resampled_qx = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_scarf_trans_x['t'], mustard_scarf_trans_x['qx'])
mustard_scarf_trans_x_resampled_qy = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_scarf_trans_x['t'], mustard_scarf_trans_x['qy'])
mustard_scarf_trans_x_resampled_qz = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_scarf_trans_x['t'], mustard_scarf_trans_x['qz'])
mustard_scarf_trans_x_resampled_qw = resampling_by_interpolate(mustard_gt_trans_x['t'], mustard_scarf_trans_x['t'], mustard_scarf_trans_x['qw'])

mustard_scarf_trans_y_resampled_x = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_scarf_trans_y['t'], mustard_scarf_trans_y['x'])
mustard_scarf_trans_y_resampled_y = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_scarf_trans_y['t'], mustard_scarf_trans_y['y'])
mustard_scarf_trans_y_resampled_z = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_scarf_trans_y['t'], mustard_scarf_trans_y['z'])
mustard_scarf_trans_y_resampled_qx = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_scarf_trans_y['t'], mustard_scarf_trans_y['qx'])
mustard_scarf_trans_y_resampled_qy = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_scarf_trans_y['t'], mustard_scarf_trans_y['qy'])
mustard_scarf_trans_y_resampled_qz = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_scarf_trans_y['t'], mustard_scarf_trans_y['qz'])
mustard_scarf_trans_y_resampled_qw = resampling_by_interpolate(mustard_gt_trans_y['t'], mustard_scarf_trans_y['t'], mustard_scarf_trans_y['qw'])

mustard_scarf_trans_z_resampled_x = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_scarf_trans_z['t'], mustard_scarf_trans_z['x'])
mustard_scarf_trans_z_resampled_y = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_scarf_trans_z['t'], mustard_scarf_trans_z['y'])
mustard_scarf_trans_z_resampled_z = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_scarf_trans_z['t'], mustard_scarf_trans_z['z'])
mustard_scarf_trans_z_resampled_qx = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_scarf_trans_z['t'], mustard_scarf_trans_z['qx'])
mustard_scarf_trans_z_resampled_qy = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_scarf_trans_z['t'], mustard_scarf_trans_z['qy'])
mustard_scarf_trans_z_resampled_qz = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_scarf_trans_z['t'], mustard_scarf_trans_z['qz'])
mustard_scarf_trans_z_resampled_qw = resampling_by_interpolate(mustard_gt_trans_z['t'], mustard_scarf_trans_z['t'], mustard_scarf_trans_z['qw'])

mustard_scarf_roll_resampled_x = resampling_by_interpolate(mustard_gt_roll['t'], mustard_scarf_roll['t'], mustard_scarf_roll['x'])
mustard_scarf_roll_resampled_y = resampling_by_interpolate(mustard_gt_roll['t'], mustard_scarf_roll['t'], mustard_scarf_roll['y'])
mustard_scarf_roll_resampled_z = resampling_by_interpolate(mustard_gt_roll['t'], mustard_scarf_roll['t'], mustard_scarf_roll['z'])
mustard_scarf_roll_resampled_qx = resampling_by_interpolate(mustard_gt_roll['t'], mustard_scarf_roll['t'], mustard_scarf_roll['qx'])
mustard_scarf_roll_resampled_qy = resampling_by_interpolate(mustard_gt_roll['t'], mustard_scarf_roll['t'], mustard_scarf_roll['qy'])
mustard_scarf_roll_resampled_qz = resampling_by_interpolate(mustard_gt_roll['t'], mustard_scarf_roll['t'], mustard_scarf_roll['qz'])
mustard_scarf_roll_resampled_qw = resampling_by_interpolate(mustard_gt_roll['t'], mustard_scarf_roll['t'], mustard_scarf_roll['qw'])

mustard_scarf_pitch_resampled_x = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_scarf_pitch['t'], mustard_scarf_pitch['x'])
mustard_scarf_pitch_resampled_y = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_scarf_pitch['t'], mustard_scarf_pitch['y'])
mustard_scarf_pitch_resampled_z = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_scarf_pitch['t'], mustard_scarf_pitch['z'])
mustard_scarf_pitch_resampled_qx = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_scarf_pitch['t'], mustard_scarf_pitch['qx'])
mustard_scarf_pitch_resampled_qy = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_scarf_pitch['t'], mustard_scarf_pitch['qy'])
mustard_scarf_pitch_resampled_qz = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_scarf_pitch['t'], mustard_scarf_pitch['qz'])
mustard_scarf_pitch_resampled_qw = resampling_by_interpolate(mustard_gt_pitch['t'], mustard_scarf_pitch['t'], mustard_scarf_pitch['qw'])

mustard_scarf_yaw_resampled_x = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_scarf_yaw['t'], mustard_scarf_yaw['x'])
mustard_scarf_yaw_resampled_y = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_scarf_yaw['t'], mustard_scarf_yaw['y'])
mustard_scarf_yaw_resampled_z = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_scarf_yaw['t'], mustard_scarf_yaw['z'])
mustard_scarf_yaw_resampled_qx = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_scarf_yaw['t'], mustard_scarf_yaw['qx'])
mustard_scarf_yaw_resampled_qy = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_scarf_yaw['t'], mustard_scarf_yaw['qy'])
mustard_scarf_yaw_resampled_qz = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_scarf_yaw['t'], mustard_scarf_yaw['qz'])
mustard_scarf_yaw_resampled_qw = resampling_by_interpolate(mustard_gt_yaw['t'], mustard_scarf_yaw['t'], mustard_scarf_yaw['qw'])

mustard_scarf_trans_x_alpha,mustard_scarf_trans_x_beta,mustard_scarf_trans_x_gamma = quaternion_to_euler_angle(mustard_scarf_trans_x['qw'], mustard_scarf_trans_x['qx'], mustard_scarf_trans_x['qy'], mustard_scarf_trans_x['qz'])
mustard_scarf_trans_x_alpha_cleaned = cleanEuler(mustard_scarf_trans_x_alpha,0)
mustard_scarf_trans_x_beta_cleaned = cleanEuler(mustard_scarf_trans_x_beta,1)
mustard_scarf_trans_x_gamma_cleaned = cleanEuler(mustard_scarf_trans_x_gamma,2)

mustard_scarf_trans_y_alpha,mustard_scarf_trans_y_beta,mustard_scarf_trans_y_gamma = quaternion_to_euler_angle(mustard_scarf_trans_y['qw'], mustard_scarf_trans_y['qx'], mustard_scarf_trans_y['qy'], mustard_scarf_trans_y['qz'])
mustard_scarf_trans_y_alpha_cleaned = cleanEuler(mustard_scarf_trans_y_alpha,0)
mustard_scarf_trans_y_beta_cleaned = cleanEuler(mustard_scarf_trans_y_beta,1)
mustard_scarf_trans_y_gamma_cleaned = cleanEuler(mustard_scarf_trans_y_gamma,2)

mustard_scarf_trans_z_alpha,mustard_scarf_trans_z_beta,mustard_scarf_trans_z_gamma = quaternion_to_euler_angle(mustard_scarf_trans_z['qw'], mustard_scarf_trans_z['qx'], mustard_scarf_trans_z['qy'], mustard_scarf_trans_z['qz'])
mustard_scarf_trans_z_alpha_cleaned = cleanEuler(mustard_scarf_trans_z_alpha,0)
mustard_scarf_trans_z_beta_cleaned = cleanEuler(mustard_scarf_trans_z_beta,1)
mustard_scarf_trans_z_gamma_cleaned = cleanEuler(mustard_scarf_trans_z_gamma,2)

mustard_scarf_roll_alpha,mustard_scarf_roll_beta,mustard_scarf_roll_gamma = quaternion_to_euler_angle(mustard_scarf_roll['qw'], mustard_scarf_roll['qx'], mustard_scarf_roll['qy'], mustard_scarf_roll['qz'])
mustard_scarf_roll_alpha_cleaned = cleanEuler(mustard_scarf_roll_alpha,0)
mustard_scarf_roll_beta_cleaned = cleanEuler(mustard_scarf_roll_beta,1)
mustard_scarf_roll_gamma_cleaned = cleanEuler(mustard_scarf_roll_gamma,2)

mustard_scarf_pitch_alpha,mustard_scarf_pitch_beta,mustard_scarf_pitch_gamma = quaternion_to_euler_angle(mustard_scarf_pitch['qw'], mustard_scarf_pitch['qx'], mustard_scarf_pitch['qy'], mustard_scarf_pitch['qz'])
mustard_scarf_pitch_alpha_cleaned = cleanEuler(mustard_scarf_pitch_alpha,0)
mustard_scarf_pitch_beta_cleaned = cleanEuler(mustard_scarf_pitch_beta,1)
mustard_scarf_pitch_gamma_cleaned = cleanEuler(mustard_scarf_pitch_gamma,2)

mustard_scarf_yaw_alpha,mustard_scarf_yaw_beta,mustard_scarf_yaw_gamma = quaternion_to_euler_angle(mustard_scarf_yaw['qw'], mustard_scarf_yaw['qx'], mustard_scarf_yaw['qy'], mustard_scarf_yaw['qz'])
mustard_scarf_yaw_alpha_cleaned = cleanEuler(mustard_scarf_yaw_alpha,0)
mustard_scarf_yaw_beta_cleaned = cleanEuler(mustard_scarf_yaw_beta,1)
mustard_scarf_yaw_gamma_cleaned = cleanEuler(mustard_scarf_yaw_gamma,2)

mustard_scarf_error_trans_x = computeEuclideanDistance(mustard_gt_trans_x['x'], mustard_gt_trans_x['y'], mustard_gt_trans_x['z'], mustard_scarf_trans_x_resampled_x, mustard_scarf_trans_x_resampled_y, mustard_scarf_trans_x_resampled_z)
mustard_scarf_error_trans_y = computeEuclideanDistance(mustard_gt_trans_y['x'], mustard_gt_trans_y['y'], mustard_gt_trans_y['z'], mustard_scarf_trans_y_resampled_x, mustard_scarf_trans_y_resampled_y, mustard_scarf_trans_y_resampled_z)
mustard_scarf_error_trans_z = computeEuclideanDistance(mustard_gt_trans_z['x'], mustard_gt_trans_z['y'], mustard_gt_trans_z['z'], mustard_scarf_trans_z_resampled_x, mustard_scarf_trans_z_resampled_y, mustard_scarf_trans_z_resampled_z)
mustard_scarf_error_roll = computeEuclideanDistance(mustard_gt_roll['x'], mustard_gt_roll['y'], mustard_gt_roll['z'], mustard_scarf_roll_resampled_x, mustard_scarf_roll_resampled_y, mustard_scarf_roll_resampled_z)
mustard_scarf_error_pitch = computeEuclideanDistance(mustard_gt_pitch['x'], mustard_gt_pitch['y'], mustard_gt_pitch['z'], mustard_scarf_pitch_resampled_x, mustard_scarf_pitch_resampled_y, mustard_scarf_pitch_resampled_z)
mustard_scarf_error_yaw = computeEuclideanDistance(mustard_gt_yaw['x'], mustard_gt_yaw['y'], mustard_gt_yaw['z'], mustard_scarf_yaw_resampled_x, mustard_scarf_yaw_resampled_y, mustard_scarf_yaw_resampled_z)

mustard_scarf_q_angle_trans_x = computeQuaternionError(mustard_scarf_trans_x_resampled_qx, mustard_scarf_trans_x_resampled_qy, mustard_scarf_trans_x_resampled_qz, mustard_scarf_trans_x_resampled_qw, mustard_gt_trans_x['qx'], mustard_gt_trans_x['qy'], mustard_gt_trans_x['qz'], mustard_gt_trans_x['qw'])
mustard_scarf_q_angle_trans_y = computeQuaternionError(mustard_scarf_trans_y_resampled_qx, mustard_scarf_trans_y_resampled_qy, mustard_scarf_trans_y_resampled_qz, mustard_scarf_trans_y_resampled_qw, mustard_gt_trans_y['qx'], mustard_gt_trans_y['qy'], mustard_gt_trans_y['qz'], mustard_gt_trans_y['qw'])
mustard_scarf_q_angle_trans_z = computeQuaternionError(mustard_scarf_trans_z_resampled_qx, mustard_scarf_trans_z_resampled_qy, mustard_scarf_trans_z_resampled_qz, mustard_scarf_trans_z_resampled_qw, mustard_gt_trans_z['qx'], mustard_gt_trans_z['qy'], mustard_gt_trans_z['qz'], mustard_gt_trans_z['qw'])
mustard_scarf_q_angle_roll = computeQuaternionError(mustard_scarf_roll_resampled_qx, mustard_scarf_roll_resampled_qy, mustard_scarf_roll_resampled_qz, mustard_scarf_roll_resampled_qw, mustard_gt_roll['qx'], mustard_gt_roll['qy'], mustard_gt_roll['qz'], mustard_gt_roll['qw'])
mustard_scarf_q_angle_pitch = computeQuaternionError(mustard_scarf_pitch_resampled_qx, mustard_scarf_pitch_resampled_qy, mustard_scarf_pitch_resampled_qz, mustard_scarf_pitch_resampled_qw, mustard_gt_pitch['qx'], mustard_gt_pitch['qy'], mustard_gt_pitch['qz'], mustard_gt_pitch['qw'])
mustard_scarf_q_angle_yaw = computeQuaternionError(mustard_scarf_yaw_resampled_qx, mustard_scarf_yaw_resampled_qy, mustard_scarf_yaw_resampled_qz, mustard_scarf_yaw_resampled_qw, mustard_gt_yaw['qx'], mustard_gt_yaw['qy'], mustard_gt_yaw['qz'], mustard_gt_yaw['qw'])

mustard_scarf_position_errors = np.concatenate((mustard_scarf_error_trans_x, mustard_scarf_error_trans_y, mustard_scarf_error_trans_z, mustard_scarf_error_roll, mustard_scarf_error_pitch, mustard_scarf_error_yaw))
mustard_scarf_rotation_errors = np.concatenate((mustard_scarf_q_angle_trans_x, mustard_scarf_q_angle_trans_y, mustard_scarf_q_angle_trans_z, mustard_scarf_q_angle_roll, mustard_scarf_q_angle_pitch, mustard_scarf_q_angle_yaw))

# ---------------------------------------------------------------------------  TOMATO  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/scarf-journal/tomato/'
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

tomato_gt_trans_x_alpha,tomato_gt_trans_x_beta,tomato_gt_trans_x_gamma = quaternion_to_euler_angle(tomato_gt_trans_x['qw'], tomato_gt_trans_x['qx'], tomato_gt_trans_x['qy'], tomato_gt_trans_x['qz'])
tomato_gt_trans_x_alpha_cleaned = cleanEuler(tomato_gt_trans_x_alpha,0)
tomato_gt_trans_x_beta_cleaned = cleanEuler(tomato_gt_trans_x_beta,1)
tomato_gt_trans_x_gamma_cleaned = cleanEuler(tomato_gt_trans_x_gamma,1)

tomato_gt_trans_y_alpha,tomato_gt_trans_y_beta,tomato_gt_trans_y_gamma = quaternion_to_euler_angle(tomato_gt_trans_y['qw'], tomato_gt_trans_y['qx'], tomato_gt_trans_y['qy'], tomato_gt_trans_y['qz'])
tomato_gt_trans_y_alpha_cleaned = cleanEuler(tomato_gt_trans_y_alpha,0)
tomato_gt_trans_y_beta_cleaned = cleanEuler(tomato_gt_trans_y_beta,1)
tomato_gt_trans_y_gamma_cleaned = cleanEuler(tomato_gt_trans_y_gamma,2)

tomato_gt_trans_z_alpha,tomato_gt_trans_z_beta,tomato_gt_trans_z_gamma = quaternion_to_euler_angle(tomato_gt_trans_z['qw'], tomato_gt_trans_z['qx'], tomato_gt_trans_z['qy'], tomato_gt_trans_z['qz'])
tomato_gt_trans_z_alpha_cleaned = cleanEuler(tomato_gt_trans_z_alpha,0)
tomato_gt_trans_z_beta_cleaned = cleanEuler(tomato_gt_trans_z_beta,1)
tomato_gt_trans_z_gamma_cleaned = cleanEuler(tomato_gt_trans_z_gamma,2)

tomato_gt_roll_alpha,tomato_gt_roll_beta,tomato_gt_roll_gamma = quaternion_to_euler_angle(tomato_gt_roll['qw'], tomato_gt_roll['qx'], tomato_gt_roll['qy'], tomato_gt_roll['qz'])
tomato_gt_roll_alpha_cleaned = cleanEuler(tomato_gt_roll_alpha,0)
tomato_gt_roll_beta_cleaned = cleanEuler(tomato_gt_roll_beta,1)
tomato_gt_roll_gamma_cleaned = cleanEuler(tomato_gt_roll_gamma,2)

tomato_gt_pitch_alpha,tomato_gt_pitch_beta,tomato_gt_pitch_gamma = quaternion_to_euler_angle(tomato_gt_pitch['qw'], tomato_gt_pitch['qx'], tomato_gt_pitch['qy'], tomato_gt_pitch['qz'])
tomato_gt_pitch_alpha_cleaned = cleanEuler(tomato_gt_pitch_alpha,0)
tomato_gt_pitch_beta_cleaned = cleanEuler(tomato_gt_pitch_beta,1)
tomato_gt_pitch_gamma_cleaned = cleanEuler(tomato_gt_pitch_gamma,2)

tomato_gt_yaw_alpha,tomato_gt_yaw_beta,tomato_gt_yaw_gamma = quaternion_to_euler_angle(tomato_gt_yaw['qw'], tomato_gt_yaw['qx'], tomato_gt_yaw['qy'], tomato_gt_yaw['qz'])
tomato_gt_yaw_alpha_cleaned = cleanEuler(tomato_gt_yaw_alpha,0)
tomato_gt_yaw_beta_cleaned = cleanEuler(tomato_gt_yaw_beta,1)
tomato_gt_yaw_gamma_cleaned = cleanEuler(tomato_gt_yaw_gamma,2)

tomato_eros_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_eros_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_eros_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_eros_roll = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_eros_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_eros_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

tomato_eros_trans_x_resampled_x = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_eros_trans_x['t'], tomato_eros_trans_x['x'])
tomato_eros_trans_x_resampled_y = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_eros_trans_x['t'], tomato_eros_trans_x['y'])
tomato_eros_trans_x_resampled_z = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_eros_trans_x['t'], tomato_eros_trans_x['z'])
tomato_eros_trans_x_resampled_qx = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_eros_trans_x['t'], tomato_eros_trans_x['qx'])
tomato_eros_trans_x_resampled_qy = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_eros_trans_x['t'], tomato_eros_trans_x['qy'])
tomato_eros_trans_x_resampled_qz = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_eros_trans_x['t'], tomato_eros_trans_x['qz'])
tomato_eros_trans_x_resampled_qw = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_eros_trans_x['t'], tomato_eros_trans_x['qw'])

tomato_eros_trans_y_resampled_x = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_eros_trans_y['t'], tomato_eros_trans_y['x'])
tomato_eros_trans_y_resampled_y = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_eros_trans_y['t'], tomato_eros_trans_y['y'])
tomato_eros_trans_y_resampled_z = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_eros_trans_y['t'], tomato_eros_trans_y['z'])
tomato_eros_trans_y_resampled_qx = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_eros_trans_y['t'], tomato_eros_trans_y['qx'])
tomato_eros_trans_y_resampled_qy = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_eros_trans_y['t'], tomato_eros_trans_y['qy'])
tomato_eros_trans_y_resampled_qz = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_eros_trans_y['t'], tomato_eros_trans_y['qz'])
tomato_eros_trans_y_resampled_qw = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_eros_trans_y['t'], tomato_eros_trans_y['qw'])

tomato_eros_trans_z_resampled_x = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_eros_trans_z['t'], tomato_eros_trans_z['x'])
tomato_eros_trans_z_resampled_y = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_eros_trans_z['t'], tomato_eros_trans_z['y'])
tomato_eros_trans_z_resampled_z = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_eros_trans_z['t'], tomato_eros_trans_z['z'])
tomato_eros_trans_z_resampled_qx = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_eros_trans_z['t'], tomato_eros_trans_z['qx'])
tomato_eros_trans_z_resampled_qy = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_eros_trans_z['t'], tomato_eros_trans_z['qy'])
tomato_eros_trans_z_resampled_qz = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_eros_trans_z['t'], tomato_eros_trans_z['qz'])
tomato_eros_trans_z_resampled_qw = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_eros_trans_z['t'], tomato_eros_trans_z['qw'])

tomato_eros_roll_resampled_x = resampling_by_interpolate(tomato_gt_roll['t'], tomato_eros_roll['t'], tomato_eros_roll['x'])
tomato_eros_roll_resampled_y = resampling_by_interpolate(tomato_gt_roll['t'], tomato_eros_roll['t'], tomato_eros_roll['y'])
tomato_eros_roll_resampled_z = resampling_by_interpolate(tomato_gt_roll['t'], tomato_eros_roll['t'], tomato_eros_roll['z'])
tomato_eros_roll_resampled_qx = resampling_by_interpolate(tomato_gt_roll['t'], tomato_eros_roll['t'], tomato_eros_roll['qx'])
tomato_eros_roll_resampled_qy = resampling_by_interpolate(tomato_gt_roll['t'], tomato_eros_roll['t'], tomato_eros_roll['qy'])
tomato_eros_roll_resampled_qz = resampling_by_interpolate(tomato_gt_roll['t'], tomato_eros_roll['t'], tomato_eros_roll['qz'])
tomato_eros_roll_resampled_qw = resampling_by_interpolate(tomato_gt_roll['t'], tomato_eros_roll['t'], tomato_eros_roll['qw'])

tomato_eros_pitch_resampled_x = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_eros_pitch['t'], tomato_eros_pitch['x'])
tomato_eros_pitch_resampled_y = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_eros_pitch['t'], tomato_eros_pitch['y'])
tomato_eros_pitch_resampled_z = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_eros_pitch['t'], tomato_eros_pitch['z'])
tomato_eros_pitch_resampled_qx = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_eros_pitch['t'], tomato_eros_pitch['qx'])
tomato_eros_pitch_resampled_qy = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_eros_pitch['t'], tomato_eros_pitch['qy'])
tomato_eros_pitch_resampled_qz = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_eros_pitch['t'], tomato_eros_pitch['qz'])
tomato_eros_pitch_resampled_qw = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_eros_pitch['t'], tomato_eros_pitch['qw'])

tomato_eros_yaw_resampled_x = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_eros_yaw['t'], tomato_eros_yaw['x'])
tomato_eros_yaw_resampled_y = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_eros_yaw['t'], tomato_eros_yaw['y'])
tomato_eros_yaw_resampled_z = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_eros_yaw['t'], tomato_eros_yaw['z'])
tomato_eros_yaw_resampled_qx = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_eros_yaw['t'], tomato_eros_yaw['qx'])
tomato_eros_yaw_resampled_qy = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_eros_yaw['t'], tomato_eros_yaw['qy'])
tomato_eros_yaw_resampled_qz = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_eros_yaw['t'], tomato_eros_yaw['qz'])
tomato_eros_yaw_resampled_qw = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_eros_yaw['t'], tomato_eros_yaw['qw'])

tomato_eros_trans_x_alpha,tomato_eros_trans_x_beta,tomato_eros_trans_x_gamma = quaternion_to_euler_angle(tomato_eros_trans_x['qw'], tomato_eros_trans_x['qx'], tomato_eros_trans_x['qy'], tomato_eros_trans_x['qz'])
tomato_eros_trans_x_alpha_cleaned = cleanEuler(tomato_eros_trans_x_alpha,0)
tomato_eros_trans_x_beta_cleaned = cleanEuler(tomato_eros_trans_x_beta,1)
tomato_eros_trans_x_gamma_cleaned = cleanEuler(tomato_eros_trans_x_gamma,2)

tomato_eros_trans_y_alpha,tomato_eros_trans_y_beta,tomato_eros_trans_y_gamma = quaternion_to_euler_angle(tomato_eros_trans_y['qw'], tomato_eros_trans_y['qx'], tomato_eros_trans_y['qy'], tomato_eros_trans_y['qz'])
tomato_eros_trans_y_alpha_cleaned = cleanEuler(tomato_eros_trans_y_alpha,0)
tomato_eros_trans_y_beta_cleaned = cleanEuler(tomato_eros_trans_y_beta,1)
tomato_eros_trans_y_gamma_cleaned = cleanEuler(tomato_eros_trans_y_gamma,2)

tomato_eros_trans_z_alpha,tomato_eros_trans_z_beta,tomato_eros_trans_z_gamma = quaternion_to_euler_angle(tomato_eros_trans_z['qw'], tomato_eros_trans_z['qx'], tomato_eros_trans_z['qy'], tomato_eros_trans_z['qz'])
tomato_eros_trans_z_alpha_cleaned = cleanEuler(tomato_eros_trans_z_alpha,0)
tomato_eros_trans_z_beta_cleaned = cleanEuler(tomato_eros_trans_z_beta,1)
tomato_eros_trans_z_gamma_cleaned = cleanEuler(tomato_eros_trans_z_gamma,2)

tomato_eros_roll_alpha,tomato_eros_roll_beta,tomato_eros_roll_gamma = quaternion_to_euler_angle(tomato_eros_roll['qw'], tomato_eros_roll['qx'], tomato_eros_roll['qy'], tomato_eros_roll['qz'])
tomato_eros_roll_alpha_cleaned = cleanEuler(tomato_eros_roll_alpha,0)
tomato_eros_roll_beta_cleaned = cleanEuler(tomato_eros_roll_beta,1)
tomato_eros_roll_gamma_cleaned = cleanEuler(tomato_eros_roll_gamma,2)

tomato_eros_pitch_alpha,tomato_eros_pitch_beta,tomato_eros_pitch_gamma = quaternion_to_euler_angle(tomato_eros_pitch['qw'], tomato_eros_pitch['qx'], tomato_eros_pitch['qy'], tomato_eros_pitch['qz'])
tomato_eros_pitch_alpha_cleaned = cleanEuler(tomato_eros_pitch_alpha,0)
tomato_eros_pitch_beta_cleaned = cleanEuler(tomato_eros_pitch_beta,1)
tomato_eros_pitch_gamma_cleaned = cleanEuler(tomato_eros_pitch_gamma,2)

tomato_eros_yaw_alpha,tomato_eros_yaw_beta,tomato_eros_yaw_gamma = quaternion_to_euler_angle(tomato_eros_yaw['qw'], tomato_eros_yaw['qx'], tomato_eros_yaw['qy'], tomato_eros_yaw['qz'])
tomato_eros_yaw_alpha_cleaned = cleanEuler(tomato_eros_yaw_alpha,0)
tomato_eros_yaw_beta_cleaned = cleanEuler(tomato_eros_yaw_beta,1)
tomato_eros_yaw_gamma_cleaned = cleanEuler(tomato_eros_yaw_gamma,2)

tomato_eros_error_trans_x = computeEuclideanDistance(tomato_gt_trans_x['x'], tomato_gt_trans_x['y'], tomato_gt_trans_x['z'], tomato_eros_trans_x_resampled_x, tomato_eros_trans_x_resampled_y, tomato_eros_trans_x_resampled_z)
tomato_eros_error_trans_y = computeEuclideanDistance(tomato_gt_trans_y['x'], tomato_gt_trans_y['y'], tomato_gt_trans_y['z'], tomato_eros_trans_y_resampled_x, tomato_eros_trans_y_resampled_y, tomato_eros_trans_y_resampled_z)
tomato_eros_error_trans_z = computeEuclideanDistance(tomato_gt_trans_z['x'], tomato_gt_trans_z['y'], tomato_gt_trans_z['z'], tomato_eros_trans_z_resampled_x, tomato_eros_trans_z_resampled_y, tomato_eros_trans_z_resampled_z)
tomato_eros_error_roll = computeEuclideanDistance(tomato_gt_roll['x'], tomato_gt_roll['y'], tomato_gt_roll['z'], tomato_eros_roll_resampled_x, tomato_eros_roll_resampled_y, tomato_eros_roll_resampled_z)
tomato_eros_error_pitch = computeEuclideanDistance(tomato_gt_pitch['x'], tomato_gt_pitch['y'], tomato_gt_pitch['z'], tomato_eros_pitch_resampled_x, tomato_eros_pitch_resampled_y, tomato_eros_pitch_resampled_z)
tomato_eros_error_yaw = computeEuclideanDistance(tomato_gt_yaw['x'], tomato_gt_yaw['y'], tomato_gt_yaw['z'], tomato_eros_yaw_resampled_x, tomato_eros_yaw_resampled_y, tomato_eros_yaw_resampled_z)

tomato_eros_q_angle_trans_x = computeQuaternionError(tomato_eros_trans_x_resampled_qx, tomato_eros_trans_x_resampled_qy, tomato_eros_trans_x_resampled_qz, tomato_eros_trans_x_resampled_qw, tomato_gt_trans_x['qx'], tomato_gt_trans_x['qy'], tomato_gt_trans_x['qz'], tomato_gt_trans_x['qw'])
tomato_eros_q_angle_trans_y = computeQuaternionError(tomato_eros_trans_y_resampled_qx, tomato_eros_trans_y_resampled_qy, tomato_eros_trans_y_resampled_qz, tomato_eros_trans_y_resampled_qw, tomato_gt_trans_y['qx'], tomato_gt_trans_y['qy'], tomato_gt_trans_y['qz'], tomato_gt_trans_y['qw'])
tomato_eros_q_angle_trans_z = computeQuaternionError(tomato_eros_trans_z_resampled_qx, tomato_eros_trans_z_resampled_qy, tomato_eros_trans_z_resampled_qz, tomato_eros_trans_z_resampled_qw, tomato_gt_trans_z['qx'], tomato_gt_trans_z['qy'], tomato_gt_trans_z['qz'], tomato_gt_trans_z['qw'])
tomato_eros_q_angle_roll = computeQuaternionError(tomato_eros_roll_resampled_qx, tomato_eros_roll_resampled_qy, tomato_eros_roll_resampled_qz, tomato_eros_roll_resampled_qw, tomato_gt_roll['qx'], tomato_gt_roll['qy'], tomato_gt_roll['qz'], tomato_gt_roll['qw'])
tomato_eros_q_angle_pitch = computeQuaternionError(tomato_eros_pitch_resampled_qx, tomato_eros_pitch_resampled_qy, tomato_eros_pitch_resampled_qz, tomato_eros_pitch_resampled_qw, tomato_gt_pitch['qx'], tomato_gt_pitch['qy'], tomato_gt_pitch['qz'], tomato_gt_pitch['qw'])
tomato_eros_q_angle_yaw = computeQuaternionError(tomato_eros_yaw_resampled_qx, tomato_eros_yaw_resampled_qy, tomato_eros_yaw_resampled_qz, tomato_eros_yaw_resampled_qw, tomato_gt_yaw['qx'], tomato_gt_yaw['qy'], tomato_gt_yaw['qz'], tomato_gt_yaw['qw'])

tomato_eros_position_errors = np.concatenate((tomato_eros_error_trans_x, tomato_eros_error_trans_y, tomato_eros_error_trans_z, tomato_eros_error_roll, tomato_eros_error_pitch, tomato_eros_error_yaw))
tomato_eros_rotation_errors = np.concatenate((tomato_eros_q_angle_trans_x, tomato_eros_q_angle_trans_y, tomato_eros_q_angle_trans_z, tomato_eros_q_angle_roll, tomato_eros_q_angle_pitch, tomato_eros_q_angle_yaw))

tomato_scarf_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_scarf_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_scarf_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_scarf_roll = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_scarf_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
tomato_scarf_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

tomato_scarf_trans_x_resampled_x = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_scarf_trans_x['t'], tomato_scarf_trans_x['x'])
tomato_scarf_trans_x_resampled_y = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_scarf_trans_x['t'], tomato_scarf_trans_x['y'])
tomato_scarf_trans_x_resampled_z = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_scarf_trans_x['t'], tomato_scarf_trans_x['z'])
tomato_scarf_trans_x_resampled_qx = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_scarf_trans_x['t'], tomato_scarf_trans_x['qx'])
tomato_scarf_trans_x_resampled_qy = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_scarf_trans_x['t'], tomato_scarf_trans_x['qy'])
tomato_scarf_trans_x_resampled_qz = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_scarf_trans_x['t'], tomato_scarf_trans_x['qz'])
tomato_scarf_trans_x_resampled_qw = resampling_by_interpolate(tomato_gt_trans_x['t'], tomato_scarf_trans_x['t'], tomato_scarf_trans_x['qw'])

tomato_scarf_trans_y_resampled_x = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_scarf_trans_y['t'], tomato_scarf_trans_y['x'])
tomato_scarf_trans_y_resampled_y = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_scarf_trans_y['t'], tomato_scarf_trans_y['y'])
tomato_scarf_trans_y_resampled_z = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_scarf_trans_y['t'], tomato_scarf_trans_y['z'])
tomato_scarf_trans_y_resampled_qx = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_scarf_trans_y['t'], tomato_scarf_trans_y['qx'])
tomato_scarf_trans_y_resampled_qy = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_scarf_trans_y['t'], tomato_scarf_trans_y['qy'])
tomato_scarf_trans_y_resampled_qz = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_scarf_trans_y['t'], tomato_scarf_trans_y['qz'])
tomato_scarf_trans_y_resampled_qw = resampling_by_interpolate(tomato_gt_trans_y['t'], tomato_scarf_trans_y['t'], tomato_scarf_trans_y['qw'])

tomato_scarf_trans_z_resampled_x = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_scarf_trans_z['t'], tomato_scarf_trans_z['x'])
tomato_scarf_trans_z_resampled_y = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_scarf_trans_z['t'], tomato_scarf_trans_z['y'])
tomato_scarf_trans_z_resampled_z = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_scarf_trans_z['t'], tomato_scarf_trans_z['z'])
tomato_scarf_trans_z_resampled_qx = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_scarf_trans_z['t'], tomato_scarf_trans_z['qx'])
tomato_scarf_trans_z_resampled_qy = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_scarf_trans_z['t'], tomato_scarf_trans_z['qy'])
tomato_scarf_trans_z_resampled_qz = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_scarf_trans_z['t'], tomato_scarf_trans_z['qz'])
tomato_scarf_trans_z_resampled_qw = resampling_by_interpolate(tomato_gt_trans_z['t'], tomato_scarf_trans_z['t'], tomato_scarf_trans_z['qw'])

tomato_scarf_roll_resampled_x = resampling_by_interpolate(tomato_gt_roll['t'], tomato_scarf_roll['t'], tomato_scarf_roll['x'])
tomato_scarf_roll_resampled_y = resampling_by_interpolate(tomato_gt_roll['t'], tomato_scarf_roll['t'], tomato_scarf_roll['y'])
tomato_scarf_roll_resampled_z = resampling_by_interpolate(tomato_gt_roll['t'], tomato_scarf_roll['t'], tomato_scarf_roll['z'])
tomato_scarf_roll_resampled_qx = resampling_by_interpolate(tomato_gt_roll['t'], tomato_scarf_roll['t'], tomato_scarf_roll['qx'])
tomato_scarf_roll_resampled_qy = resampling_by_interpolate(tomato_gt_roll['t'], tomato_scarf_roll['t'], tomato_scarf_roll['qy'])
tomato_scarf_roll_resampled_qz = resampling_by_interpolate(tomato_gt_roll['t'], tomato_scarf_roll['t'], tomato_scarf_roll['qz'])
tomato_scarf_roll_resampled_qw = resampling_by_interpolate(tomato_gt_roll['t'], tomato_scarf_roll['t'], tomato_scarf_roll['qw'])

tomato_scarf_pitch_resampled_x = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_scarf_pitch['t'], tomato_scarf_pitch['x'])
tomato_scarf_pitch_resampled_y = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_scarf_pitch['t'], tomato_scarf_pitch['y'])
tomato_scarf_pitch_resampled_z = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_scarf_pitch['t'], tomato_scarf_pitch['z'])
tomato_scarf_pitch_resampled_qx = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_scarf_pitch['t'], tomato_scarf_pitch['qx'])
tomato_scarf_pitch_resampled_qy = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_scarf_pitch['t'], tomato_scarf_pitch['qy'])
tomato_scarf_pitch_resampled_qz = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_scarf_pitch['t'], tomato_scarf_pitch['qz'])
tomato_scarf_pitch_resampled_qw = resampling_by_interpolate(tomato_gt_pitch['t'], tomato_scarf_pitch['t'], tomato_scarf_pitch['qw'])

tomato_scarf_yaw_resampled_x = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_scarf_yaw['t'], tomato_scarf_yaw['x'])
tomato_scarf_yaw_resampled_y = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_scarf_yaw['t'], tomato_scarf_yaw['y'])
tomato_scarf_yaw_resampled_z = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_scarf_yaw['t'], tomato_scarf_yaw['z'])
tomato_scarf_yaw_resampled_qx = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_scarf_yaw['t'], tomato_scarf_yaw['qx'])
tomato_scarf_yaw_resampled_qy = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_scarf_yaw['t'], tomato_scarf_yaw['qy'])
tomato_scarf_yaw_resampled_qz = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_scarf_yaw['t'], tomato_scarf_yaw['qz'])
tomato_scarf_yaw_resampled_qw = resampling_by_interpolate(tomato_gt_yaw['t'], tomato_scarf_yaw['t'], tomato_scarf_yaw['qw'])

tomato_scarf_trans_x_alpha,tomato_scarf_trans_x_beta,tomato_scarf_trans_x_gamma = quaternion_to_euler_angle(tomato_scarf_trans_x['qw'], tomato_scarf_trans_x['qx'], tomato_scarf_trans_x['qy'], tomato_scarf_trans_x['qz'])
tomato_scarf_trans_x_alpha_cleaned = cleanEuler(tomato_scarf_trans_x_alpha,0)
tomato_scarf_trans_x_beta_cleaned = cleanEuler(tomato_scarf_trans_x_beta,1)
tomato_scarf_trans_x_gamma_cleaned = cleanEuler(tomato_scarf_trans_x_gamma,2)

tomato_scarf_trans_y_alpha,tomato_scarf_trans_y_beta,tomato_scarf_trans_y_gamma = quaternion_to_euler_angle(tomato_scarf_trans_y['qw'], tomato_scarf_trans_y['qx'], tomato_scarf_trans_y['qy'], tomato_scarf_trans_y['qz'])
tomato_scarf_trans_y_alpha_cleaned = cleanEuler(tomato_scarf_trans_y_alpha,0)
tomato_scarf_trans_y_beta_cleaned = cleanEuler(tomato_scarf_trans_y_beta,1)
tomato_scarf_trans_y_gamma_cleaned = cleanEuler(tomato_scarf_trans_y_gamma,2)

tomato_scarf_trans_z_alpha,tomato_scarf_trans_z_beta,tomato_scarf_trans_z_gamma = quaternion_to_euler_angle(tomato_scarf_trans_z['qw'], tomato_scarf_trans_z['qx'], tomato_scarf_trans_z['qy'], tomato_scarf_trans_z['qz'])
tomato_scarf_trans_z_alpha_cleaned = cleanEuler(tomato_scarf_trans_z_alpha,0)
tomato_scarf_trans_z_beta_cleaned = cleanEuler(tomato_scarf_trans_z_beta,1)
tomato_scarf_trans_z_gamma_cleaned = cleanEuler(tomato_scarf_trans_z_gamma,2)

tomato_scarf_roll_alpha,tomato_scarf_roll_beta,tomato_scarf_roll_gamma = quaternion_to_euler_angle(tomato_scarf_roll['qw'], tomato_scarf_roll['qx'], tomato_scarf_roll['qy'], tomato_scarf_roll['qz'])
tomato_scarf_roll_alpha_cleaned = cleanEuler(tomato_scarf_roll_alpha,0)
tomato_scarf_roll_beta_cleaned = cleanEuler(tomato_scarf_roll_beta,1)
tomato_scarf_roll_gamma_cleaned = cleanEuler(tomato_scarf_roll_gamma,2)

tomato_scarf_pitch_alpha,tomato_scarf_pitch_beta,tomato_scarf_pitch_gamma = quaternion_to_euler_angle(tomato_scarf_pitch['qw'], tomato_scarf_pitch['qx'], tomato_scarf_pitch['qy'], tomato_scarf_pitch['qz'])
tomato_scarf_pitch_alpha_cleaned = cleanEuler(tomato_scarf_pitch_alpha,0)
tomato_scarf_pitch_beta_cleaned = cleanEuler(tomato_scarf_pitch_beta,1)
tomato_scarf_pitch_gamma_cleaned = cleanEuler(tomato_scarf_pitch_gamma,2)

tomato_scarf_yaw_alpha,tomato_scarf_yaw_beta,tomato_scarf_yaw_gamma = quaternion_to_euler_angle(tomato_scarf_yaw['qw'], tomato_scarf_yaw['qx'], tomato_scarf_yaw['qy'], tomato_scarf_yaw['qz'])
tomato_scarf_yaw_alpha_cleaned = cleanEuler(tomato_scarf_yaw_alpha,0)
tomato_scarf_yaw_beta_cleaned = cleanEuler(tomato_scarf_yaw_beta,1)
tomato_scarf_yaw_gamma_cleaned = cleanEuler(tomato_scarf_yaw_gamma,2)

tomato_scarf_error_trans_x = computeEuclideanDistance(tomato_gt_trans_x['x'], tomato_gt_trans_x['y'], tomato_gt_trans_x['z'], tomato_scarf_trans_x_resampled_x, tomato_scarf_trans_x_resampled_y, tomato_scarf_trans_x_resampled_z)
tomato_scarf_error_trans_y = computeEuclideanDistance(tomato_gt_trans_y['x'], tomato_gt_trans_y['y'], tomato_gt_trans_y['z'], tomato_scarf_trans_y_resampled_x, tomato_scarf_trans_y_resampled_y, tomato_scarf_trans_y_resampled_z)
tomato_scarf_error_trans_z = computeEuclideanDistance(tomato_gt_trans_z['x'], tomato_gt_trans_z['y'], tomato_gt_trans_z['z'], tomato_scarf_trans_z_resampled_x, tomato_scarf_trans_z_resampled_y, tomato_scarf_trans_z_resampled_z)
tomato_scarf_error_roll = computeEuclideanDistance(tomato_gt_roll['x'], tomato_gt_roll['y'], tomato_gt_roll['z'], tomato_scarf_roll_resampled_x, tomato_scarf_roll_resampled_y, tomato_scarf_roll_resampled_z)
tomato_scarf_error_pitch = computeEuclideanDistance(tomato_gt_pitch['x'], tomato_gt_pitch['y'], tomato_gt_pitch['z'], tomato_scarf_pitch_resampled_x, tomato_scarf_pitch_resampled_y, tomato_scarf_pitch_resampled_z)
tomato_scarf_error_yaw = computeEuclideanDistance(tomato_gt_yaw['x'], tomato_gt_yaw['y'], tomato_gt_yaw['z'], tomato_scarf_yaw_resampled_x, tomato_scarf_yaw_resampled_y, tomato_scarf_yaw_resampled_z)

tomato_scarf_q_angle_trans_x = computeQuaternionError(tomato_scarf_trans_x_resampled_qx, tomato_scarf_trans_x_resampled_qy, tomato_scarf_trans_x_resampled_qz, tomato_scarf_trans_x_resampled_qw, tomato_gt_trans_x['qx'], tomato_gt_trans_x['qy'], tomato_gt_trans_x['qz'], tomato_gt_trans_x['qw'])
tomato_scarf_q_angle_trans_y = computeQuaternionError(tomato_scarf_trans_y_resampled_qx, tomato_scarf_trans_y_resampled_qy, tomato_scarf_trans_y_resampled_qz, tomato_scarf_trans_y_resampled_qw, tomato_gt_trans_y['qx'], tomato_gt_trans_y['qy'], tomato_gt_trans_y['qz'], tomato_gt_trans_y['qw'])
tomato_scarf_q_angle_trans_z = computeQuaternionError(tomato_scarf_trans_z_resampled_qx, tomato_scarf_trans_z_resampled_qy, tomato_scarf_trans_z_resampled_qz, tomato_scarf_trans_z_resampled_qw, tomato_gt_trans_z['qx'], tomato_gt_trans_z['qy'], tomato_gt_trans_z['qz'], tomato_gt_trans_z['qw'])
tomato_scarf_q_angle_roll = computeQuaternionError(tomato_scarf_roll_resampled_qx, tomato_scarf_roll_resampled_qy, tomato_scarf_roll_resampled_qz, tomato_scarf_roll_resampled_qw, tomato_gt_roll['qx'], tomato_gt_roll['qy'], tomato_gt_roll['qz'], tomato_gt_roll['qw'])
tomato_scarf_q_angle_pitch = computeQuaternionError(tomato_scarf_pitch_resampled_qx, tomato_scarf_pitch_resampled_qy, tomato_scarf_pitch_resampled_qz, tomato_scarf_pitch_resampled_qw, tomato_gt_pitch['qx'], tomato_gt_pitch['qy'], tomato_gt_pitch['qz'], tomato_gt_pitch['qw'])
tomato_scarf_q_angle_yaw = computeQuaternionError(tomato_scarf_yaw_resampled_qx, tomato_scarf_yaw_resampled_qy, tomato_scarf_yaw_resampled_qz, tomato_scarf_yaw_resampled_qw, tomato_gt_yaw['qx'], tomato_gt_yaw['qy'], tomato_gt_yaw['qz'], tomato_gt_yaw['qw'])

tomato_scarf_position_errors = np.concatenate((tomato_scarf_error_trans_x, tomato_scarf_error_trans_y, tomato_scarf_error_trans_z, tomato_scarf_error_roll, tomato_scarf_error_pitch, tomato_scarf_error_yaw))
tomato_scarf_rotation_errors = np.concatenate((tomato_scarf_q_angle_trans_x, tomato_scarf_q_angle_trans_y, tomato_scarf_q_angle_trans_z, tomato_scarf_q_angle_roll, tomato_scarf_q_angle_pitch, tomato_scarf_q_angle_yaw))


# ---------------------------------------------------------------------------  potted  ---------------------------------------------------------------------------------------

filePath_dataset = '/data/scarf-journal/potted/'
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

potted_gt_trans_x_alpha,potted_gt_trans_x_beta,potted_gt_trans_x_gamma = quaternion_to_euler_angle(potted_gt_trans_x['qw'], potted_gt_trans_x['qx'], potted_gt_trans_x['qy'], potted_gt_trans_x['qz'])
potted_gt_trans_x_alpha_cleaned = cleanEuler(potted_gt_trans_x_alpha,0)
potted_gt_trans_x_beta_cleaned = cleanEuler(potted_gt_trans_x_beta,1)
potted_gt_trans_x_gamma_cleaned = cleanEuler(potted_gt_trans_x_gamma,1)

potted_gt_trans_y_alpha,potted_gt_trans_y_beta,potted_gt_trans_y_gamma = quaternion_to_euler_angle(potted_gt_trans_y['qw'], potted_gt_trans_y['qx'], potted_gt_trans_y['qy'], potted_gt_trans_y['qz'])
potted_gt_trans_y_alpha_cleaned = cleanEuler(potted_gt_trans_y_alpha,0)
potted_gt_trans_y_beta_cleaned = cleanEuler(potted_gt_trans_y_beta,1)
potted_gt_trans_y_gamma_cleaned = cleanEuler(potted_gt_trans_y_gamma,2)

potted_gt_trans_z_alpha,potted_gt_trans_z_beta,potted_gt_trans_z_gamma = quaternion_to_euler_angle(potted_gt_trans_z['qw'], potted_gt_trans_z['qx'], potted_gt_trans_z['qy'], potted_gt_trans_z['qz'])
potted_gt_trans_z_alpha_cleaned = cleanEuler(potted_gt_trans_z_alpha,0)
potted_gt_trans_z_beta_cleaned = cleanEuler(potted_gt_trans_z_beta,1)
potted_gt_trans_z_gamma_cleaned = cleanEuler(potted_gt_trans_z_gamma,2)

potted_gt_roll_alpha,potted_gt_roll_beta,potted_gt_roll_gamma = quaternion_to_euler_angle(potted_gt_roll['qw'], potted_gt_roll['qx'], potted_gt_roll['qy'], potted_gt_roll['qz'])
potted_gt_roll_alpha_cleaned = cleanEuler(potted_gt_roll_alpha,0)
potted_gt_roll_beta_cleaned = cleanEuler(potted_gt_roll_beta,1)
potted_gt_roll_gamma_cleaned = cleanEuler(potted_gt_roll_gamma,2)

potted_gt_pitch_alpha,potted_gt_pitch_beta,potted_gt_pitch_gamma = quaternion_to_euler_angle(potted_gt_pitch['qw'], potted_gt_pitch['qx'], potted_gt_pitch['qy'], potted_gt_pitch['qz'])
potted_gt_pitch_alpha_cleaned = cleanEuler(potted_gt_pitch_alpha,0)
potted_gt_pitch_beta_cleaned = cleanEuler(potted_gt_pitch_beta,1)
potted_gt_pitch_gamma_cleaned = cleanEuler(potted_gt_pitch_gamma,2)

potted_gt_yaw_alpha,potted_gt_yaw_beta,potted_gt_yaw_gamma = quaternion_to_euler_angle(potted_gt_yaw['qw'], potted_gt_yaw['qx'], potted_gt_yaw['qy'], potted_gt_yaw['qz'])
potted_gt_yaw_alpha_cleaned = cleanEuler(potted_gt_yaw_alpha,0)
potted_gt_yaw_beta_cleaned = cleanEuler(potted_gt_yaw_beta,1)
potted_gt_yaw_gamma_cleaned = cleanEuler(potted_gt_yaw_gamma,2)

potted_eros_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_eros_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_eros_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_eros_roll = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_eros_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_eros_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'eros-icra/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

potted_eros_trans_x_resampled_x = resampling_by_interpolate(potted_gt_trans_x['t'], potted_eros_trans_x['t'], potted_eros_trans_x['x'])
potted_eros_trans_x_resampled_y = resampling_by_interpolate(potted_gt_trans_x['t'], potted_eros_trans_x['t'], potted_eros_trans_x['y'])
potted_eros_trans_x_resampled_z = resampling_by_interpolate(potted_gt_trans_x['t'], potted_eros_trans_x['t'], potted_eros_trans_x['z'])
potted_eros_trans_x_resampled_qx = resampling_by_interpolate(potted_gt_trans_x['t'], potted_eros_trans_x['t'], potted_eros_trans_x['qx'])
potted_eros_trans_x_resampled_qy = resampling_by_interpolate(potted_gt_trans_x['t'], potted_eros_trans_x['t'], potted_eros_trans_x['qy'])
potted_eros_trans_x_resampled_qz = resampling_by_interpolate(potted_gt_trans_x['t'], potted_eros_trans_x['t'], potted_eros_trans_x['qz'])
potted_eros_trans_x_resampled_qw = resampling_by_interpolate(potted_gt_trans_x['t'], potted_eros_trans_x['t'], potted_eros_trans_x['qw'])

potted_eros_trans_y_resampled_x = resampling_by_interpolate(potted_gt_trans_y['t'], potted_eros_trans_y['t'], potted_eros_trans_y['x'])
potted_eros_trans_y_resampled_y = resampling_by_interpolate(potted_gt_trans_y['t'], potted_eros_trans_y['t'], potted_eros_trans_y['y'])
potted_eros_trans_y_resampled_z = resampling_by_interpolate(potted_gt_trans_y['t'], potted_eros_trans_y['t'], potted_eros_trans_y['z'])
potted_eros_trans_y_resampled_qx = resampling_by_interpolate(potted_gt_trans_y['t'], potted_eros_trans_y['t'], potted_eros_trans_y['qx'])
potted_eros_trans_y_resampled_qy = resampling_by_interpolate(potted_gt_trans_y['t'], potted_eros_trans_y['t'], potted_eros_trans_y['qy'])
potted_eros_trans_y_resampled_qz = resampling_by_interpolate(potted_gt_trans_y['t'], potted_eros_trans_y['t'], potted_eros_trans_y['qz'])
potted_eros_trans_y_resampled_qw = resampling_by_interpolate(potted_gt_trans_y['t'], potted_eros_trans_y['t'], potted_eros_trans_y['qw'])

potted_eros_trans_z_resampled_x = resampling_by_interpolate(potted_gt_trans_z['t'], potted_eros_trans_z['t'], potted_eros_trans_z['x'])
potted_eros_trans_z_resampled_y = resampling_by_interpolate(potted_gt_trans_z['t'], potted_eros_trans_z['t'], potted_eros_trans_z['y'])
potted_eros_trans_z_resampled_z = resampling_by_interpolate(potted_gt_trans_z['t'], potted_eros_trans_z['t'], potted_eros_trans_z['z'])
potted_eros_trans_z_resampled_qx = resampling_by_interpolate(potted_gt_trans_z['t'], potted_eros_trans_z['t'], potted_eros_trans_z['qx'])
potted_eros_trans_z_resampled_qy = resampling_by_interpolate(potted_gt_trans_z['t'], potted_eros_trans_z['t'], potted_eros_trans_z['qy'])
potted_eros_trans_z_resampled_qz = resampling_by_interpolate(potted_gt_trans_z['t'], potted_eros_trans_z['t'], potted_eros_trans_z['qz'])
potted_eros_trans_z_resampled_qw = resampling_by_interpolate(potted_gt_trans_z['t'], potted_eros_trans_z['t'], potted_eros_trans_z['qw'])

potted_eros_roll_resampled_x = resampling_by_interpolate(potted_gt_roll['t'], potted_eros_roll['t'], potted_eros_roll['x'])
potted_eros_roll_resampled_y = resampling_by_interpolate(potted_gt_roll['t'], potted_eros_roll['t'], potted_eros_roll['y'])
potted_eros_roll_resampled_z = resampling_by_interpolate(potted_gt_roll['t'], potted_eros_roll['t'], potted_eros_roll['z'])
potted_eros_roll_resampled_qx = resampling_by_interpolate(potted_gt_roll['t'], potted_eros_roll['t'], potted_eros_roll['qx'])
potted_eros_roll_resampled_qy = resampling_by_interpolate(potted_gt_roll['t'], potted_eros_roll['t'], potted_eros_roll['qy'])
potted_eros_roll_resampled_qz = resampling_by_interpolate(potted_gt_roll['t'], potted_eros_roll['t'], potted_eros_roll['qz'])
potted_eros_roll_resampled_qw = resampling_by_interpolate(potted_gt_roll['t'], potted_eros_roll['t'], potted_eros_roll['qw'])

potted_eros_pitch_resampled_x = resampling_by_interpolate(potted_gt_pitch['t'], potted_eros_pitch['t'], potted_eros_pitch['x'])
potted_eros_pitch_resampled_y = resampling_by_interpolate(potted_gt_pitch['t'], potted_eros_pitch['t'], potted_eros_pitch['y'])
potted_eros_pitch_resampled_z = resampling_by_interpolate(potted_gt_pitch['t'], potted_eros_pitch['t'], potted_eros_pitch['z'])
potted_eros_pitch_resampled_qx = resampling_by_interpolate(potted_gt_pitch['t'], potted_eros_pitch['t'], potted_eros_pitch['qx'])
potted_eros_pitch_resampled_qy = resampling_by_interpolate(potted_gt_pitch['t'], potted_eros_pitch['t'], potted_eros_pitch['qy'])
potted_eros_pitch_resampled_qz = resampling_by_interpolate(potted_gt_pitch['t'], potted_eros_pitch['t'], potted_eros_pitch['qz'])
potted_eros_pitch_resampled_qw = resampling_by_interpolate(potted_gt_pitch['t'], potted_eros_pitch['t'], potted_eros_pitch['qw'])

potted_eros_yaw_resampled_x = resampling_by_interpolate(potted_gt_yaw['t'], potted_eros_yaw['t'], potted_eros_yaw['x'])
potted_eros_yaw_resampled_y = resampling_by_interpolate(potted_gt_yaw['t'], potted_eros_yaw['t'], potted_eros_yaw['y'])
potted_eros_yaw_resampled_z = resampling_by_interpolate(potted_gt_yaw['t'], potted_eros_yaw['t'], potted_eros_yaw['z'])
potted_eros_yaw_resampled_qx = resampling_by_interpolate(potted_gt_yaw['t'], potted_eros_yaw['t'], potted_eros_yaw['qx'])
potted_eros_yaw_resampled_qy = resampling_by_interpolate(potted_gt_yaw['t'], potted_eros_yaw['t'], potted_eros_yaw['qy'])
potted_eros_yaw_resampled_qz = resampling_by_interpolate(potted_gt_yaw['t'], potted_eros_yaw['t'], potted_eros_yaw['qz'])
potted_eros_yaw_resampled_qw = resampling_by_interpolate(potted_gt_yaw['t'], potted_eros_yaw['t'], potted_eros_yaw['qw'])

potted_eros_trans_x_alpha,potted_eros_trans_x_beta,potted_eros_trans_x_gamma = quaternion_to_euler_angle(potted_eros_trans_x['qw'], potted_eros_trans_x['qx'], potted_eros_trans_x['qy'], potted_eros_trans_x['qz'])
potted_eros_trans_x_alpha_cleaned = cleanEuler(potted_eros_trans_x_alpha,0)
potted_eros_trans_x_beta_cleaned = cleanEuler(potted_eros_trans_x_beta,1)
potted_eros_trans_x_gamma_cleaned = cleanEuler(potted_eros_trans_x_gamma,2)

potted_eros_trans_y_alpha,potted_eros_trans_y_beta,potted_eros_trans_y_gamma = quaternion_to_euler_angle(potted_eros_trans_y['qw'], potted_eros_trans_y['qx'], potted_eros_trans_y['qy'], potted_eros_trans_y['qz'])
potted_eros_trans_y_alpha_cleaned = cleanEuler(potted_eros_trans_y_alpha,0)
potted_eros_trans_y_beta_cleaned = cleanEuler(potted_eros_trans_y_beta,1)
potted_eros_trans_y_gamma_cleaned = cleanEuler(potted_eros_trans_y_gamma,2)

potted_eros_trans_z_alpha,potted_eros_trans_z_beta,potted_eros_trans_z_gamma = quaternion_to_euler_angle(potted_eros_trans_z['qw'], potted_eros_trans_z['qx'], potted_eros_trans_z['qy'], potted_eros_trans_z['qz'])
potted_eros_trans_z_alpha_cleaned = cleanEuler(potted_eros_trans_z_alpha,0)
potted_eros_trans_z_beta_cleaned = cleanEuler(potted_eros_trans_z_beta,1)
potted_eros_trans_z_gamma_cleaned = cleanEuler(potted_eros_trans_z_gamma,2)

potted_eros_roll_alpha,potted_eros_roll_beta,potted_eros_roll_gamma = quaternion_to_euler_angle(potted_eros_roll['qw'], potted_eros_roll['qx'], potted_eros_roll['qy'], potted_eros_roll['qz'])
potted_eros_roll_alpha_cleaned = cleanEuler(potted_eros_roll_alpha,0)
potted_eros_roll_beta_cleaned = cleanEuler(potted_eros_roll_beta,1)
potted_eros_roll_gamma_cleaned = cleanEuler(potted_eros_roll_gamma,2)

potted_eros_pitch_alpha,potted_eros_pitch_beta,potted_eros_pitch_gamma = quaternion_to_euler_angle(potted_eros_pitch['qw'], potted_eros_pitch['qx'], potted_eros_pitch['qy'], potted_eros_pitch['qz'])
potted_eros_pitch_alpha_cleaned = cleanEuler(potted_eros_pitch_alpha,0)
potted_eros_pitch_beta_cleaned = cleanEuler(potted_eros_pitch_beta,1)
potted_eros_pitch_gamma_cleaned = cleanEuler(potted_eros_pitch_gamma,2)

potted_eros_yaw_alpha,potted_eros_yaw_beta,potted_eros_yaw_gamma = quaternion_to_euler_angle(potted_eros_yaw['qw'], potted_eros_yaw['qx'], potted_eros_yaw['qy'], potted_eros_yaw['qz'])
potted_eros_yaw_alpha_cleaned = cleanEuler(potted_eros_yaw_alpha,0)
potted_eros_yaw_beta_cleaned = cleanEuler(potted_eros_yaw_beta,1)
potted_eros_yaw_gamma_cleaned = cleanEuler(potted_eros_yaw_gamma,2)

potted_eros_error_trans_x = computeEuclideanDistance(potted_gt_trans_x['x'], potted_gt_trans_x['y'], potted_gt_trans_x['z'], potted_eros_trans_x_resampled_x, potted_eros_trans_x_resampled_y, potted_eros_trans_x_resampled_z)
potted_eros_error_trans_y = computeEuclideanDistance(potted_gt_trans_y['x'], potted_gt_trans_y['y'], potted_gt_trans_y['z'], potted_eros_trans_y_resampled_x, potted_eros_trans_y_resampled_y, potted_eros_trans_y_resampled_z)
potted_eros_error_trans_z = computeEuclideanDistance(potted_gt_trans_z['x'], potted_gt_trans_z['y'], potted_gt_trans_z['z'], potted_eros_trans_z_resampled_x, potted_eros_trans_z_resampled_y, potted_eros_trans_z_resampled_z)
potted_eros_error_roll = computeEuclideanDistance(potted_gt_roll['x'], potted_gt_roll['y'], potted_gt_roll['z'], potted_eros_roll_resampled_x, potted_eros_roll_resampled_y, potted_eros_roll_resampled_z)
potted_eros_error_pitch = computeEuclideanDistance(potted_gt_pitch['x'], potted_gt_pitch['y'], potted_gt_pitch['z'], potted_eros_pitch_resampled_x, potted_eros_pitch_resampled_y, potted_eros_pitch_resampled_z)
potted_eros_error_yaw = computeEuclideanDistance(potted_gt_yaw['x'], potted_gt_yaw['y'], potted_gt_yaw['z'], potted_eros_yaw_resampled_x, potted_eros_yaw_resampled_y, potted_eros_yaw_resampled_z)

potted_eros_q_angle_trans_x = computeQuaternionError(potted_eros_trans_x_resampled_qx, potted_eros_trans_x_resampled_qy, potted_eros_trans_x_resampled_qz, potted_eros_trans_x_resampled_qw, potted_gt_trans_x['qx'], potted_gt_trans_x['qy'], potted_gt_trans_x['qz'], potted_gt_trans_x['qw'])
potted_eros_q_angle_trans_y = computeQuaternionError(potted_eros_trans_y_resampled_qx, potted_eros_trans_y_resampled_qy, potted_eros_trans_y_resampled_qz, potted_eros_trans_y_resampled_qw, potted_gt_trans_y['qx'], potted_gt_trans_y['qy'], potted_gt_trans_y['qz'], potted_gt_trans_y['qw'])
potted_eros_q_angle_trans_z = computeQuaternionError(potted_eros_trans_z_resampled_qx, potted_eros_trans_z_resampled_qy, potted_eros_trans_z_resampled_qz, potted_eros_trans_z_resampled_qw, potted_gt_trans_z['qx'], potted_gt_trans_z['qy'], potted_gt_trans_z['qz'], potted_gt_trans_z['qw'])
potted_eros_q_angle_roll = computeQuaternionError(potted_eros_roll_resampled_qx, potted_eros_roll_resampled_qy, potted_eros_roll_resampled_qz, potted_eros_roll_resampled_qw, potted_gt_roll['qx'], potted_gt_roll['qy'], potted_gt_roll['qz'], potted_gt_roll['qw'])
potted_eros_q_angle_pitch = computeQuaternionError(potted_eros_pitch_resampled_qx, potted_eros_pitch_resampled_qy, potted_eros_pitch_resampled_qz, potted_eros_pitch_resampled_qw, potted_gt_pitch['qx'], potted_gt_pitch['qy'], potted_gt_pitch['qz'], potted_gt_pitch['qw'])
potted_eros_q_angle_yaw = computeQuaternionError(potted_eros_yaw_resampled_qx, potted_eros_yaw_resampled_qy, potted_eros_yaw_resampled_qz, potted_eros_yaw_resampled_qw, potted_gt_yaw['qx'], potted_gt_yaw['qy'], potted_gt_yaw['qz'], potted_gt_yaw['qw'])

potted_eros_position_errors = np.concatenate((potted_eros_error_trans_x, potted_eros_error_trans_y, potted_eros_error_trans_z, potted_eros_error_roll, potted_eros_error_pitch, potted_eros_error_yaw))
potted_eros_rotation_errors = np.concatenate((potted_eros_q_angle_trans_x, potted_eros_q_angle_trans_y, potted_eros_q_angle_trans_z, potted_eros_q_angle_roll, potted_eros_q_angle_pitch, potted_eros_q_angle_yaw))

potted_scarf_trans_x = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/x.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_scarf_trans_y = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/y.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_scarf_trans_z = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/z.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_scarf_roll = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/roll.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_scarf_pitch = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/pitch.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])
potted_scarf_yaw = np.genfromtxt(os.path.join(filePath_dataset, 'scarf/yaw.csv'), delimiter=",", names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"])

potted_scarf_trans_x_resampled_x = resampling_by_interpolate(potted_gt_trans_x['t'], potted_scarf_trans_x['t'], potted_scarf_trans_x['x'])
potted_scarf_trans_x_resampled_y = resampling_by_interpolate(potted_gt_trans_x['t'], potted_scarf_trans_x['t'], potted_scarf_trans_x['y'])
potted_scarf_trans_x_resampled_z = resampling_by_interpolate(potted_gt_trans_x['t'], potted_scarf_trans_x['t'], potted_scarf_trans_x['z'])
potted_scarf_trans_x_resampled_qx = resampling_by_interpolate(potted_gt_trans_x['t'], potted_scarf_trans_x['t'], potted_scarf_trans_x['qx'])
potted_scarf_trans_x_resampled_qy = resampling_by_interpolate(potted_gt_trans_x['t'], potted_scarf_trans_x['t'], potted_scarf_trans_x['qy'])
potted_scarf_trans_x_resampled_qz = resampling_by_interpolate(potted_gt_trans_x['t'], potted_scarf_trans_x['t'], potted_scarf_trans_x['qz'])
potted_scarf_trans_x_resampled_qw = resampling_by_interpolate(potted_gt_trans_x['t'], potted_scarf_trans_x['t'], potted_scarf_trans_x['qw'])

potted_scarf_trans_y_resampled_x = resampling_by_interpolate(potted_gt_trans_y['t'], potted_scarf_trans_y['t'], potted_scarf_trans_y['x'])
potted_scarf_trans_y_resampled_y = resampling_by_interpolate(potted_gt_trans_y['t'], potted_scarf_trans_y['t'], potted_scarf_trans_y['y'])
potted_scarf_trans_y_resampled_z = resampling_by_interpolate(potted_gt_trans_y['t'], potted_scarf_trans_y['t'], potted_scarf_trans_y['z'])
potted_scarf_trans_y_resampled_qx = resampling_by_interpolate(potted_gt_trans_y['t'], potted_scarf_trans_y['t'], potted_scarf_trans_y['qx'])
potted_scarf_trans_y_resampled_qy = resampling_by_interpolate(potted_gt_trans_y['t'], potted_scarf_trans_y['t'], potted_scarf_trans_y['qy'])
potted_scarf_trans_y_resampled_qz = resampling_by_interpolate(potted_gt_trans_y['t'], potted_scarf_trans_y['t'], potted_scarf_trans_y['qz'])
potted_scarf_trans_y_resampled_qw = resampling_by_interpolate(potted_gt_trans_y['t'], potted_scarf_trans_y['t'], potted_scarf_trans_y['qw'])

potted_scarf_trans_z_resampled_x = resampling_by_interpolate(potted_gt_trans_z['t'], potted_scarf_trans_z['t'], potted_scarf_trans_z['x'])
potted_scarf_trans_z_resampled_y = resampling_by_interpolate(potted_gt_trans_z['t'], potted_scarf_trans_z['t'], potted_scarf_trans_z['y'])
potted_scarf_trans_z_resampled_z = resampling_by_interpolate(potted_gt_trans_z['t'], potted_scarf_trans_z['t'], potted_scarf_trans_z['z'])
potted_scarf_trans_z_resampled_qx = resampling_by_interpolate(potted_gt_trans_z['t'], potted_scarf_trans_z['t'], potted_scarf_trans_z['qx'])
potted_scarf_trans_z_resampled_qy = resampling_by_interpolate(potted_gt_trans_z['t'], potted_scarf_trans_z['t'], potted_scarf_trans_z['qy'])
potted_scarf_trans_z_resampled_qz = resampling_by_interpolate(potted_gt_trans_z['t'], potted_scarf_trans_z['t'], potted_scarf_trans_z['qz'])
potted_scarf_trans_z_resampled_qw = resampling_by_interpolate(potted_gt_trans_z['t'], potted_scarf_trans_z['t'], potted_scarf_trans_z['qw'])

potted_scarf_roll_resampled_x = resampling_by_interpolate(potted_gt_roll['t'], potted_scarf_roll['t'], potted_scarf_roll['x'])
potted_scarf_roll_resampled_y = resampling_by_interpolate(potted_gt_roll['t'], potted_scarf_roll['t'], potted_scarf_roll['y'])
potted_scarf_roll_resampled_z = resampling_by_interpolate(potted_gt_roll['t'], potted_scarf_roll['t'], potted_scarf_roll['z'])
potted_scarf_roll_resampled_qx = resampling_by_interpolate(potted_gt_roll['t'], potted_scarf_roll['t'], potted_scarf_roll['qx'])
potted_scarf_roll_resampled_qy = resampling_by_interpolate(potted_gt_roll['t'], potted_scarf_roll['t'], potted_scarf_roll['qy'])
potted_scarf_roll_resampled_qz = resampling_by_interpolate(potted_gt_roll['t'], potted_scarf_roll['t'], potted_scarf_roll['qz'])
potted_scarf_roll_resampled_qw = resampling_by_interpolate(potted_gt_roll['t'], potted_scarf_roll['t'], potted_scarf_roll['qw'])

potted_scarf_pitch_resampled_x = resampling_by_interpolate(potted_gt_pitch['t'], potted_scarf_pitch['t'], potted_scarf_pitch['x'])
potted_scarf_pitch_resampled_y = resampling_by_interpolate(potted_gt_pitch['t'], potted_scarf_pitch['t'], potted_scarf_pitch['y'])
potted_scarf_pitch_resampled_z = resampling_by_interpolate(potted_gt_pitch['t'], potted_scarf_pitch['t'], potted_scarf_pitch['z'])
potted_scarf_pitch_resampled_qx = resampling_by_interpolate(potted_gt_pitch['t'], potted_scarf_pitch['t'], potted_scarf_pitch['qx'])
potted_scarf_pitch_resampled_qy = resampling_by_interpolate(potted_gt_pitch['t'], potted_scarf_pitch['t'], potted_scarf_pitch['qy'])
potted_scarf_pitch_resampled_qz = resampling_by_interpolate(potted_gt_pitch['t'], potted_scarf_pitch['t'], potted_scarf_pitch['qz'])
potted_scarf_pitch_resampled_qw = resampling_by_interpolate(potted_gt_pitch['t'], potted_scarf_pitch['t'], potted_scarf_pitch['qw'])

potted_scarf_yaw_resampled_x = resampling_by_interpolate(potted_gt_yaw['t'], potted_scarf_yaw['t'], potted_scarf_yaw['x'])
potted_scarf_yaw_resampled_y = resampling_by_interpolate(potted_gt_yaw['t'], potted_scarf_yaw['t'], potted_scarf_yaw['y'])
potted_scarf_yaw_resampled_z = resampling_by_interpolate(potted_gt_yaw['t'], potted_scarf_yaw['t'], potted_scarf_yaw['z'])
potted_scarf_yaw_resampled_qx = resampling_by_interpolate(potted_gt_yaw['t'], potted_scarf_yaw['t'], potted_scarf_yaw['qx'])
potted_scarf_yaw_resampled_qy = resampling_by_interpolate(potted_gt_yaw['t'], potted_scarf_yaw['t'], potted_scarf_yaw['qy'])
potted_scarf_yaw_resampled_qz = resampling_by_interpolate(potted_gt_yaw['t'], potted_scarf_yaw['t'], potted_scarf_yaw['qz'])
potted_scarf_yaw_resampled_qw = resampling_by_interpolate(potted_gt_yaw['t'], potted_scarf_yaw['t'], potted_scarf_yaw['qw'])

potted_scarf_trans_x_alpha,potted_scarf_trans_x_beta,potted_scarf_trans_x_gamma = quaternion_to_euler_angle(potted_scarf_trans_x['qw'], potted_scarf_trans_x['qx'], potted_scarf_trans_x['qy'], potted_scarf_trans_x['qz'])
potted_scarf_trans_x_alpha_cleaned = cleanEuler(potted_scarf_trans_x_alpha,0)
potted_scarf_trans_x_beta_cleaned = cleanEuler(potted_scarf_trans_x_beta,1)
potted_scarf_trans_x_gamma_cleaned = cleanEuler(potted_scarf_trans_x_gamma,2)

potted_scarf_trans_y_alpha,potted_scarf_trans_y_beta,potted_scarf_trans_y_gamma = quaternion_to_euler_angle(potted_scarf_trans_y['qw'], potted_scarf_trans_y['qx'], potted_scarf_trans_y['qy'], potted_scarf_trans_y['qz'])
potted_scarf_trans_y_alpha_cleaned = cleanEuler(potted_scarf_trans_y_alpha,0)
potted_scarf_trans_y_beta_cleaned = cleanEuler(potted_scarf_trans_y_beta,1)
potted_scarf_trans_y_gamma_cleaned = cleanEuler(potted_scarf_trans_y_gamma,2)

potted_scarf_trans_z_alpha,potted_scarf_trans_z_beta,potted_scarf_trans_z_gamma = quaternion_to_euler_angle(potted_scarf_trans_z['qw'], potted_scarf_trans_z['qx'], potted_scarf_trans_z['qy'], potted_scarf_trans_z['qz'])
potted_scarf_trans_z_alpha_cleaned = cleanEuler(potted_scarf_trans_z_alpha,0)
potted_scarf_trans_z_beta_cleaned = cleanEuler(potted_scarf_trans_z_beta,1)
potted_scarf_trans_z_gamma_cleaned = cleanEuler(potted_scarf_trans_z_gamma,2)

potted_scarf_roll_alpha,potted_scarf_roll_beta,potted_scarf_roll_gamma = quaternion_to_euler_angle(potted_scarf_roll['qw'], potted_scarf_roll['qx'], potted_scarf_roll['qy'], potted_scarf_roll['qz'])
potted_scarf_roll_alpha_cleaned = cleanEuler(potted_scarf_roll_alpha,0)
potted_scarf_roll_beta_cleaned = cleanEuler(potted_scarf_roll_beta,1)
potted_scarf_roll_gamma_cleaned = cleanEuler(potted_scarf_roll_gamma,2)

potted_scarf_pitch_alpha,potted_scarf_pitch_beta,potted_scarf_pitch_gamma = quaternion_to_euler_angle(potted_scarf_pitch['qw'], potted_scarf_pitch['qx'], potted_scarf_pitch['qy'], potted_scarf_pitch['qz'])
potted_scarf_pitch_alpha_cleaned = cleanEuler(potted_scarf_pitch_alpha,0)
potted_scarf_pitch_beta_cleaned = cleanEuler(potted_scarf_pitch_beta,1)
potted_scarf_pitch_gamma_cleaned = cleanEuler(potted_scarf_pitch_gamma,2)

potted_scarf_yaw_alpha,potted_scarf_yaw_beta,potted_scarf_yaw_gamma = quaternion_to_euler_angle(potted_scarf_yaw['qw'], potted_scarf_yaw['qx'], potted_scarf_yaw['qy'], potted_scarf_yaw['qz'])
potted_scarf_yaw_alpha_cleaned = cleanEuler(potted_scarf_yaw_alpha,0)
potted_scarf_yaw_beta_cleaned = cleanEuler(potted_scarf_yaw_beta,1)
potted_scarf_yaw_gamma_cleaned = cleanEuler(potted_scarf_yaw_gamma,2)

potted_scarf_error_trans_x = computeEuclideanDistance(potted_gt_trans_x['x'], potted_gt_trans_x['y'], potted_gt_trans_x['z'], potted_scarf_trans_x_resampled_x, potted_scarf_trans_x_resampled_y, potted_scarf_trans_x_resampled_z)
potted_scarf_error_trans_y = computeEuclideanDistance(potted_gt_trans_y['x'], potted_gt_trans_y['y'], potted_gt_trans_y['z'], potted_scarf_trans_y_resampled_x, potted_scarf_trans_y_resampled_y, potted_scarf_trans_y_resampled_z)
potted_scarf_error_trans_z = computeEuclideanDistance(potted_gt_trans_z['x'], potted_gt_trans_z['y'], potted_gt_trans_z['z'], potted_scarf_trans_z_resampled_x, potted_scarf_trans_z_resampled_y, potted_scarf_trans_z_resampled_z)
potted_scarf_error_roll = computeEuclideanDistance(potted_gt_roll['x'], potted_gt_roll['y'], potted_gt_roll['z'], potted_scarf_roll_resampled_x, potted_scarf_roll_resampled_y, potted_scarf_roll_resampled_z)
potted_scarf_error_pitch = computeEuclideanDistance(potted_gt_pitch['x'], potted_gt_pitch['y'], potted_gt_pitch['z'], potted_scarf_pitch_resampled_x, potted_scarf_pitch_resampled_y, potted_scarf_pitch_resampled_z)
potted_scarf_error_yaw = computeEuclideanDistance(potted_gt_yaw['x'], potted_gt_yaw['y'], potted_gt_yaw['z'], potted_scarf_yaw_resampled_x, potted_scarf_yaw_resampled_y, potted_scarf_yaw_resampled_z)

potted_scarf_q_angle_trans_x = computeQuaternionError(potted_scarf_trans_x_resampled_qx, potted_scarf_trans_x_resampled_qy, potted_scarf_trans_x_resampled_qz, potted_scarf_trans_x_resampled_qw, potted_gt_trans_x['qx'], potted_gt_trans_x['qy'], potted_gt_trans_x['qz'], potted_gt_trans_x['qw'])
potted_scarf_q_angle_trans_y = computeQuaternionError(potted_scarf_trans_y_resampled_qx, potted_scarf_trans_y_resampled_qy, potted_scarf_trans_y_resampled_qz, potted_scarf_trans_y_resampled_qw, potted_gt_trans_y['qx'], potted_gt_trans_y['qy'], potted_gt_trans_y['qz'], potted_gt_trans_y['qw'])
potted_scarf_q_angle_trans_z = computeQuaternionError(potted_scarf_trans_z_resampled_qx, potted_scarf_trans_z_resampled_qy, potted_scarf_trans_z_resampled_qz, potted_scarf_trans_z_resampled_qw, potted_gt_trans_z['qx'], potted_gt_trans_z['qy'], potted_gt_trans_z['qz'], potted_gt_trans_z['qw'])
potted_scarf_q_angle_roll = computeQuaternionError(potted_scarf_roll_resampled_qx, potted_scarf_roll_resampled_qy, potted_scarf_roll_resampled_qz, potted_scarf_roll_resampled_qw, potted_gt_roll['qx'], potted_gt_roll['qy'], potted_gt_roll['qz'], potted_gt_roll['qw'])
potted_scarf_q_angle_pitch = computeQuaternionError(potted_scarf_pitch_resampled_qx, potted_scarf_pitch_resampled_qy, potted_scarf_pitch_resampled_qz, potted_scarf_pitch_resampled_qw, potted_gt_pitch['qx'], potted_gt_pitch['qy'], potted_gt_pitch['qz'], potted_gt_pitch['qw'])
potted_scarf_q_angle_yaw = computeQuaternionError(potted_scarf_yaw_resampled_qx, potted_scarf_yaw_resampled_qy, potted_scarf_yaw_resampled_qz, potted_scarf_yaw_resampled_qw, potted_gt_yaw['qx'], potted_gt_yaw['qy'], potted_gt_yaw['qz'], potted_gt_yaw['qw'])

potted_scarf_position_errors = np.concatenate((potted_scarf_error_trans_x, potted_scarf_error_trans_y, potted_scarf_error_trans_z, potted_scarf_error_roll, potted_scarf_error_pitch, potted_scarf_error_yaw))
potted_scarf_rotation_errors = np.concatenate((potted_scarf_q_angle_trans_x, potted_scarf_q_angle_trans_y, potted_scarf_q_angle_trans_z, potted_scarf_q_angle_roll, potted_scarf_q_angle_pitch, potted_scarf_q_angle_yaw))


labels = ['O5', 'O4', 'O3', 'O2', 'O1']
ticks = np.arange(len(labels))
medianprops = dict(color='k')
rad_to_deg = 180/math.pi

all_objects_position_errors_eros = [potted_eros_position_errors, tomato_eros_position_errors, mustard_eros_position_errors, gelatin_eros_position_errors, dragon_eros_position_errors]
all_objects_position_errors_scarf = [potted_scarf_position_errors, tomato_scarf_position_errors, mustard_scarf_position_errors, gelatin_scarf_position_errors, dragon_scarf_position_errors]

all_objects_rotation_errors_eros = [potted_eros_rotation_errors*rad_to_deg, tomato_eros_rotation_errors*rad_to_deg, mustard_eros_rotation_errors*rad_to_deg, gelatin_eros_rotation_errors*rad_to_deg, dragon_eros_rotation_errors*rad_to_deg]
all_objects_rotation_errors_scarf = [potted_scarf_rotation_errors*rad_to_deg, tomato_scarf_rotation_errors*rad_to_deg, mustard_scarf_rotation_errors*rad_to_deg, gelatin_scarf_rotation_errors*rad_to_deg, dragon_scarf_rotation_errors*rad_to_deg]

# Vertical positions for double bars
offset = 0.15
y_positions_eros = ticks - offset
y_positions_scarf = ticks + offset

fig15, ax1 = plt.subplots(1,2)
fig15.set_size_inches(8, 6)
ax1[0].set_xlabel('Position error [m]', color='k')
ax1[1].set_xlabel('Rotation error [deg]', color='k')
res1_eros = ax1[0].boxplot(all_objects_position_errors_eros, positions=y_positions_eros, vert=False,
                           widths=0.25, patch_artist=True, showfliers=False, medianprops=medianprops)
res2_eros = ax1[1].boxplot(all_objects_rotation_errors_eros, positions=y_positions_eros, vert=False,
                           widths=0.25, patch_artist=True, showfliers=False, medianprops=medianprops)

res1_scarf = ax1[0].boxplot(all_objects_position_errors_scarf, positions=y_positions_scarf, vert=False,
                           widths=0.25, patch_artist=True, showfliers=False, medianprops=medianprops)
res2_scarf = ax1[1].boxplot(all_objects_rotation_errors_scarf, positions=y_positions_scarf, vert=False,
                           widths=0.25, patch_artist=True, showfliers=False, medianprops=medianprops)

# Set colors
for patch in res1_eros['boxes']:
    patch.set_facecolor('tab:orange')
    patch.set_edgecolor('k')

for patch in res2_eros['boxes']:
    patch.set_facecolor('tab:orange')
    patch.set_edgecolor('k')

for patch in res1_scarf['boxes']:
    patch.set_facecolor('tab:green')
    patch.set_edgecolor('k')

for patch in res2_scarf['boxes']:
    patch.set_facecolor('tab:green')
    patch.set_edgecolor('k')

# Ticks and limits
ax1[0].set_yticks(ticks)
ax1[0].set_yticklabels(labels)
ax1[0].set_xlim(-0.001, 0.03)

ax1[1].set_yticks([])
ax1[1].set_xlim(-1, 22)
ax1[1].xaxis.set_major_locator(plt.MaxNLocator(4))

# Legend 
fig15.legend([res1_eros['boxes'][0], res1_scarf['boxes'][0]], ['eros', 'scarf'], loc='upper center', ncol=2)

fig15.subplots_adjust(wspace=0.1)
plt.savefig("scarf_vs_eros.png", dpi=300, bbox_inches='tight')