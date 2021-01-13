
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
fullpath = parentdir + '\\python'
sys.path.append(fullpath)

import numpy as np
import pickle
from utils import dataset_types 
import matplotlib.pyplot as plt


# containter for the data



def get_traj_filepath(scenario, file_number):
    return "../trajectory_files/" + str(scenario) + "/vehicle_tracks_" + str(file_number).zfill(3) + "_trajs.pickle"

def extract_dataset_traj(scenario, is_interaction, entrances=[],exits=[],data_lim=sys.maxsize):
    ''' 
        This function takes in a scenario name and a trajectory type (interaction
        vs no interaction) and returns x, y, xv, yv, psi values from cars from the 
        specified scenario that fit the trajectory type

        Specifying data_lim will take only the first <data_lim> cars from 
        <scenario> that have the specified trajectory type
    '''
    car_id = []
    big_x_traj = []
    big_y_traj = []
    big_vx_traj = []
    big_vy_traj = []
    big_psi_traj = []
    traj_dict = {}

    # load first available traj file
    file_number = 0
    file_path = get_traj_filepath(scenario, file_number)
    
    while os.path.isfile(file_path): 
        # open trajectory file
        with open(file_path, 'rb') as handle:
            traj_dict = pickle.load(handle)
        
        # for all cars in current traj file 
        for car in traj_dict.values():
            # check if data exceeds data limits
            data_count = len(big_x_traj)
            if data_count >= data_lim:
                return(big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)
            # extract car data
            if is_interaction == car.interaction:
                if (not exits and not entrances) or (car.exit_id in exits and car.entrance_id in entrances):
                    #car_id.append([file_number, car.track_id])    # -- debug
                    big_x_traj.append(np.array(car.x_vals))
                    big_y_traj.append(np.array(car.y_vals))
                    big_vx_traj.append(np.array(car.vx_vals))
                    big_vy_traj.append(np.array(car.vy_vals))
                    big_psi_traj.append(np.array(car.psi_vals))

        # iterate to next file
        file_number += 1
        file_path = get_traj_filepath(scenario, file_number)

    # return all cars found 
    return(big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)
    


#x, y, vx, vy, psi  = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=10)

def generate_demo_traj():
    big_x_traj = []
    big_y_traj = []
    big_vx_traj = []
    big_vy_traj = []
    big_psi_traj = []
    psi = np.linspace(-np.pi / 4, np.pi / 4, 15)
    ts = 0.1
    v = 1.0
    L = 4
    # x = 0
    # y = 0
    # theta = 0
    # demonstaration of the data
    for demo_psi in range(len(psi)):
        x = 0
        y = 0
        theta = 0
        x_traj = []
        y_traj = []
        vx_traj = []
        vy_traj = []
        psi_traj = []

        for t in range(100):
            #print(psi[demo_psi])
            theta_dot = (v/L) * np.tan(psi[demo_psi])
            theta = theta + ts* theta_dot
            x_dot = v* np.cos(theta)
            y_dot = v* np.sin(theta)
            x = x + ts * x_dot
            y = y + ts* y_dot

            x_traj.append(x)
            y_traj.append(y)
            vx_traj.append(x_dot)
            vy_traj.append(y_dot)
            psi_traj.append(theta)

        big_x_traj.append(np.array(x_traj))
        big_y_traj.append(np.array(y_traj))
        big_vx_traj.append(np.array(vx_traj))
        big_vy_traj.append(np.array(vy_traj))
        big_psi_traj.append(np.array(psi_traj))

        #
        # plt.figure(1)
        # plt.plot(x_traj, y_traj)
        x = 0
        y = 0
        theta = 0
        x_traj = []
        y_traj = []
        vx_traj = []
        vy_traj = []
        psi_traj = []

        for t in range(100):
            #print(psi[demo_psi])
            theta_dot = -(v/L) * np.tan(psi[demo_psi])
            theta = theta + ts* theta_dot
            x_dot = -v* np.cos(theta)
            y_dot = - v* np.sin(theta)
            x = x + ts * x_dot
            y = y + ts* y_dot

            x_traj.append(x)
            y_traj.append(y)
            vx_traj.append(x_dot)
            vy_traj.append(y_dot)
            psi_traj.append(theta)

        # plt.plot(x_traj, y_traj)
        big_x_traj.append(np.array(x_traj))
        big_y_traj.append(np.array(y_traj))
        big_vx_traj.append(np.array(vx_traj))
        big_vy_traj.append(np.array(vy_traj))
        big_psi_traj.append(np.array(psi_traj))

    # #
    # plt.title('tester_plot')
    # plt.show()


    return(big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)

all_x, all_y, all_vx, all_vy, all_psi= generate_demo_traj()


