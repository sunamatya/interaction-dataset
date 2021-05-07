import numpy as np
import matplotlib.pyplot as plt
from example_tester import generate_demo_traj
from example_tester import extract_dataset_traj
import scipy.interpolate
import sys
import csv
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndimage
from GT_primitive_primary import state_seg_v2
from GT_primitive_primary import state_seg_cluster_gmm
from GT_primitive_primary import get_Ridge_Refression
#from GT_primitive_primary import generate_psi_dt
#from GT_primitive_primary import weight_gen

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
T = 20
K = 80
D = 5

def get_GausRBF_alpha(z_t,mean_k,sigma,D, alpha):
    return np.exp((-0.5*(alpha*z_t-mean_k)*(alpha* z_t-mean_k))*(1./sigma)) / (np.sqrt(np.power(2*np.pi,D) * sigma))

def compute_phi_index_obs(data, all_phi, all_alpha,mean_weight):
    #data is the observation (x,y,vx,vy,psi)
    #segment is original training data (x, y, vx, vy, psi)
    #can be used for the finding out the mean trajectory from the class or from the separate data
    obs_len = len(data)/D
    diff_sum=[]
    data = np.array(data)
    for i in range (len(all_alpha)):
        phi_data=np.dot(all_phi[i],mean_weight[i])
        #phi_data = np.dot(all_phi[i], mean_weight)
        if len(phi_data)< len(data):
            diff_data = data[0:len(phi_data)] - phi_data
        else:

            diff_data = data- phi_data[0: len(data)]

            #diff_sum[i] = sum(data - phi_data[0:D*obs_len])
        #diff_sum.append(sum(diff_data))
        diff_sum.append(np.linalg.norm(diff_data))
    #index = diff_sum.index(min(diff_sum))
    index = np.argmin(diff_sum)
    return all_phi[index], mean_weight[index]


def compute_class(data, all_phi, all_alpha,mean_weights):
    #data is the observation (x,y,vx,vy,psi)
    #segment is original training data (x, y, vx, vy, psi)
    #can be used for the finding out the mean trajectory from the class or from the separate data
    obs_len = len(data)/D
    diff_sum=[]
    for i in range (len(all_alpha)):
        phi_data=np.dot(all_phi[i],mean_weights[i])
        #print(data)
        #print(phi_data[0: len(data)])
        diff_data = data- phi_data[0: len(data)]
        #print(diff_data)
        diff_sum.append(sum(diff_data))
        #diff_sum[i] = sum(diff_data)

    index = diff_sum.index(min(diff_sum))
    return index


def generate_psi_dt_ts(T, alpha):
    dk = np.linspace(0,1,K)
    dt = np.linspace(0,1,T)
    # dt = np.arange(0,T)
    #primitive_mean = dt.mean() #1
    primitive_variance = 0.2
    #primitive_variance = 5
    #primitive_variance = np.linspace(0.2,10, 0.2)
    # list_psi = []
    #list_traj = []
    #list_ww = []


    # equation 12 and 13
    big_psi = np.zeros((D * T, D * K))
    for ii in range(T):  # number of sequence
        for kk in range(K):
            b_t_x = get_GausRBF_alpha(dt[ii], dk[kk], primitive_variance, D, alpha)
            # b_t_y = get_GausRBF(z_y[ii], primitive_mean, primitive_variance, D)
            # b_t_o = get_GausRBF(z_psi[ii], primitive_mean, primitive_variance, D)
            # b_t_vx = get_GausRBF(z_vx[ii], primitive_mean, primitive_variance, D)
            # b_t_vy = get_GausRBF(z_vy[ii], primitive_mean, primitive_variance, D)
            # b_t_vy = get_GausRBF(z_vy[ii], primitive_mean, primitive_variance, D)
            small_psi = np.identity(D)
            # np.fill_diagonal(small_psi, [b_t_x, b_t_y, b_t_o, b_t_vx, b_t_vy])
            np.fill_diagonal(small_psi, [b_t_x, b_t_x, b_t_x, b_t_x, b_t_x])
            # np.fill_diagonal(small_psi, [b_t_x])
            big_psi[ii * D:(ii + 1) * D, kk * D:(kk + 1) * D] = np.copy(small_psi)
    print("Big Psi shape: ", big_psi.shape)
    return big_psi, dt

def weight_gen_T_wo_int(seg_range, x, y, vx, vy, psi, big_psi, dt, T, plot = False, calc_accuracy= False, plot_states = True):
    list_ww = []
    list_RMSE = []
    for seg_idx in range(len(seg_range)):
        Trajectory = []
        segment = seg_range[seg_idx]
        data_r = segment["data_range"]
        t_start = data_r[0]
        t_end = data_r[1]
        t_data = t_end - t_start + 1
        seg_x = x[t_start:t_end+1]
        #print("seg_x shape: ",len(seg_x))
        seg_y = y[t_start:t_end+1]
        seg_vx = vx[t_start:t_end+1]
        seg_vy = vy[t_start:t_end+1]
        seg_psi = psi[t_start:t_end+1]
        for t in range(T[seg_idx]):
            Trajectory.append(seg_x[t])
            Trajectory.append(seg_y[t])
            Trajectory.append(seg_psi[t])
            Trajectory.append(seg_vx[t])
            Trajectory.append(seg_vy[t])
        Trajectory = np.array(Trajectory)
        alpha = 1e-13  # from promp papaer from france
        # alpha = 0
        # alpha = 1e-1
        # alpha = 0.1
        ww = get_Ridge_Refression(big_psi[seg_idx], Trajectory, alpha)
        # list_psi.append(np.copy(big_psi))
        # list_traj.append(np.copy(Trajectory))
        list_ww.append(np.copy(ww))

        # TESTER

        traj_x = Trajectory[0:T[seg_idx] * D:D]
        traj_y = Trajectory[1:T[seg_idx] * D:D]  # np.arange(traj_x.shape[0])#
        traj_psi = Trajectory[2:T[seg_idx] * D:D]
        traj_vx = Trajectory[3:T[seg_idx] * D:D]
        traj_vy = Trajectory[4:T[seg_idx] * D:D]

        new_traj = np.dot(big_psi[seg_idx], ww)
        traj_x_p = new_traj[0:T[seg_idx] * D:D]
        traj_y_p = new_traj[1:T[seg_idx]* D:D]
        traj_psi_p = new_traj[2:T[seg_idx] * D:D]
        traj_vx_p = new_traj[3:T[seg_idx] * D:D]
        traj_vy_p = new_traj[4:T[seg_idx] * D:D]

        if calc_accuracy:
            diff_x = traj_x - traj_x_p
            diff_y = traj_y - traj_y_p
            diff_psi = traj_psi - traj_psi_p
            diff_vx = traj_vx - traj_vx_p
            diff_vy = traj_vy - traj_vy_p

            RMSE_x = np.sqrt(sum(diff_x ** (2)) / T)
            RMSE_y = np.sqrt(sum(diff_y ** (2)) / T)
            RMSE_psi = np.sqrt(sum(diff_psi ** (2)) / T)
            RMSE_vx = np.sqrt(sum(diff_vx ** (2)) / T)
            RMSE_vy = np.sqrt(sum(diff_vy ** (2)) / T)
            print("Primitive variance", primitive_variance, " RMSE accuracy", RMSE_x, RMSE_y, RMSE_psi, RMSE_vx,
                  RMSE_vy)
            # list_RMSE.append([RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy])

        if plot:
            if plot_states:
                fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
                fig.suptitle('Segmented Trajectory and States')
                ax1.plot(traj_x, "--", color="#ff6a6a", label="Origin x", linewidth=2.0)
                ax2.plot(traj_y, "--", color="#ff6a6a", label="Origin y", linewidth=2.0)
                ax3.plot(traj_psi, "--", color="#ff6a6a", label="Origin psi", linewidth=2.0)
                ax4.plot(traj_vx, "--", color="#ff6a6a", label="Origin vx", linewidth=2.0)
                ax5.plot(traj_vy, "--", color="#ff6a6a", label="Origin vy", linewidth=2.0)

                ax1.plot(traj_x_p, ".", color="b", label="Primitive x", linewidth=2.0)
                ax2.plot(traj_y_p, ".", color="b", label="Primitive y", linewidth=2.0)
                ax3.plot(traj_psi_p, ".", color="b", label="Primitive psi", linewidth=2.0)
                ax4.plot(traj_vx_p, ".", color="b", label="Primitive vx", linewidth=2.0)
                ax5.plot(traj_vy_p, ".", color="b", label="Primitive vy", linewidth=2.0)
                ax1.legend()
                ax2.legend()
                ax3.legend()
                ax4.legend()
                ax5.legend()
                plt.show()


            else:
                fig = plt.figure()
                plt.plot(traj_x, traj_y, "--", color="#ff6a6a", label="Original", linewidth=2.0)
                # plt.show()
                # plt.holdon()

                # fig = plt.figure()

                plt.plot(traj_x_p, traj_y_p, ".", color="b", label="Primitive", linewidth=2.0)
                fig.suptitle('Trained trajectory and Primitive Trajectory')
                plt.xlabel('X')
                plt.ylabel('Y')

                plt.legend()
                plt.show()
    return list_ww


def weight_gen_T(seg_range,x,y,vx,vy,psi,big_psi, dt, T, plot = False, calc_accuracy = False, plot_states= True):
    #equation 11
    list_ww = []
    list_RMSE = []
    alphas = [1e-9, 1e-11, 1e-13]
    for seg_idx in range(len(seg_range)):
    #for seg_idx in range(1):
        Trajectory = []
        segment= seg_range[seg_idx]
        data_r = segment["data_range"]
        t_start = data_r[0]
        t_end = data_r[1]
        t_data = t_end-t_start+1
        t_steps = np.linspace(0, 1, t_data)
        #t_steps =3
        #x, y, vx, vy, psi
        seg_x = x[t_start:t_end+1]
        #print("seg_x shape: ",len(seg_x))
        seg_y = y[t_start:t_end+1]
        seg_vx = vx[t_start:t_end+1]
        seg_vy = vy[t_start:t_end+1]
        seg_psi = psi[t_start:t_end+1]

        path_gen_x = scipy.interpolate.interp1d(t_steps, seg_x)
        path_gen_y = scipy.interpolate.interp1d(t_steps, seg_y)
        path_gen_vx = scipy.interpolate.interp1d(t_steps, seg_vx)
        path_gen_vy = scipy.interpolate.interp1d(t_steps, seg_vy)
        path_gen_psi = scipy.interpolate.interp1d(t_steps, seg_psi)

        z_x = np.zeros(T)
        z_y = np.zeros(T)
        z_vx = np.zeros(T)
        z_vy = np.zeros(T)
        z_psi = np.zeros(T)
        #print(T)
        for t in range(T):
            #print(dt[t])
            z_x[t] = path_gen_x(dt[t])
            z_y[t] = path_gen_y(dt[t])
            z_vx[t] = path_gen_vx(dt[t])
            z_vy[t] = path_gen_vy(dt[t])
            z_psi[t] = path_gen_psi(dt[t])
            Trajectory.append(z_x[t])
            Trajectory.append(z_y[t])
            Trajectory.append(z_psi[t])
            Trajectory.append(z_vx[t])
            Trajectory.append(z_vy[t])

        Trajectory = np.array(Trajectory)
        alpha = 1e-13 #from promp papaer from france
        #alpha = 0
        #alpha = 1e-1
        #alpha = 0.1
        ww =get_Ridge_Refression (big_psi, Trajectory, alpha)
        # list_psi.append(np.copy(big_psi))
        #list_traj.append(np.copy(Trajectory))
        list_ww.append(np.copy(ww))

        # TESTER

        traj_x = Trajectory[0:T * D:D]
        traj_y = Trajectory[1:T * D:D]  # np.arange(traj_x.shape[0])#
        traj_psi = Trajectory[2:T * D:D]
        traj_vx = Trajectory[3:T * D:D]
        traj_vy = Trajectory[4:T * D:D]

        new_traj = np.dot(big_psi, ww)
        traj_x_p = new_traj[0:T * D:D]
        traj_y_p = new_traj[1:T * D:D]
        traj_psi_p = new_traj[2:T * D:D]
        traj_vx_p = new_traj[3:T * D:D]
        traj_vy_p = new_traj[4:T * D:D]

        if calc_accuracy:
            diff_x =  traj_x- traj_x_p
            diff_y =  traj_y - traj_y_p
            diff_psi = traj_psi- traj_psi_p
            diff_vx =  traj_vx- traj_vx_p
            diff_vy =  traj_vy - traj_vy_p

            RMSE_x = np.sqrt(sum(diff_x**(2))/T)
            RMSE_y = np.sqrt(sum(diff_y**(2))/T)
            RMSE_psi = np.sqrt(sum(diff_psi ** (2))/T)
            RMSE_vx = np.sqrt(sum(diff_vx**(2))/T)
            RMSE_vy= np.sqrt(sum(diff_vy ** (2))/T)
            print("Primitive variance", primitive_variance, " RMSE accuracy", RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy)
            #list_RMSE.append([RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy])





        if plot:
            if plot_states:
                fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
                fig.suptitle('Segmented Trajectory and States')
                ax1.plot(traj_x, "--", color="#ff6a6a", label="Origin x", linewidth=2.0)
                ax2.plot(traj_y,  "--", color="#ff6a6a", label="Origin y" , linewidth=2.0)
                ax3.plot(traj_psi, "--", color="#ff6a6a", label="Origin psi", linewidth=2.0)
                ax4.plot(traj_vx,  "--", color="#ff6a6a", label="Origin vx", linewidth=2.0)
                ax5.plot(traj_vy,  "--", color="#ff6a6a", label="Origin vy", linewidth=2.0)

                ax1.plot(traj_x_p, ".", color="b", label="Primitive x", linewidth=2.0)
                ax2.plot(traj_y_p,   ".", color="b", label="Primitive y" , linewidth=2.0)
                ax3.plot(traj_psi_p,  ".", color="b", label="Primitive psi", linewidth=2.0)
                ax4.plot(traj_vx_p,   ".", color="b", label="Primitive vx", linewidth=2.0)
                ax5.plot(traj_vy_p,   ".", color="b", label="Primitive vy", linewidth=2.0)
                ax1.legend()
                ax2.legend()
                ax3.legend()
                ax4.legend()
                ax5.legend()
                plt.show()


            else:
                fig = plt.figure()
                plt.plot(traj_x, traj_y, "--", color="#ff6a6a", label="Original", linewidth=2.0)
                #plt.show()
                #plt.holdon()

                #fig = plt.figure()

                plt.plot(traj_x_p, traj_y_p, ".", color="b", label="Primitive", linewidth=2.0)
                fig.suptitle('Trained trajectory and Primitive Trajectory')
                plt.xlabel('X')
                plt.ylabel('Y')

                plt.legend()
                plt.show()
    return list_ww


def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = (x[j]-y[i])**2
    return dist

def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0, 0] = distances[0, 0]

    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i - 1, 0]

    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j - 1]

        # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i - 1, j],  # insertion
                cost[i, j - 1],  # deletion
                cost[i - 1, j - 1]  # match
            ) + distances[i, j]

    return cost


def compute_euclidean_distance_matrix_2D(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y[0]), len(x[0])))
    for i in range(len(y[0])):
        for j in range(len(x[0])):
            dist[i,j] = (x[0][j]-y[0][i])**2+ (x[1][j]-y[1][i])**2
    return dist

def compute_accumulated_cost_matrix_2D(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    #distances = compute_euclidean_distance_matrix(x, y)
    distances = compute_euclidean_distance_matrix_2D(x,y)

    # Initialization
    cost = np.zeros((len(y[0]), len(x[0])))
    cost[0, 0] = distances[0, 0]

    for i in range(1, len(y[0])):
        cost[i, 0] = distances[i, 0] + cost[i - 1, 0]

    for j in range(1, len(x[0])):
        cost[0, j] = distances[0, j] + cost[0, j - 1]

        # Accumulated warp path cost
    for i in range(1, len(y[0])):
        for j in range(1, len(x[0])):
            cost[i, j] = min(
                cost[i - 1, j],  # insertion
                cost[i, j - 1],  # deletion
                cost[i - 1, j - 1]  # match
            ) + distances[i, j]

    return cost

def get_passed_phase(obs_traj, demonstrated_traj, plot= True):
    # x = [3, 1, 2, 2, 1]
    # y = [2, 0, 0, 3, 3, 1, 0]

    y = demonstrated_traj # demonstrated trajectory
    x = obs_traj #observed trajectory
    #distance, warp_path = fastdtw(obs_traj, demonstrated_traj, dist=euclidean)

    cost_matrix1 = compute_accumulated_cost_matrix_2D(x, y)

    res = [min(i) for i in zip(*cost_matrix1)][len(x[0])-1]
    r, c = np.where(cost_matrix1 == res)
    phase_completed = (r+1)/(len(y[0]))
    demonstration_time = len(x[0])/phase_completed
    print ("phase1 =", (r+1)/(len(y[0])))
    print("total time if phase is constant = ",  demonstration_time)

    # cost_matrix = compute_accumulated_cost_matrix(x, y)
    # res = [min(i) for i in zip(*cost_matrix)][len(x)-1]
    # r, c = np.where(cost_matrix == res)
    # phase_completed = (r+1)/(len(y))
    # demonstration_time = len(x)/phase_completed
    # print ("phase =", phase_completed)
    # print("total time if phase is constant = ", demonstration_time)
    if plot:
        plt.plot(obs_traj[0],obs_traj[1],'r')
        plt.plot(demonstrated_traj[0], demonstrated_traj[1],'b')
        plt.show()

    return phase_completed, demonstration_time

def compute_phi_index_obs_dtw(data, phi ,mean_weight, sigma_weight):
    #data is the observation (x,y,vx,vy,psi)
    #segment is original training data (x, y, vx, vy, psi)
    #can be used for the finding out the mean trajectory from the class or from the separate data

    #checking only in one dimension
    obs_len = int(len(data)/D)
    data_xy = [data[0:obs_len*D:D], data[1:obs_len*D:D]]
    demonstrated_data = np.dot(phi,mean_weight)
    demonstrated_len = int(len(demonstrated_data)/D)
    demonstrated_data_xy = [demonstrated_data[0:demonstrated_len*D:D], demonstrated_data[1:demonstrated_len*D:D]]
    phase_passed, tf = get_passed_phase(data_xy,demonstrated_data_xy, plot= True)


    T = demonstrated_len
    #T = 20
    print("the final time=", tf )
    print("demonstrated data length", demonstrated_len)
    print("observed data length ",obs_len)
    alpha = T/round(tf[0])





    phi_data, _ = generate_psi_dt_ts(int(tf[0]), alpha)



    inferred_traj = np.dot(phi_data, mean_weight)
    #inferred_traj = np.dot(phi, mean_weight)

    #tested with filter
    #inferred_traj, _, _ = reconstruct_seg_part_obv(data,mean_weight, sigma_weight, phi_data)

    #tested with extrapolation


    return phi, inferred_traj



def reconstruct_seg_part_obv(observation_data, mean_ww, sigma_ww, big_psi, prior_seg=[] ):

    span = D
    len_observation = int(len(observation_data)/span)
    #psi_obv_s = np.copy(big_psi[0:D])
    psi_obv = np.copy(big_psi[0:len_observation*D])
    plt.plot(observation_data[0:len_observation*D: D], observation_data[1:len_observation*D: D])
    plt.show()
    k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    mean_ww = mean_ww + np.dot(k_seg, (observation_data - np.dot(psi_obv, mean_ww)))
    sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))

    inferred_traj = np.dot(big_psi, mean_ww)

    # # each segment into T steps
    #
    # observation_data = np.array(observation_data)
    # observation_x = observation_data[0:T*D:D]
    # observation_x = observation_data[0:T*D:D]
    # observation_y = observation_data[1:T*D:D]
    # # observation_traj = np.vstack((observation_x,observation_y))
    #
    # span = D
    # len_observation = int(observation_data.shape[0]/span)
    # if not len_observation == T:
    #     print("Observed data length: ", len(observation_data))
    # prior = 2
    # lookahead = 2
    # # for the beginning of the segment
    # # if not len(prior_seg) == 0:
    # #     observation = prior_seg
    # #     psi_obv = np.copy(big_psi[-2*span:0])
    # #     k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    # #     mean_ww = mean_ww + np.dot(k_seg, (observation - np.dot(psi_obv, mean_ww)))
    # #     sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
    #
    #
    # # for the rest of the trajectory
    # for i in range (prior,len_observation):
    #     # Update K, mean, sigma, psi_obv
    #     #observation = observation_data[0:(i + 1) * span]
    #     observation = observation_data[(i-prior)*span:i*span]
    #     #psi_obv = np.copy(big_psi[0:(i+1)*span])
    #     psi_obv = np.copy(big_psi[(i-prior)*span:i*span])
    #
    #     k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    #     mean_ww = mean_ww + np.dot(k_seg, (observation - np.dot(psi_obv, mean_ww)))
    #     sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
    #
    #     #Infer for new trajectory
    #     # inferred_traj = np.dot(big_psi, mean_ww)
    #     # inferred_x = inferred_traj[0:inferred_traj.shape[0]:5]
    #     # inferred_y = inferred_traj[1:inferred_traj.shape[0]:5]
    #     # inferred = np.vstack((inferred_x, inferred_y))
    #
    #
    #     # plt.plot(observation[0:observation.shape[0]:5], ".", color="#ff6a6a", label="Observation", linewidth=2.0)
    #     # plt.plot(inferred[0],inferred[1], color = "#85d87f", label = "Inferred", linewidth = 3.0)
    #     # plt.show()
    #     inferred_traj = np.dot(big_psi, mean_ww)
    #     inferred_traj = inferred_traj[0:(i+lookahead)*span]
    #
    #     #inferred at each step
    #     inferred_x = inferred_traj[0:inferred_traj.shape[0]:D]
    #     inferred_y = inferred_traj[1:inferred_traj.shape[0]:D]
    #     # plt.plot(observation_x, observation_y, "--", color="#ff6a6a", label="Observation", linewidth=2.0)
    #     # plt.plot(inferred_x, inferred_y, ".", color = "#85d87f", label = "Inferred", linewidth = 2.0)
    #     # plt.legend()
    #     # plt.show()

    return inferred_traj, mean_ww, sigma_ww


if __name__ == "__main__":
    all_x, all_y, all_vx, all_vy, all_psi = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=100)
    indices = np.arange(len(all_x))
    np.random.shuffle(indices)
    train_x, train_y, train_vx, train_vy, train_psi = [], [], [], [], []
    test_x, test_y, test_vx, test_vy, test_psi = [], [], [], [], []
    for i in range(len(indices)):
        if i < len(indices) * 0.9:  # percent of data
            train_x.append(all_x[indices[i]])
            train_y.append(all_y[indices[i]])
            train_vx.append(all_vx[indices[i]])
            train_vy.append(all_vy[indices[i]])
            train_psi.append(all_psi[indices[i]])
            # print(np.sum(all_vx[indices[i]]- all_vy[indices[i]]))
        else:
            test_x.append(all_x[indices[i]])
            test_y.append(all_y[indices[i]])
            test_vx.append(all_vx[indices[i]])
            test_vy.append(all_vy[indices[i]])
            test_psi.append(all_psi[indices[i]])

    all_states_seg = []
    all_states_traj = []
    for i in range(len(train_x)):
        all_states_seg_temp, all_states_traj_temp = (
            state_seg_v2(train_x[i], train_y[i], train_psi[i], train_vx[i], train_vy[i], plot=True, seg_whole=False,
                         plot_states=True))
        all_states_seg.extend(all_states_seg_temp)
        all_states_traj.append(all_states_traj_temp)

    all_states_seg = np.array(all_states_seg)
    gmm = GaussianMixture(n_components=3, reg_covar=1, covariance_type="full", verbose=1)  # testing with 10, 5
    gmm.fit(all_states_seg)

    ###########################################################GMM segmentation block#############################################
    seg_ranges = state_seg_cluster_gmm(gmm, all_states_traj, plot=False)
    list_ent_ts = []
    list_mid_ts = []
    list_exit_ts = []
    # for i in range(len(seg_ranges)):
    #     for j in range(len(seg_ranges[i])):
    #         print(seg_ranges[i][j])
    #         print("-"*20)
    # for i in range(len(seg_ranges)):
    #     for j in range(len(seg_ranges[i])):
    #         # print(seg_ranges[i][j]['class'], "data_range", seg_ranges[i][j]['data_range'])
    #         # print ("-"*20)
    #         seg_class = seg_ranges[i][j]['class']
    #         seg_len = seg_ranges[i][j]['data_range'][1]- seg_ranges[i][j]['data_range'][0]
    #         print(seg_class, "data length", seg_len)
    #         print ("-"*20)

    #         if seg_class == 0:
    #             list_ent_ts.append(seg_len)
    #         elif seg_class == 1:
    #             list_mid_ts.append(seg_len)
    #         elif seg_class == 2:
    #             list_exit_ts.append(seg_len)
    #
    # list_ent_ts = np.array(list_ent_ts)
    # list_mid_ts = np.array(list_mid_ts)
    # list_exit_ts = np.array(list_exit_ts)
    # T = []
    # T.append(list_ent_ts)
    # T.append(list_mid_ts)
    # T.append(list_exit_ts)
    #
    # mean_ent_ts = list_ent_ts.mean(0)
    # mean_mid_ts = list_mid_ts.mean(0)
    # mean_exit_ts = list_exit_ts.mean(0)
    #
    # print(round(mean_ent_ts), len(list_ent_ts))
    # print (max(list_ent_ts), "-----", min(list_ent_ts))
    # print(round(mean_mid_ts),  len(list_mid_ts))
    # print (max(list_mid_ts), "-----", min(list_mid_ts))
    # print(round(mean_exit_ts),  len(list_exit_ts))
    # print (max(list_exit_ts), "-----", min(list_exit_ts))


    # clustering
    list_pos_ww = []
    list_neg_ww = []

    list_ent_ww = []
    list_mid_ww = []
    list_exit_ww = []
    psi_ent = []
    psi_mid = []
    psi_exit = []
    alphas_ent = []
    alphas_mid = []
    alphas_exit =[]
    j = 1
    T = 20 # normal
    for i in range(len(seg_ranges)):
        #data_tf = len(train_x[i])
        big_psi_list = []
        dt_list = []
        data_tf_list = []
        alpha_list = []
        for j in range(len(seg_ranges[i])):
            # print(seg_ranges[i][j]['class'], "data_range", seg_ranges[i][j]['data_range'])
            # print ("-"*20)
            seg_class = seg_ranges[i][j]['class']
            seg_len = seg_ranges[i][j]['data_range'][1] - seg_ranges[i][j]['data_range'][0]
            data_tf = seg_len
            data_tf_list.append(data_tf)





            alpha = T/data_tf
            big_psi, dt = generate_psi_dt_ts(data_tf,alpha)
            big_psi_list.append(big_psi)
            dt_list.append(dt)
            alpha_list.append(alpha)

            # list_ww = weight_gen_T(seg_ranges[i], train_x[i], train_y[i], train_vx[i], train_vy[i], train_psi[i], big_psi, dt,data_tf,
            #                      plot=False, calc_accuracy=False, plot_states=False)
        list_ww = weight_gen_T_wo_int(seg_ranges[i], train_x[i], train_y[i], train_vx[i], train_vy[i], train_psi[i], big_psi_list, dt_list,data_tf_list,
                         plot=False, calc_accuracy=False, plot_states=False)

        for j in range(len(list_ww)):
            seg_class = seg_ranges[i][j]['class']
            if seg_class == 0:
                list_ent_ww.append(list_ww[j])
                psi_ent.append(big_psi_list[j])
                alphas_ent.append(alpha_list[j])

            elif seg_class == 1:
                list_mid_ww.append(list_ww[j])
                psi_mid.append(big_psi_list[j])
                alphas_mid.append(alpha_list[j])
            elif seg_class == 2:
                list_exit_ww.append(list_ww[j])
                psi_exit.append(big_psi_list[j])
                alphas_exit.append(alpha_list[j])

    list_ent_ww = np.array(list_ent_ww)
    list_mid_ww = np.array(list_mid_ww)
    list_exit_ww = np.array(list_exit_ww)

    mean_ent_ww = list_ent_ww.mean(0)
    mean_mid_ww = list_mid_ww.mean(0)
    mean_exit_ww = list_exit_ww.mean(0)

    plt_psi_weight = False

    if plt_psi_weight:
        for i in range (len(psi_ent)):
            traj = np.dot(psi_ent[i], list_ent_ww[i])
            obs_data_len = int(len(traj)/D)
            print(obs_data_len)
            plt.plot(traj[0:obs_data_len*D: D], traj[1:obs_data_len*D: D], 'g')
            plt.plot(train_x[i], train_y[i], 'ro')
            plt.show()






    # trained_traj = np.dot(big_psi, mean_pos_ww)
    alphas_ent = np.array(alphas_ent)
    alphas_mid = np.array(alphas_mid)
    alphas_exit =np.array(alphas_exit)

    mean_alpha_ent = alphas_ent.mean(0)
    mean_alpha_mid = alphas_mid.mean(0)
    mean_alpha_exit = alphas_exit.mean(0)

    tf_mean_ent = round(T/mean_alpha_ent)
    tf_mean_mid = round(T/mean_alpha_mid)
    tf_mean_exit = round(T/mean_alpha_exit)

    big_psi_ent_mean, _ = generate_psi_dt_ts(tf_mean_ent, mean_alpha_ent)
    big_psi_mid_mean, _ = generate_psi_dt_ts(tf_mean_mid, mean_alpha_mid)
    big_psi_exit_mean, _ = generate_psi_dt_ts(tf_mean_exit, mean_alpha_exit)


    #diff_alpha tester
    big_psi_ent_mean_15, _ = generate_psi_dt_ts(tf_mean_ent, mean_alpha_ent*1.5)
    #this has been stored to find the mean trajectory
    psi_mean = [big_psi_ent_mean, big_psi_mid_mean, big_psi_exit_mean]
    alphas_mean = [mean_alpha_ent, mean_alpha_mid, mean_alpha_exit]
    weights_mean = [np.copy(mean_ent_ww), np.copy(mean_mid_ww), np.copy(mean_exit_ww)]

    trained_traj_ent = np.dot(big_psi_ent_mean, mean_ent_ww)
    trained_traj_ent_15 = np.dot(big_psi_ent_mean_15, mean_ent_ww)

    trained_traj_mid = np.dot(big_psi_mid_mean, mean_mid_ww)
    trained_traj_exit = np.dot(big_psi_exit_mean, mean_exit_ww)

    plot_tester_diff_alpha = True
    if plot_tester_diff_alpha:
        # for j in range(len(list_ent_ww)):
        #     # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        #     traj_all = np.dot(big_psi_ent_mean, list_ent_ww[j])
        #     plt.plot(traj_all[0:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        #     # plt.show()
        plt.plot(trained_traj_ent[0:len(trained_traj_ent):D], trained_traj_ent[1:len(trained_traj_ent) :D], 'bx', label="Primtive ", linewidth=2.0)
        plt.plot(trained_traj_ent_15[0:len(trained_traj_ent): D],  trained_traj_ent_15[1:len(trained_traj_ent):D], 'ro', label="mod ", linewidth=2.0)
    plt.show()

    weight_var_init = 0.2 * 0.2
    weights_covar_ent = np.ones((D * K, D * K)) * weight_var_init
    weight_mean_ent = list_ent_ww[0]
    weights_covar_mid = np.ones((D * K, D * K)) * weight_var_init
    weight_mean_mid = list_mid_ww[0]
    weights_covar_exit = np.ones((D * K, D * K)) * weight_var_init
    weight_mean_exit = list_exit_ww[0]
    weight_mean = list_ent_ww[0]
    weights_covar = np.ones((D * K, D * K)) * weight_var_init
    for demo_idx in range(list_ent_ww.shape[0]):
        state = list_ent_ww[demo_idx]
        weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
        temp = np.expand_dims(state - weight_mean, 1)
        weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (
                    demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
    #mean_ent_ww = weight_mean
    sigma_ent_ww = weights_covar

    weight_mean = list_mid_ww[0]
    weights_covar = np.ones((D * K, D * K)) * weight_var_init
    for demo_idx in range(list_mid_ww.shape[0]):
        state = list_mid_ww[demo_idx]
        weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
        temp = np.expand_dims(state - weight_mean, 1)
        weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (
                    demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
    #mean_mid_ww = weight_mean
    sigma_mid_ww = weights_covar

    weight_mean = list_exit_ww[0]
    weights_covar = np.ones((D * K, D * K)) * weight_var_init
    for demo_idx in range(list_exit_ww.shape[0]):
        state = list_exit_ww[demo_idx]
        weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
        temp = np.expand_dims(state - weight_mean, 1)
        weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (
                    demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
    #mean_exit_ww = weight_mean
    sigma_exit_ww = weights_covar




    plot_wts_mn_cluster = True
    if plot_wts_mn_cluster:
        trained_traj_upper = ((np.dot(np.dot(big_psi_ent_mean, sigma_ent_ww), big_psi_ent_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_ent_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_ent_mean, list_ent_ww[j])
            plt.plot(traj_all[0:len(trained_traj_ent):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_ent[0:len(trained_traj_ent):D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[0:len(trained_traj_ent):D]) + trained_traj_ent[0:len(trained_traj_ent):D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[0:len(trained_traj_ent):D]) + trained_traj_ent[0:len(trained_traj_ent):D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 0")
        plt.show()
        plt.figure()

        trained_traj_upper = ((np.dot(np.dot(big_psi_ent_mean, sigma_ent_ww), big_psi_ent_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_ent_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_ent_mean, list_ent_ww[j])
            plt.plot(traj_all[1:len(trained_traj_ent):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_ent[1:len(trained_traj_ent):D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[1:len(trained_traj_ent):D]) + trained_traj_ent[1:len(trained_traj_ent):D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[1:len(trained_traj_ent):D]) + trained_traj_ent[1:len(trained_traj_ent):D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 0")
        plt.show()
        plt.figure()

        trained_traj_upper = ((np.dot(np.dot(big_psi_mid_mean, sigma_mid_ww), big_psi_mid_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_mid_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_mid_mean, list_mid_ww[j])
            plt.plot(traj_all[0:len(trained_traj_mid):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_mid[0:len(trained_traj_mid):D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[0:len(trained_traj_mid):D]) + trained_traj_mid[0:len(trained_traj_mid):D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[0:len(trained_traj_mid):D]) + trained_traj_mid[0:len(trained_traj_mid):D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 1")
        plt.show()
        plt.figure()
        trained_traj_upper = ((np.dot(np.dot(big_psi_mid_mean, sigma_mid_ww), big_psi_mid_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_mid_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_mid_mean, list_mid_ww[j])
            plt.plot(traj_all[1:len(trained_traj_mid):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_mid[1:len(trained_traj_mid):D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[1:len(trained_traj_mid):D]) + trained_traj_mid[1:len(trained_traj_mid):D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[1:len(trained_traj_mid):D]) + trained_traj_mid[1:len(trained_traj_mid):D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 1")
        plt.show()
        plt.figure()
        trained_traj_upper = ((np.dot(np.dot(big_psi_exit_mean, sigma_exit_ww), big_psi_exit_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_exit_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_exit_mean, list_exit_ww[j])
            plt.plot(traj_all[0:len(trained_traj_exit):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_exit[0:len(trained_traj_exit):D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[0:len(trained_traj_exit):D]) + trained_traj_exit[0:len(trained_traj_exit):D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[0:len(trained_traj_exit):D]) + trained_traj_exit[0:len(trained_traj_exit):D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 2")
        plt.show()
        plt.figure()
        trained_traj_upper = ((np.dot(np.dot(big_psi_exit_mean, sigma_exit_ww), big_psi_exit_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_exit_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_exit_mean, list_exit_ww[j])
            plt.plot(traj_all[1:len(trained_traj_exit):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_exit[1:len(trained_traj_exit):D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[1:len(trained_traj_exit):D]) + trained_traj_exit[1:len(trained_traj_exit):D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[1:len(trained_traj_exit):D]) + trained_traj_exit[1:len(trained_traj_exit):D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 2")
        plt.show()
        plt.figure()


    plot_wts_mn_traj_cluster = True
    if plot_wts_mn_traj_cluster:
        trained_traj_upper = ((np.dot(np.dot(big_psi_ent_mean, sigma_ent_ww), big_psi_ent_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_ent_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_ent_mean, list_ent_ww[j])
            plt.plot(traj_all[0:len(trained_traj_ent):D], traj_all[1:len(trained_traj_ent):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_ent[0:len(trained_traj_ent):D], trained_traj_ent[1:len(trained_traj_ent):D], color="b", label="Primtive ", linewidth=2.0)
        # plt.plot(np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_ent[0:T * D:D], np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_ent[1:T * D:D], color="g", label="Upper SD ",
        #          linewidth=2.0)
        # plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_ent[0:T * D:D], color="m", label="Lower SD ",
        #          linewidth=2.0)


        trained_traj_upper = ((np.dot(np.dot(big_psi_mid_mean, sigma_mid_ww), big_psi_mid_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_mid_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_mid_mean, list_mid_ww[j])
            plt.plot(traj_all[0:len(trained_traj_mid):D], traj_all[1:len(trained_traj_mid):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_mid[0:len(trained_traj_ent):D], trained_traj_mid[1:len(trained_traj_ent):D], color="c", label="Primtive ", linewidth=2.0)
        # plt.plot(np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_mid[1:T * D:D], color="g", label="Upper SD ",
        #          linewidth=2.0)
        # plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_mid[1:T * D:D], color="m", label="Lower SD ",
        #          linewidth=2.0)
        # plt.legend()
        # plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 1")
        # plt.show()

        trained_traj_upper = ((np.dot(np.dot(big_psi_exit_mean, sigma_exit_ww), big_psi_exit_mean.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_exit_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi_exit_mean, list_exit_ww[j])
            plt.plot(traj_all[0:len(trained_traj_exit):D],traj_all[1:len(trained_traj_exit):D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_exit[0:len(trained_traj_exit):D], trained_traj_exit[1:len(trained_traj_exit):D], color="g", label="Primtive ", linewidth=2.0)
        # plt.plot(np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_exit[1:T * D:D], color="g", label="Upper SD ",
        #          linewidth=2.0)
        # plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_exit[1:T * D:D], color="m", label="Lower SD ",
        #          linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean of X-Y state segments with average cluster real time data")

        plot_real_data= True
        if plot_real_data:
            for i in range(len(test_x)):
                plt.plot(test_x[i], test_y[i], "--",  color="y")

        plt.show()


        # trained_traj_upper = ((np.dot(np.dot(big_psi_ent_mean, sigma_ent_ww), big_psi_ent_mean.T)).diagonal()) * 1  # + trained_traj
        # for j in range(len(list_ent_ww)):
        #     # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        #     traj_all = np.dot(big_psi_ent_mean, list_ent_ww[j])
        #     plt.plot(traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        #     # plt.show()
        # plt.plot(trained_traj_ent[1:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        # plt.plot(np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_ent[1:T * D:D], color="g", label="Upper SD ",
        #          linewidth=2.0)
        # plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_ent[1:T * D:D], color="m", label="Lower SD ",
        #          linewidth=2.0)
        # plt.legend()
        # plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 0")
        # plt.show()
        # plt.figure()
    #reconstruction of the vehicle for testing
    obs_traj= []
    obs_data_len = 50
    for i in range(len(test_x)): # all data for testing
        for j in range(obs_data_len):
            obs_traj.append(test_x[i][j])
            obs_traj.append(test_y[i][j])
            obs_traj.append(test_psi[i][j])
            obs_traj.append(test_vx[i][j])
            obs_traj.append(test_vy[i][j])

        #obs_traj = [test_x[i][1:10], test_y[i][1:10], test_vx[i][1:10], test_vy[i][1:10], test_psi[i][1:10]]
        #find out which cluster of data it belongs to
        # class_index = compute_class(obs_traj, psi_mean, alphas_mean, weights_mean)
        #
        # if class_index == 0:
        #     all_phi = psi_ent
        #     all_alpha = alphas_ent
        #     mean_weight = weights_mean[0]
        # if class_index == 1:
        #     all_phi = psi_mid
        #     all_alpha = alphas_mid
        #     mean_weight = weights_mean[1]
        # if class_index == 2:
        #     all_phi = psi_exit
        #     all_alpha = alphas_exit
        #     mean_weight = weights_mean[2]

        # find out the length of data
        #phi_obs, _ = compute_phi_index_obs(obs_traj, all_phi, all_alpha,mean_weight)

        #phi_obs, mean_weight = compute_phi_index_obs(obs_traj, psi_ent, alphas_ent, list_ent_ww)
        #inferred_traj= np.dot(big_psi_exit_mean, weights_mean[0])
        #inferred_traj = np.dot(phi_obs,mean_mid_ww)
        #inferred_traj = np.dot(phi_obs, mean_weight)
        #inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg_part_obv(obs_traj, mean_weight, sigma_ent_ww, phi_obs)
        # #inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg_part_obv(obs_traj, mean_ent_ww, sigma_ent_ww,
        #                                                                     phi_obs)



        ###################dtw data#############333
        obs_traj = np.array(obs_traj)
        #big_psi_ent_mean, _ = generate_psi_dt_ts(20, 1)
        #_, inferred_traj = compute_phi_index_obs_dtw(obs_traj,big_psi_ent_mean,mean_ent_ww, sigma_ent_ww)
        #plt.plot(obs_traj[0:obs_data_len*D: D],obs_traj[1:obs_data_len*D:D], 'ro')

        plt.plot(test_x[i], test_y[i], 'g')
        #inferred_traj_len = int(len(inferred_traj)/D)
        # inferred_traj_mp1 = np.dot(big_psi_ent_mean,mean_ent_ww)
        # inferred_traj_mp2 = np.dot(big_psi_mid_mean, mean_mid_ww)
        # inferred_traj_mp3 = np.dot(big_psi_exit_mean, mean_exit_ww)


        inferred_traj_mp1 = np.copy(trained_traj_ent)
        inferred_traj_mp2 = np.copy(trained_traj_mid)
        inferred_traj_mp3 = np.copy(trained_traj_exit)

        inferred_traj_mp1_len= int(len(inferred_traj_mp1)/D)
        inferred_traj_mp2_len = int(len(inferred_traj_mp2) / D)
        inferred_traj_mp3_len = int(len(inferred_traj_mp3) / D)
        # #plt.plot(trained_traj_ent[0:obs_data_len*D: D],trained_traj_ent [1:obs_data_len*D:D], 'ro' )
        # #plt.plot()
        plt.plot(obs_traj[0:obs_data_len*D: D],obs_traj[1:obs_data_len*D:D], 'ro')
        #plt.plot(inferred_traj[0:obs_data_len*D: D], inferred_traj[1: obs_data_len*D: D], 'bo')
        plt.plot(inferred_traj_mp1[0:inferred_traj_mp1_len*D: D], inferred_traj_mp1[1: inferred_traj_mp1_len*D: D], 'yo')
        plt.plot(inferred_traj_mp2[0:inferred_traj_mp2_len * D: D], inferred_traj_mp2[1: inferred_traj_mp2_len * D: D],
                 'co')
        plt.plot(inferred_traj_mp3[0:inferred_traj_mp3_len * D: D], inferred_traj_mp1[1: inferred_traj_mp3_len * D: D],
                 'mo')

        #plt.plot(inferred_traj[0:obs_data_len*D: D],inferred_traj[1:obs_data_len*D:D], 'bo')
        plt.show()
        obs_traj = []


        # timestep = np.arange(1, obs_data_len+1)
        # plt.plot(timestep, obs_traj[0:obs_data_len*D: D])
        # plt.plot(timestep, inferred_traj[0:obs_data_len*D: D])
        # plt.show()

    # def_mean_pos = mean_pos_ww.copy()
    # def_pos_sigma = sigma_pos_ww.copy()
    # inferred_test_traj = []
    # for i in range (len(seg_ranges_obv)):
    #     current_test_traj= seg_ranges_obv[i]
    #     current_inferred_traj = []
    #     #plt.figure(10)
    #
    #     mean_pos_ww = def_mean_pos
    #     sigma_pos_ww = def_pos_sigma
    #     for j in range (len (current_test_traj)):
    #         current_segment = current_test_traj[j]
    #         #check is positive or negative class
    #         if current_segment[-D]>0 :
    #             # if j ==0:
    #             inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg(current_segment,mean_pos_ww, sigma_pos_ww)
    #             # else:
    #             #     inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg(current_segment, mean_pos_ww, sigma_pos_ww, prior_seg= current_segment[-2*D: 0])
    #         # else:
    #         #     inferred_traj, mean_neg_ww, sigma_neg_ww = reconstruct_seg(current_segment, mean_neg_ww, sigma_neg_ww)                    # --- COMMENTED OUT NEGATIVE CLUSTERING
    #
    #
    #         current_inferred_traj.append(inferred_traj)
    #
    #         inferred_x = inferred_traj[0:inferred_traj.shape[0]:D]
    #         inferred_y = inferred_traj[1:inferred_traj.shape[0]:D]
    #         observation_x = current_segment[0:current_segment.shape[0]:D]
    #         observation_y = current_segment[1:current_segment.shape[0]:D]
    #         plt.plot(observation_x, observation_y, "--", color="#ff6a6a", label="Observation", linewidth=2.0)
    #         plt.plot(inferred_x, inferred_y, ".", color="b", label="Inferred", linewidth=2.0)
    #
    #     plt.show()
    #     inferred_test_traj.append(current_inferred_traj)
    # plt.legend()
    # plt.show()




