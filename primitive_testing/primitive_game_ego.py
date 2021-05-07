import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy.interpolate
import csv
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sklearn.mixture import GaussianMixture

from example_tester import extract_dataset_traj_active
from example_tester import extract_dataset_traj
from GT_primitive_primary import state_seg_v2
from GT_primitive_primary import state_seg_cluster_gmm
from GT_primitive_primary import generate_psi_dt
from GT_primitive_primary import weight_gen
#from GT_primitive_primary import seg_2_traj
from primtive_4 import compute_phi_index_obs_dtw

#sys.path.append('C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset/data')
np.random.seed(123)

T = 20 #number of timesteps
K = 80 # number of basis function acc to the paper
D = 5 # dimension of data


def compute_class(data, all_phi, mean_weights, sigma_weights):
    #data is the observation (x,y,vx,vy,psi)
    #segment is original training data (x, y, vx, vy, psi)
    #can be used for the finding out the mean trajectory from the class or from the separate data
    obs_len = 100
    diff_sum=[]
    for i in range (len(mean_weights)):
        phi_data=np.dot(all_phi, mean_weights[i])
        #print(data)
        #print(phi_data[0: len(data)])
        diff_data = abs(data[0:obs_len]- phi_data[0:obs_len])
        #print(diff_data)
        diff_sum.append(sum(diff_data))
        #diff_sum[i] = sum(diff_data)

    index = diff_sum.index(min(diff_sum))
    return mean_weights[index], sigma_weights[index]

def reconstruct_seg_lookahead(observation_data, mean_weights_data, sigma_weights_data, prior_seg=[] ):
    obs_traj_1 = observation_data[0:100] #first 20 observations
    mean_ww, sigma_ww = compute_class(obs_traj_1, big_psi, mean_weights_data, sigma_weights_data)
    # each segment into T steps
    span = D
    len_observation = int(len(observation_data) / span)
    # observation_x = observation_data[0:len_observation*D:D]
    # observation_y = observation_data[1:len_observation*D:D]
    observation_data = np.array(observation_data)

    phi_data, inferred_traj = compute_phi_index_obs_dtw(obs_traj_1, big_psi, mean_ww, sigma_ww, use_KF=False)


    # with open('tester_file_primitive3.csv', 'w', ) as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     for word in inferred_traj:
    #         wr.writerow([word])
    #np.savetxt("primitive_traj3.csv", inferred_traj, delimiter=",")

    inferred_traj_len = int(len(inferred_traj) / span)
    observation_x = observation_data[0:inferred_traj_len*D:D]
    observation_y = observation_data[1:inferred_traj_len*D:D]
    plt.plot(observation_x, observation_y, "x", color="#ff6a6a", label="Observation", linewidth=2.0)
    plt.plot(inferred_traj[0: inferred_traj_len*D: D], inferred_traj[1:inferred_traj_len*D:D], ".", color = "b", label= "Inferred traj")
    plt.legend()
    plt.show()




    #len_observation = int(observation_data.shape[0]/span)

    if not len_observation == T:
        print("Observed data length: ", len(observation_data))
    #prior = 10
    #lookahead = 10

    # for the rest of the trajectory
    #for i in range (span, len_observation):
    for i in range(1, inferred_traj_len-3):
        # Update K, mean, sigma, psi_obv
        observation = observation_data[(i-1)*span:i * span]

        #observation = observation_data[(i-prior)*span:i*span]
        psi_obv = np.copy(phi_data[(i-1)*span:i*span])
        #psi_obv = np.copy(big_psi[(i-prior)*span:i*span])

        k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
        mean_ww = mean_ww + np.dot(k_seg, (observation - np.dot(psi_obv, mean_ww)))
        sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))

        #Infer for new trajectory
        # inferred_traj = np.dot(big_psi, mean_ww)
        # inferred_x = inferred_traj[0:inferred_traj.shape[0]:5]
        # inferred_y = inferred_traj[1:inferred_traj.shape[0]:5]
        # inferred = np.vstack((inferred_x, inferred_y))


        # plt.plot(observation[0:observation.shape[0]:5], ".", color="#ff6a6a", label="Observation", linewidth=2.0)
        # plt.plot(inferred[0],inferred[1], color = "#85d87f", label = "Inferred", linewidth = 3.0)
        # plt.show()
        inferred_traj = np.dot(phi_data, mean_ww)
        #inferred_traj = inferred_traj[i*span:(i+1)*span]


        inferred_traj = inferred_traj[(i-1)*span:(i+1) * span]

        #inferred at each step
        inferred_x = inferred_traj[0:inferred_traj.shape[0]:D]
        inferred_y = inferred_traj[1:inferred_traj.shape[0]:D]
        plt.plot(observation_x, observation_y, "x", color="#ff6a6a", label="Observation", linewidth=2.0)
        plt.plot(inferred_x, inferred_y, ".", color = "#85d87f", label = "Inferred", linewidth = 2.0)
        plt.legend()
        plt.show()

    return inferred_traj, mean_ww, sigma_ww


def reconstruct_seg(observation_data, mean_ww, sigma_ww, prior_seg=[] ):
    # each segment into T steps

    observation_data = np.array(observation_data)
    observation_x = observation_data[0:T*D:D]
    observation_y = observation_data[1:T*D:D]
    # observation_traj = np.vstack((observation_x,observation_y))

    span = D
    len_observation = int(observation_data.shape[0]/span)
    if not len_observation == T:
        print("Observed data length: ", len(observation_data))
    prior = 2
    lookahead = 2
    # for the beginning of the segment
    # if not len(prior_seg) == 0:
    #     observation = prior_seg
    #     psi_obv = np.copy(big_psi[-2*span:0])
    #     k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    #     mean_ww = mean_ww + np.dot(k_seg, (observation - np.dot(psi_obv, mean_ww)))
    #     sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))


    # for the rest of the trajectory
    for i in range (prior,len_observation):
        # Update K, mean, sigma, psi_obv
        #observation = observation_data[0:(i + 1) * span]
        observation = observation_data[(i-prior)*span:i*span]
        #psi_obv = np.copy(big_psi[0:(i+1)*span])
        psi_obv = np.copy(big_psi[(i-prior)*span:i*span])

        k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
        mean_ww = mean_ww + np.dot(k_seg, (observation - np.dot(psi_obv, mean_ww)))
        sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))

        #Infer for new trajectory
        # inferred_traj = np.dot(big_psi, mean_ww)
        # inferred_x = inferred_traj[0:inferred_traj.shape[0]:5]
        # inferred_y = inferred_traj[1:inferred_traj.shape[0]:5]
        # inferred = np.vstack((inferred_x, inferred_y))


        # plt.plot(observation[0:observation.shape[0]:5], ".", color="#ff6a6a", label="Observation", linewidth=2.0)
        # plt.plot(inferred[0],inferred[1], color = "#85d87f", label = "Inferred", linewidth = 3.0)
        # plt.show()
        inferred_traj = np.dot(big_psi, mean_ww)
        inferred_traj = inferred_traj[0:(i+lookahead)*span]

        #inferred at each step
        inferred_x = inferred_traj[0:inferred_traj.shape[0]:D]
        inferred_y = inferred_traj[1:inferred_traj.shape[0]:D]
        # plt.plot(observation_x, observation_y, "--", color="#ff6a6a", label="Observation", linewidth=2.0)
        # plt.plot(inferred_x, inferred_y, ".", color = "#85d87f", label = "Inferred", linewidth = 2.0)
        # plt.legend()
        # plt.show()

    return inferred_traj, mean_ww, sigma_ww


def seg_2_traj(seg_range,x,y,vx,vy,psi):
    list_traj =[]
    for seg_idx in range(len(seg_range)):
    #for seg_idx in range(1):
        Trajectory = []
        segment= seg_range[seg_idx]
        data_r = segment["data_range"]
        t_start = data_r[0]
        t_end = data_r[1]
        t_data = t_end-t_start+1
        t_steps = np.linspace(0, 1, t_data)
        #x, y, vx, vy, psi
        seg_x = x[t_start:t_end+1]
        #print("seg_x shape: ",len(seg_x))
        #print(t_data)
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
        for t in range(T):
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
        list_traj.append(Trajectory)
    return list_traj

if __name__ == "__main__":
    #interacting_all = get_primitive_info()
    track_file_name = "../recorded_trackfiles/Scenario4/vehicle_tracks_000.csv"
    #interacting trajectory
    active_x, active_y, active_vx, active_vy, active_psi, active_int, active_b, active_e =  extract_dataset_traj_active("Scenario4", False, [3], [5], data_lim=100, track_id_list = [37,47,77]) # [37,47,77]
    active_2x, active_2y, active_2vx, active_2vy, active_2psi, active_2int, active_2b, active_2e =  extract_dataset_traj_active("Scenario4", False, [3], [5], data_lim=100, track_id_list = [41, 46, 60]) # [41,46,60]
    #baseline trajectory
    all_x, all_y, all_vx, all_vy, all_psi = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=1000)
    all_x2, all_y2, all_vx2, all_vy2, all_psi2, all_int, all_2b, all_2e = extract_dataset_traj_active("Scenario4", True,[4], [2], data_lim=100, track_id_list=[19, 107])

    #extract the primtive mean and variance
    # c1_mean_ent_ww, c1_mean_mid_ww, c1_mean_exit_ww, c1_sigma_ent_ww, c1_sigma_mid_ww, c1_sigma_exit_ww, gmm1 =  get_primitive(all_x, all_y, all_psi, all_vx, all_vy)
    # c2_mean_ent_ww, c2_mean_mid_ww, c2_mean_exit_ww, c2_sigma_ent_ww, c2_sigma_mid_ww, c2_sigma_exit_ww, gmm2 = get_primitive(
    #     all_x2, all_y2, all_psi2, all_vx2, all_vy2)


    # print("total traj ", len(int_x))
    plt.figure()
    # plt.plot(active_x[1], active_y[1], 'b', label='interacting vehicle')
    # plt.plot(active_2x[1], active_2y[1], 'r', label='other interacting vehicle')
    plt.plot(active_x[0], active_y[0], 'b', label='interacting vehicle')
    plt.plot(active_2x[0], active_2y[0], 'r', label='other interacting vehicle')
    plt.plot(all_x[0], all_y[0], 'g', label = 'non interacting vehicle')
    plt.plot(all_x2[0], all_y2[0], 'y', label = 'other non interacting vehicle')
    plt.legend()
    plt.show()

    #motion primitive gives the trajectory points and the trajectory points
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
    # redoing testing based on interacting cases
    test_x, test_y, test_vx, test_vy, test_psi = [], [], [], [], []

    for i in range(len(active_x)):
        test_x.append(active_x[i])
        test_y.append(active_y[i])
        test_vx.append(active_vx[i])
        test_vy.append(active_vy[i])
        test_psi.append(active_psi[i])

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
    big_psi, dt = generate_psi_dt()

    list_ent_ww = []
    list_mid_ww = []
    list_exit_ww = []

    j = 1
    for i in range(len(seg_ranges)):
        list_ww = weight_gen(seg_ranges[i], train_x[i], train_y[i], train_vx[i], train_vy[i], train_psi[i], big_psi, dt,
                             plot=False, calc_accuracy=False, plot_states=False)

        for j in range(len(list_ww)):
            seg_class = seg_ranges[i][j]['class']
            if seg_class == 0:
                list_ent_ww.append(list_ww[j])
            elif seg_class == 1:
                list_mid_ww.append(list_ww[j])
            elif seg_class == 2:
                list_exit_ww.append(list_ww[j])

    list_ent_ww = np.array(list_ent_ww)
    list_mid_ww = np.array(list_mid_ww)
    list_exit_ww = np.array(list_exit_ww)

    mean_ent_ww = list_ent_ww.mean(0)
    mean_mid_ww = list_mid_ww.mean(0)
    mean_exit_ww = list_exit_ww.mean(0)

    # trained_traj = np.dot(big_psi, mean_pos_ww)
    trained_traj_ent = np.dot(big_psi, mean_ent_ww)
    trained_traj_mid = np.dot(big_psi, mean_mid_ww)
    trained_traj_exit = np.dot(big_psi, mean_exit_ww)

    #############################################initializing for 3 clusters#############
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
    mean_ent_ww = weight_mean
    sigma_ent_ww = weights_covar

    weight_mean = list_mid_ww[0]
    weights_covar = np.ones((D * K, D * K)) * weight_var_init
    for demo_idx in range(list_mid_ww.shape[0]):
        state = list_mid_ww[demo_idx]
        weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
        temp = np.expand_dims(state - weight_mean, 1)
        weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (
                demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
    mean_mid_ww = weight_mean
    sigma_mid_ww = weights_covar

    weight_mean = list_exit_ww[0]
    weights_covar = np.ones((D * K, D * K)) * weight_var_init
    for demo_idx in range(list_exit_ww.shape[0]):
        state = list_exit_ww[demo_idx]
        weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
        temp = np.expand_dims(state - weight_mean, 1)
        weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (
                demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
    mean_exit_ww = weight_mean
    sigma_exit_ww = weights_covar

    mean_weights_data = [mean_ent_ww, mean_mid_ww, mean_exit_ww]
    sigma_weights_data = [sigma_ent_ww, sigma_mid_ww, sigma_exit_ww]

    #############################################GT approximation of the motion primitive ################################
    ###################new reconstruction with the KF points ##############
    seg_ranges_obv = []
    all_states_seg_test = []
    all_states_traj_test = []
    for i in range(len(test_x)):
        all_states_seg_temp, all_states_traj_temp = (
            state_seg_v2(test_x[i], test_y[i], test_psi[i], test_vx[i], test_vy[i], plot=False, seg_whole=False,
                         plot_states=False))
        all_states_seg_test.extend(all_states_seg_temp)
        all_states_traj_test.append(all_states_traj_temp)

        # seg_test_obs = state_segmentation(test_x[i], test_y[i],  test_psi[i], test_vx[i], test_vy[i], plot=False, seg_whole= False)
        # seg_ranges_obv.append(seg_2_traj(seg_test_obs, test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i]))

    seg_ranges_test = state_seg_cluster_gmm(gmm, all_states_traj_test, plot=False)

    for i in range(len(seg_ranges_test)):
        seg_ranges_obv.append(seg_2_traj(seg_ranges_test[i], test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i]))
        print(seg_ranges_test[i][2]["data_range"], "--------", len(test_x[i]))






    inferred_test_traj = []
    RMSE_ent = []
    RMSE_mid = []
    RMSE_exit = []
    for i in range(len(seg_ranges_obv)):
        current_test_traj = seg_ranges_obv[2]
        current_inferred_traj = []
        # plt.figure(10)

        #for j in range(len(current_test_traj)):
        current_segment = current_test_traj[2]
        current_segment_len = int(len(current_segment)/5)
        plt.plot(current_segment[0:current_segment_len*D:D], current_segment[1:current_segment_len*D:D])
        plt.show()
        # check if the trajectory is close by
        inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg_lookahead(current_segment, mean_weights_data,
                                                                             sigma_weights_data)
        #use the game primitive then

        #current_cluster = int(seg_ranges_test[i][2]['class'])

            # if current_cluster == 0:
            #     inferred_traj, _, _ = reconstruct_seg_goal(current_segment, np.copy(mean_ent_ww), np.copy(sigma_ent_ww))
            # elif current_cluster == 1:
            #     inferred_traj, _, _ = reconstruct_seg_goal(current_segment, np.copy(mean_mid_ww), np.copy(sigma_mid_ww))
            # elif current_cluster == 2:
            #     inferred_traj, _, _ = reconstruct_seg_goal(current_segment, np.copy(mean_exit_ww),
            #                                                np.copy(sigma_exit_ww))

    ############################################### this has the kf for one timestep lookahead
    # obs_traj = []
    # inferred_test_traj = []
    # for i in range(len(test_x)):  # all data for testing
    #     for j in range(len(test_x[i])):
    #         obs_traj.append(test_x[i][j])
    #         obs_traj.append(test_y[i][j])
    #         obs_traj.append(test_psi[i][j])
    #         obs_traj.append(test_vx[i][j])
    #         obs_traj.append(test_vy[i][j])
    #
    #     obs_traj = np.array(obs_traj)
    #     current_inferred_traj = []
    #
    #
    #
    #     inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg_lookahead(obs_traj, mean_weights_data, sigma_weights_data)
    #     obs_traj = []


    #     if current_segment[-D]>0 :
    #             inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg_goal(current_segment,mean_pos_ww, sigma_pos_ww)
    #         else:
    #             inferred_traj, mean_neg_ww, sigma_neg_ww = reconstruct_seg_goal(current_segment, mean_neg_ww, sigma_neg_ww)                    # --- COMMENTED OUT NEGATIVE CLUSTERING
    #         #
    #         #
    #         # current_inferred_traj.append(inferred_traj)
    #
    #         inferred_x = inferred_traj[0:inferred_traj.shape[0]:D]
    #         inferred_y = inferred_traj[1:inferred_traj.shape[0]:D]
    #         # primitive_traj= big_psi* mean_pos_ww
    #         # primitive_x = primitive_traj[0:current_segment.shape[0]:D]
    #         # primitive_y = primitive_traj[0:current_segment.shape[0]:D]
    #         observation_x = current_segment[0:current_segment.shape[0]:D]
    #         observation_y = current_segment[1:current_segment.shape[0]:D]
    #         plt.plot(observation_x, observation_y, "--", color="#ff6a6a", label="Observation", linewidth=2.0)
    #         plt.plot(inferred_x, inferred_y, ".", color="b", label="Inferred", linewidth=2.0)
    #
    #         plt.show()
    #     inferred_test_traj.append(current_inferred_traj)
    # plt.legend()
    # #plt.show()
    #

    ################# this has the KF with each time step

    #
    # obs_traj = []
    # obs_data_len = 50
    # for i in range(len(test_x)):  # all data for testing
    #     for j in range(obs_data_len):
    #         obs_traj.append(test_x[i][j])
    #         obs_traj.append(test_y[i][j])
    #         obs_traj.append(test_psi[i][j])
    #         obs_traj.append(test_vx[i][j])
    #         obs_traj.append(test_vy[i][j])
    #
    #     obs_traj = np.array(obs_traj)
    #     plt.plot(test_x[i], test_y[i], 'g')
    #
    #     #compute which class the trajectory belongs to
    #     mean_class, sigma_class = compute_class(obs_traj,big_psi, mean_weights_data, sigma_weights_data)
    #
    #     #_, inferred_traj = compute_phi_index_obs_dtw(obs_traj, big_psi, mean_ent_ww, sigma_ent_ww, use_KF=False)
    #     #_, inferred_traj = compute_phi_index_obs_dtw(obs_traj, big_psi, mean_exit_ww, sigma_exit_ww, use_KF=False)
    #     _, inferred_traj = compute_phi_index_obs_dtw(obs_traj, big_psi, mean_class, sigma_class, use_KF=False)
    #
    #
    #     # inferred_traj_len = int(len(inferred_traj)/D)
    #     plt.plot(test_x[i], test_y[i], 'g')
    #     plt.plot(obs_traj[0:obs_data_len * D: D], obs_traj[1:obs_data_len * D:D], 'ro', label='Observation')
    #     plt.plot(inferred_traj[0:len(inferred_traj): D], inferred_traj[1: len(inferred_traj): D], 'bo',
    #              label='inferred all')
    #     plt.plot(inferred_traj[0:obs_data_len * D: D], inferred_traj[1:obs_data_len * D: D], 'go',
    #              label='inferred obs')
    #     plt.legend()
    #     plt.show()
    #     inferred_x = inferred_traj[0:obs_data_len * D:D]
    #     inferred_y = inferred_traj[1:obs_data_len * D:D]
    #     inferred_psi = inferred_traj[2:obs_data_len * D:D]
    #     inferred_vx = inferred_traj[3:obs_data_len * D:D]
    #     inferred_vy = inferred_traj[4:obs_data_len * D:D]
    #     # primitive_traj= big_psi* mean_pos_ww
    #     # primitive_x = primitive_traj[0:current_segment.shape[0]:D]
    #     # primitive_y = primitive_traj[0:current_segment.shape[0]:D]
    #     observation_x = obs_traj[0:obs_data_len * D:D]
    #     observation_y = obs_traj[1:obs_data_len * D:D]
    #     observation_psi = obs_traj[2:obs_data_len * D:D]
    #     observation_vx = obs_traj[3:obs_data_len * D:D]
    #     observation_vy = obs_traj[4:obs_data_len * D:D]
    #     calc_accuracy = True
    #     if calc_accuracy:
    #         diff_x = inferred_x - observation_x
    #         diff_y = inferred_y - observation_y
    #         diff_psi = inferred_psi - observation_psi
    #         diff_vx = inferred_vx - observation_vx
    #         diff_vy = inferred_vy - observation_vy
    #
    #         RMSE_x = np.sqrt(sum(diff_x ** (2)) / obs_data_len)
    #         RMSE_y = np.sqrt(sum(diff_y ** (2)) / obs_data_len)
    #         RMSE_psi = np.sqrt(sum(diff_psi ** (2)) / obs_data_len)
    #         RMSE_vx = np.sqrt(sum(diff_vx ** (2)) / obs_data_len)
    #         RMSE_vy = np.sqrt(sum(diff_vy ** (2)) / obs_data_len)
    #         print("Segment ", i, " RMSE accuracy", RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy)
    #
    #     obs_traj = []
