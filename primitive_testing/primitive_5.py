import numpy as np
import matplotlib.pyplot as plt
from example_tester import generate_demo_traj
from example_tester import extract_dataset_traj
from example_tester import extract_dataset_traj_active
import scipy.interpolate
import sys
import csv
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndimage


sys.path.append('C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset/data')
np.random.seed(123)

# constants for the primitive
# T = 20 #number of timesteps
# K = 8 # number of basis function acc to the paper
# T = 100 #number of timesteps
# K = 25 # number of basis function acc to the paper
T = 20 #number of timesteps
K = 80 # number of basis function acc to the paper
D = 5 # dimension of data


class data_node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.isroot= False
        self.previous_states = []
        self.cost= []
        self.level= -1
        self.parent = ""
        self.current_action= []
        self.current_states = []
        self.child_count= 0
        self.finish_count= 0

def dynamics(X, u):
    sx, sy, v_x, v_y, ang = X[0], X[1], X[2], X[3], X[4]
    dt = 0.1

    ang_new = ang + u[0] *dt
    vx_new = v_x + u[1] * np.cos(ang)*dt
    vy_new = v_y + u[1] * np.sin(ang)* dt
    sx_new = sx + vx_new * dt
    sy_new = sy + vy_new * dt

    return [sx_new, sy_new, vx_new, vy_new, ang_new]

def constant_velocity_model(X):
    sx, sy, v_x, v_y, ang = X[0], X[1], X[2], X[3], X[4]
    dt = 0.1
    sx_new = sx + v_x * dt
    sy_new = sy + v_y * dt
    return [sx_new, sy_new, v_x, v_y, ang]

def Reverse(lst):
    return [ele for ele in reversed(lst)]

def make_demonstrations(vehicle_no):
    name = 'vehicle_tracks_000.csv'
    with open('C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset/data/' + name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        vx = []
        vy = []
        psi = []
        next(reader)
        for row in reader:
            row = [float(val) for val in row]

            # col.append(row[1])
            if row[0]== vehicle_no: # vehicle 2 data for testing
                x.append(row[3])
                y.append(row[4])
                vx.append(row[5])
                vy.append(row[6])
                psi.append(row[7])
        #print (col)

        # generate function to interpolate the desired trajectory
        import scipy.interpolate

    # plt.figure(0)
    # plt.plot(x,y)
    # plt.title('tester_plot')
    return x,y, vx, vy, psi

def variance(data):
# Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance

def mean(data):
    n = len(data)
    mean = sum(data)/n
    return mean

def get_GausRBF(z_t,mean_k,sigma,D):
    return np.exp((-0.5*(z_t-mean_k)*(z_t-mean_k))*(1./sigma)) / (np.sqrt(np.power(2*np.pi,D) * sigma))

def get_Ridge_Refression (X, Y, alpha):
    I = np.identity(X.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha * I), X.T), Y)
    #w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
    return w



#state segmentation and clustering
def state_segmentation(x, y, psi, vx, vy, plot = True, seg_whole = True, plot_states= True):
    var = np.zeros(D)
    state = np.array((x[0], y[0], psi[0], vx[0], vy[0]))
    #print(np.sum(vx[0:100]-vy[0:100]))
    data_mean = state
    #data_covar = np.fill_diagonal(np.identity(5), var)
    data_covar = np.zeros((D, D))

    #list for data
    seg_range = []
    current_range_start = 0
    #print(len(x))
    for demo_idx in range(1, len(x)):
        state = np.array((x[demo_idx], y[demo_idx],  psi[demo_idx], vx[demo_idx], vy[demo_idx]))
        data_mean = (data_mean * demo_idx + state) / (demo_idx+1) #eq3
        temp = np.expand_dims(state-data_mean, 1)
        data_covar = (demo_idx /(demo_idx +1))*data_covar + (demo_idx/(demo_idx+1)**2)*temp*temp.T #eq4

        #transision condition
        #eq6
        cond_pos_mean = np.expand_dims(np.array((data_mean[0], data_mean[1])),1)
        cond_orientation_mean = np.expand_dims(np.array((data_mean[2])),(0,1))
        cond_velocity_mean = np.expand_dims(np.array((data_mean[3], data_mean[4])),1)

        cond_pp = np.copy(data_covar[0:2, 0:2])
        cond_po = np.copy(data_covar[0:2, 2:3])
        cond_pv = np.copy(data_covar[0:2, 3:5])
        cond_oo = np.copy(data_covar[2:3, 2:3])
        cond_ov = np.copy(data_covar[2:3, 3:5])
        cond_op = np.copy(data_covar[2:3, 0:2])
        cond_vv = np.copy(data_covar[3:5, 3:5])
        cond_vo = np.copy(data_covar[3:5, 2:3])
        cond_vp = np.copy(data_covar[3:5, 0:2])

        #eq7
        try:
            cond_new_pos_mean = cond_pos_mean+ np.matmul (np.matmul(np.concatenate((cond_po, cond_pv),1), np.linalg.inv(data_covar[2:5, 2:5])), temp[2:5, 0:1])
        except:
            cond_new_pos_mean = cond_pos_mean + np.matmul(np.matmul(np.concatenate((cond_po, cond_pv), 1), np.linalg.pinv(data_covar[2:5, 2:5])), temp[2:5, 0:1])
        #eq8
        try:
            cond_pos_var = cond_pp - np.matmul (np.matmul(np.concatenate((cond_po, cond_pv),1), np.linalg.inv(data_covar[2:5, 2:5])), data_covar[2:5, 0:2])
        except:
            cond_pos_var = cond_pp - np.matmul(np.matmul(np.concatenate((cond_po, cond_pv), 1), np.linalg.pinv(data_covar[2:5, 2:5])), data_covar[2:5, 0:2])

        #orientation
        inv_temp_mat_1 = np.concatenate((cond_pp, cond_pv), 1)
        inv_temp_mat_2 = np.concatenate((cond_vp, cond_vv), 1)
        inv_temp_mat = np.concatenate((inv_temp_mat_1, inv_temp_mat_2), 0)
        mean_data_temp = np.concatenate((temp[0:2, 0:1], temp[3:5, 0:1]), 0)
        cond_new_orientation_mean = cond_orientation_mean + np.matmul(np.matmul(np.concatenate((cond_op, cond_ov),1), np.linalg.pinv(inv_temp_mat)), mean_data_temp)
        cond_orientation_var = cond_oo - np.matmul (np.matmul(np.concatenate((cond_op, cond_ov),1), np.linalg.pinv(inv_temp_mat)), np.concatenate((cond_po, cond_vo), 0))

        #velocity
        inv_temp_mat_1 = np.concatenate((cond_pp, cond_po), 1)
        inv_temp_mat_2 = np.concatenate((cond_op, cond_oo), 1)
        inv_temp_mat = np.concatenate((inv_temp_mat_1, inv_temp_mat_2), 0)
        cond_new_velocity_mean = cond_velocity_mean+ np.matmul (np.matmul(np.concatenate((cond_vp, cond_vo),1), np.linalg.pinv(inv_temp_mat)), temp[0:3, 0:1])
        cond_velocity_var = cond_vv - np.matmul (np.matmul(np.concatenate((cond_vp, cond_vo),1), np.linalg.pinv(inv_temp_mat)), np.concatenate((cond_pv, cond_ov),0))

        #euclidian distance system # eq9 and eq10 np.expand_dims(state-data_mean, 1)
        temp1 = np.expand_dims(state[0:2],1) - cond_new_pos_mean
        d_ep = np.sqrt(np.matmul(temp1.T, temp1))
        #d_ep = np.sqrt(np.matmul(temp1.T, np.matmul(np.linalg.pinv(cond_pos_var),temp1)))
        temp2 = np.expand_dims(state[2:3],1) - cond_new_orientation_mean
        d_mo = np.sqrt(np.matmul(temp2.T, np.matmul(np.linalg.pinv(cond_orientation_var),temp2)))
        #d_mo = np.sqrt(np.matmul(temp2.T, temp2))

        temp3 = np.expand_dims(state[3:5], 1) - cond_new_velocity_mean
        d_ve =  np.sqrt(np.matmul(temp3.T, temp3))

        #algorithm 1
        dp_emax = 5.0
        do_Mmax = np.pi/4
        # dv_emax = 0.1
        # dp_emax = 8.0
        # do_Mmax = np.pi/2

        # initiate segmentation library
        if not (seg_whole):
            if (d_ep>dp_emax and d_mo>do_Mmax):
            #if (d_ep > dp_emax and d_mo > do_Mmax and d_ve >dv_emax):
            #if (d_ep > dp_emax and d_mo > do_Mmax and d_ve >dv_emax):
                cluster_info = {}
                cluster_info["data_range"] =[current_range_start, demo_idx]
                cluster_info["states_info_mean"] = data_mean
                cluster_info["states_info_var"] = data_covar
                if (demo_idx - current_range_start) > 1:
                    seg_range.append(cluster_info)

                #reinitialize mean
                data_covar = np.zeros((5, 5))
                data_mean = state
                current_range_start = demo_idx+1

    if current_range_start< demo_idx:
        cluster_info = {}
        cluster_info["data_range"] = [current_range_start, demo_idx]
        cluster_info["states_info_mean"] = data_mean
        cluster_info["states_info_var"] = data_covar
        if (demo_idx-current_range_start) > 1:
            seg_range.append(cluster_info)
    #seg_range = seg_range[0]
    # check segmentation
    #print(seg_range)
    if plot:
        seg_colors=["b","r","g"]

        if plot_states:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        else:
            fig = plt.figure()

        for seg_idx in range(len(seg_range)):
            # for seg_idx in range(1):
            Trajectory = []
            segment = seg_range[seg_idx]
            data_r = segment["data_range"]
            t_start = data_r[0]
            t_end = data_r[1]

            traj_x = x[t_start:t_end]
            traj_y = y[t_start:t_end]


            if plot_states:
                traj_vx = vx[t_start:t_end]
                traj_vy = vy[t_start:t_end]
                traj_psi = psi[t_start:t_end]
                t = np.linspace(t_start,t_end, len(traj_vx))
                fig.suptitle('Segmented Trajectory and States')
                ax1.plot(t, traj_x, color=seg_colors[seg_idx%3], label="Segmented x"+str(seg_idx), linewidth=2.0)
                ax2.plot(t, traj_y, color=seg_colors[seg_idx%3], label="Segmented y"+str(seg_idx), linewidth=2.0)
                ax3.plot(t, traj_psi, color=seg_colors[seg_idx%3], label="Segmented psi"+str(seg_idx), linewidth=2.0)
                ax4.plot(t, traj_vx, color=seg_colors[seg_idx%3], label="Segmented vx"+str(seg_idx), linewidth=2.0)
                ax5.plot(t, traj_vy, color=seg_colors[seg_idx%3], label="Segmented vy"+str(seg_idx), linewidth=2.0)
            else:
                #fig = plt.figure()
                plt.plot(traj_x, traj_y, ".", color=seg_colors[seg_idx%3], label="Segmented "+str(seg_idx), linewidth=2.0)
                fig.suptitle('Segmented Trajectory')
                plt.xlabel('X')
                plt.ylabel('Y')

                #print("Velocity diff", traj_vx - traj_vy)

        # plt.legend()
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()
        plt.show()

    return seg_range
    #return [seg_range[0]]

def state_seg_v2(x, y, psi, vx, vy, plot = True, seg_whole = True, plot_states= True):
    #state = np.array((x[0], y[0], psi[0], vx[0], vy[0]))

    velocity=[]
    theta_dot = []
    tester_state = []
    trajectory_state = []
    for demo_idx in range(1, len(x)):
        state_p = np.array((x[demo_idx-1], y[demo_idx-1],  psi[demo_idx-1], vx[demo_idx-1], vy[demo_idx-1]))
        state = np.array((x[demo_idx], y[demo_idx],  psi[demo_idx], vx[demo_idx], vy[demo_idx]))
        velocity.append(np.sqrt(state[3]**2+ state[4]**2))
        theta_dot.append(state[2]-state_p[2])
        v = np.sqrt(state[3]**2+ state[4]**2)
        t = state[2]-state_p[2]
        #tester_state_d = np.array(np.sqrt(state[3]**2+ state[4]**2), state[2]-state_p[2])
        #tester_state.append([v, t])
        tester_state.append(state)
        #trajectory_state.append([state[0], state[1], v, t])
        trajectory_state.append(state)
    velocity = np.array(velocity)
    theta_dot = np.array(theta_dot)
    # plt.figure()
    # plt.plot(velocity, theta_dot)
    # #plt.plot(theta_dot)
    # plt.show()
    # fig, ax1 = plt.subplots()
    # ax1.plot(velocity,color="r", label="Velocity ", linewidth=2.0)
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.plot(theta_dot,color="b", label="Velocity ", linewidth=2.0)
    # plt.show()

    tester_state = np.array(tester_state)
    return tester_state, trajectory_state
    # gmm = GaussianMixture(n_components=3, reg_covar=1, covariance_type="full", verbose=1)  # testing with 10, 5
    # gmm.fit(tester_state)
    # clusters = gmm.predict(tester_state)
    # print(np.unique(clusters))

def get_GausRBF_alpha(z_t,mean_k,sigma,D, alpha):
    return np.exp((-0.5*(alpha*z_t-mean_k)*(alpha* z_t-mean_k))*(1./sigma)) / (np.sqrt(np.power(2*np.pi,D) * sigma))

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
        plt.plot(obs_traj[0],obs_traj[1],'r', label='Observation')
        plt.plot(demonstrated_traj[0], demonstrated_traj[1],'b', label= 'Demonstrated')
        plt.legend()
        plt.show()

    return phase_completed, demonstration_time

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

def generate_psi_dt_ts_woalpha(T):
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
            b_t_x = get_GausRBF(dt[ii], dk[kk], primitive_variance, D)
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



def reconstruct_seg_part_obv(observation_data, mean_ww, sigma_ww, big_psi, prior_seg=[] ):

    span = D
    len_observation = int(len(observation_data)/span)
    #psi_obv_s = np.copy(big_psi[0:D])
    psi_obv = np.copy(big_psi[0:len_observation*D])
    #plt.plot(observation_data[0:len_observation*D: D], observation_data[1:len_observation*D: D])
    #plt.show()
    k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    mean_ww = mean_ww + np.dot(k_seg, (observation_data - np.dot(psi_obv, mean_ww)))
    sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))

    inferred_traj = np.dot(big_psi, mean_ww)

    return inferred_traj, mean_ww, sigma_ww

def reconstruct_seg_sg(observation_data, mean_ww, sigma_ww, big_psi, plot = True, prior_seg=[] ):
    span = D
    len_observation = int(len(observation_data)/span)

    observation_data = np.array(observation_data)
    # observation_x = observation_data[0:T*D:D]
    # observation_y = observation_data[1:T*D:D]
    #print(observation_data.shape)
    start_point = observation_data[0:D]
    traj_before = np.dot(big_psi, mean_ww)
    goal_point= traj_before[-D:]
    psi_obv = np.copy(big_psi[-D:])
    psi_obv_g = np.copy(big_psi[-D:])
    psi_obv_s = np.copy(big_psi[0:D])
    psi_obv_sg = np.vstack((psi_obv_s, psi_obv_g))
    obj_traj =  np.hstack((start_point, goal_point))
    #print(goal_point.shape)
    #print(psi_obv.shape)

    # k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    # mean_ww = mean_ww + np.dot(k_seg, (goal_point - np.dot(psi_obv, mean_ww)))
    # sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
    k_seg = np.dot(np.dot(sigma_ww, psi_obv_sg.T), np.linalg.inv(np.dot(np.dot(psi_obv_sg, sigma_ww), psi_obv_sg.T)))
    mean_ww = mean_ww + np.dot(k_seg, (obj_traj - np.dot(psi_obv_sg, mean_ww)))
    sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv_sg, sigma_ww))
    #trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_pos_ww),big_psi.T)).diagonal())* 1
    inferred_traj = np.dot(big_psi, mean_ww) #+ trained_traj_upper
    len_inferred = int(len(inferred_traj) / span)

    if plot:
        plt.plot(observation_data[0:len_observation*span:span], observation_data[1:len_observation*span:span], 'rx', label="Observation")
        plt.plot(inferred_traj[0:len_inferred*span:span], inferred_traj[1:len_inferred*span:span], 'b.', label="Inferred")
        plt.legend()
        plt.show()

    return inferred_traj, mean_ww, sigma_ww

def reconstruct_seg_msg(observation_data, mean_ww, sigma_ww, big_psi, obs_len,time_prev, plot = True, prior_seg=[] ): # data_last_times, np.copy(mean_weight), np.copy(sigma_weight), phi_data,obs_len,time_prev plot=True)
    span = D


    observation_data = np.array(observation_data)
    #len_observation = int(len(observation_data)/span)
    # observation_x = observation_data[0:T*D:D]
    # observation_y = observation_data[1:T*D:D]
    #print(observation_data.shape)
    start_point = observation_data[0:D]
    traj_before = np.dot(big_psi, mean_ww)
    goal_point= traj_before[-D:]
    psi_obv = np.copy(big_psi[-D:])
    psi_obv_g = np.copy(big_psi[-D:])
    psi_obv_s = np.copy(big_psi[time_prev*D:time_prev*D+D]) # in between
    psi_obv_sg = np.vstack((psi_obv_s, psi_obv_g))
    obj_traj =  np.hstack((start_point, goal_point))
    #print(goal_point.shape)
    #print(psi_obv.shape)

    # k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    # mean_ww = mean_ww + np.dot(k_seg, (goal_point - np.dot(psi_obv, mean_ww)))
    # sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
    k_seg = np.dot(np.dot(sigma_ww, psi_obv_sg.T), np.linalg.inv(np.dot(np.dot(psi_obv_sg, sigma_ww), psi_obv_sg.T)))
    mean_ww = mean_ww + np.dot(k_seg, (obj_traj - np.dot(psi_obv_sg, mean_ww)))
    sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv_sg, sigma_ww))
    #trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_pos_ww),big_psi.T)).diagonal())* 1
    inferred_traj = np.dot(big_psi, mean_ww) #+ trained_traj_upper
    len_inferred = int(len(inferred_traj) / span)

    if plot:
        plt.plot(observation_data[0:obs_len*span:span], observation_data[1:obs_len*span:span], 'rx', label="Observation")
        plt.plot(inferred_traj[0:len_inferred*span:span], inferred_traj[1:len_inferred*span:span], 'b.', label="Inferred")
        plt.legend()
        plt.show()

    return inferred_traj, mean_ww, sigma_ww

def reconstruct_seg_mmg(observation_data, mean_ww, sigma_ww, big_psi, obs_len,time_now, plot = True, prior_seg=[] ): # data_last_times, np.copy(mean_weight), np.copy(sigma_weight), phi_data,obs_len,time_prev plot=True)
    span = D


    observation_data = np.array(observation_data)
    #len_observation = int(len(observation_data)/span)
    # observation_x = observation_data[0:T*D:D]
    # observation_y = observation_data[1:T*D:D]
    #print(observation_data.shape)
    start_point = observation_data[-D:]
    traj_before = np.dot(big_psi, mean_ww)
    goal_point= traj_before[-D:]
    psi_obv = np.copy(big_psi[-D:])
    psi_obv_g = np.copy(big_psi[-D:])
    psi_obv_s = np.copy(big_psi[time_now*D:time_now*D+D]) # in between
    psi_obv_sg = np.vstack((psi_obv_s, psi_obv_g))
    obj_traj =  np.hstack((start_point, goal_point))
    #print(goal_point.shape)
    #print(psi_obv.shape)

    # k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    # mean_ww = mean_ww + np.dot(k_seg, (goal_point - np.dot(psi_obv, mean_ww)))
    # sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
    k_seg = np.dot(np.dot(sigma_ww, psi_obv_sg.T), np.linalg.inv(np.dot(np.dot(psi_obv_sg, sigma_ww), psi_obv_sg.T)))
    mean_ww = mean_ww + np.dot(k_seg, (obj_traj - np.dot(psi_obv_sg, mean_ww)))
    sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv_sg, sigma_ww))
    #trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_pos_ww),big_psi.T)).diagonal())* 1
    inferred_traj = np.dot(big_psi, mean_ww) #+ trained_traj_upper
    len_inferred = int(len(inferred_traj) / span)

    if plot:
        plt.plot(observation_data[0:obs_len*span:span], observation_data[1:obs_len*span:span], 'rx', label="Observation")
        plt.plot(inferred_traj[0:len_inferred*span:span], inferred_traj[1:len_inferred*span:span], 'b.', label="Inferred")
        plt.legend()
        plt.show()

    return inferred_traj, mean_ww, sigma_ww

def reconstruct_seg_og(observation_data, mean_ww, sigma_ww, big_psi, plot = True, prior_seg=[] ):
    span = D
    len_observation = int(len(observation_data)/span)

    observation_data = np.array(observation_data)
    # observation_x = observation_data[0:T*D:D]
    # observation_y = observation_data[1:T*D:D]
    #print(observation_data.shape)
    start_point = observation_data[0:D]
    mid_point = observation_data[-D:]

    traj_before = np.dot(big_psi, mean_ww)
    goal_point= traj_before[-D:]

    psi_obv_m = np.copy(big_psi[len_observation*D:len_observation*D+D])
    psi_obv_g = np.copy(big_psi[-D:])
    psi_obv_so = np.copy(big_psi[0:D]) #observation data

    psi_obv_sg = np.vstack((psi_obv_so, psi_obv_m, psi_obv_g))
    #obj_traj =  np.hstack((start_point, goal_point))
    obj_traj =  np.hstack((start_point, mid_point, goal_point))
    #print(goal_point.shape)
    #print(psi_obv.shape)

    # k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    # mean_ww = mean_ww + np.dot(k_seg, (goal_point - np.dot(psi_obv, mean_ww)))
    # sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
    k_seg = np.dot(np.dot(sigma_ww, psi_obv_sg.T), np.linalg.inv(np.dot(np.dot(psi_obv_sg, sigma_ww), psi_obv_sg.T)))
    mean_ww = mean_ww + np.dot(k_seg, (obj_traj - np.dot(psi_obv_sg, mean_ww)))
    sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv_sg, sigma_ww))
    #trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_pos_ww),big_psi.T)).diagonal())* 1
    inferred_traj = np.dot(big_psi, mean_ww) #+ trained_traj_upper
    len_inferred = int(len(inferred_traj) / span)

    if plot:
        plt.plot(observation_data[0:len_observation*span:span], observation_data[1:len_observation*span:span], 'rx', label="Observation")
        plt.plot(inferred_traj[0:len_inferred*span:span], inferred_traj[1:len_inferred*span:span], 'b.', label="Inferred")
        plt.legend()
        plt.show()

    return inferred_traj, mean_ww, sigma_ww

def compute_phi_index_obs_mdtw(data, phi ,mean_weight, sigma_weight, data_len, phase_prev, use_KF= True, use_lookahead= True):
    #data is the observation (x,y,vx,vy,psi)
    #segment is original training data (x, y, vx, vy, psi)
    #can be used for the finding out the mean trajectory from the class or from the separate data

    #checking only in one dimension
    obs_len = int(len(data)/D)
    data_x=data[0:obs_len*D:D]
    data_y= data[1:obs_len*D:D]
    data_xy = [data[0:obs_len*D:D], data[1:obs_len*D:D]]
    demonstrated_data = np.dot(phi,mean_weight)
    demonstrated_len = int(len(demonstrated_data)/D)
    demonstrated_data_xy = [demonstrated_data[0:demonstrated_len*D:D], demonstrated_data[1:demonstrated_len*D:D]]
    phase_passed, tf = get_passed_phase(data_xy,demonstrated_data_xy, plot= False)



    print("total time if phase is constant = ", data_len / (phase_passed - phase_prev))
    T = demonstrated_len
    #T = 20
    print("the final time=", tf )
    print("demonstrated data length", demonstrated_len)
    print("observed data length ",obs_len)





    #phi_data, _ = generate_psi_dt_ts(int(tf[0]), alpha)
    tf = data_len / (phase_passed - phase_prev)
    phi_data, _ = generate_psi_dt_ts_woalpha(int(tf[0]))



    inferred_traj = np.dot(phi_data, mean_weight)
    time_prev = int(tf*phase_prev)
    time_now = int(tf*phase_passed)
    check_progress = False
    if check_progress:
        plt.plot(inferred_traj[0:len(inferred_traj): D], inferred_traj[1: len(inferred_traj): D], 'bo',
                 label='inferred all')
        plt.plot(inferred_traj[0:time_prev*D:D], inferred_traj[1:time_prev*D:D], 'rx')
        plt.plot(data_x[-data_len:], data_y[-data_len:], 'g.')
        plt.show()

    data_last_times= data[-data_len*D:]
    #tested with filter
    if use_KF:

        if use_lookahead:
            inferred_traj, _, _ = reconstruct_seg_part_obv(data,mean_weight, sigma_weight, phi_data)

        else:
            #inferred_traj, _, _ = reconstruct_seg_sg(data, np.copy(mean_weight), np.copy(sigma_weight), phi_data, plot=True)
            #inferred_traj, _, _ = reconstruct_seg_og(data,mean_weight, sigma_weight, phi_data, plot=True)

            inferred_traj, _, _ = reconstruct_seg_msg(data_last_times, np.copy(mean_weight), np.copy(sigma_weight), phi_data,obs_len,time_prev, plot=True)
            # inferred_traj, _, _ = reconstruct_seg_mmg(data_last_times, np.copy(mean_weight), np.copy(sigma_weight),
            #                                           phi_data, obs_len, time_now, plot=True)


    #tested with extrapolation


    return phi_data, inferred_traj, phase_passed, time_now




def state_seg_cluster_gmm(gmm, all_states_traj, plot = False):
    #clusters = gmm.predict(all_states_seg)
    #print(np.unique(clusters))
    color_l = ["r", "b", "g", "y", "cyan", "m", "lime"]
    time_stamp_holder = []
    seg_ranges_train= []
    for i in range(len(all_states_traj)):
        current_traj = all_states_traj[i]
        current_timestamp = []

        c1=[]
        c2 =[]
        c3 =[]
        for j in range(len(current_traj)):
            state = current_traj[j]
            state = np.array(state)
            state = np.expand_dims(state, 0)
            #clst = gmm.predict(state[0:1,2:4])
            clst = gmm.predict(state)
            current_timestamp.append(clst)

            if clst == 0:
                c1.extend(state)
            elif clst == 1:
                c2.extend(state)
            elif clst == 2:
                c3.extend(state)
        current_timestamp = np.array(current_timestamp)
        time_stamp_holder.append(current_timestamp)
        current_timestamp = ndimage.median_filter(current_timestamp, size = 5)
        previous_timestamp = current_timestamp[0]
        current_range_start = 0
        seg_ranges= []
        for j in range (len(current_timestamp)):
            if not current_timestamp[j] == previous_timestamp or j == len(current_timestamp)-1:
                cluster_info = {}
                cluster_info["data_range"] = [current_range_start, j-1]
                cluster_info["class"] = previous_timestamp
                previous_timestamp = current_timestamp[j]
                current_range_start = j
                seg_ranges.append(cluster_info)
                #print("cluster_info ", cluster_info["data_range"])

        seg_ranges_train.append(seg_ranges)


        if plot:
            c1 = np.array(c1)
            c2 = np.array(c2)
            c3 = np.array(c3)
            #print(c2.shape)
            #print(c3.shape)
            plt.figure()
            if c1.shape[0]>0:
                plt.plot(c1[:, 0], c1[:, 1], '.', color="r", label= 'Cluster 1')
            if c2.shape[0] > 0:
                plt.plot(c2[:, 0], c2[:, 1],  '.', color="b", label= 'Cluster 2')
            if c3.shape[0] > 0:
                plt.plot(c3[:, 0], c3[:, 1], '.',  color="g", label= 'Cluster 3')
            plt.legend()
            plt.show()

    return seg_ranges_train



def generate_psi_dt():
    dk = np.linspace(0,1,K)
    dt = np.linspace(0,1,T)
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
            b_t_x = get_GausRBF(dt[ii], dk[kk], primitive_variance, D)
            # b_t_y = get_GausRBF(z_y[ii], primitive_mean, primitive_variance, D)
            # b_t_o = get_GausRBF(z_psi[ii], primitive_mean, primitive_variance, D)
            # b_t_vx = get_GausRBF(z_vx[ii], primitive_mean, primitive_variance, D)
            # b_t_vy = get_GausRBF(z_vy[ii], primitive_mean, primitive_variance, D)
            small_psi = np.identity(D)
            # np.fill_diagonal(small_psi, [b_t_x, b_t_y, b_t_o, b_t_vx, b_t_vy])
            np.fill_diagonal(small_psi, [b_t_x, b_t_x, b_t_x, b_t_x, b_t_x])
            # np.fill_diagonal(small_psi, [b_t_x])
            big_psi[ii * D:(ii + 1) * D, kk * D:(kk + 1) * D] = np.copy(small_psi)
    print("Big Psi shape: ", big_psi.shape)
    return big_psi, dt


def weight_gen(seg_range,x,y,vx,vy,psi,big_psi, dt, plot = False, calc_accuracy = False, plot_states= True):
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


def get_observation_for_vehicle(vehicle_no, T):
    x, y, vx, vy, psi= make_demonstrations(vehicle_no)
    Trajectory = []
    for i in range(0, len(x), 50):
        seg_x = x[i:(i + 50)]
        print(len(seg_x))
        seg_y = y[i:(i + 50)]
        seg_vx = vx[i:(i + 50)]
        seg_vy = vy[i:(i + 50)]
        seg_psi = psi[i:(i + 50)]

        #look ahead ateps of 50
        t_steps = np.linspace(0, 1, len(seg_x))
        # T = 8

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
    return Trajectory


def get_observation_for_vehicle_whole(vehicle_no, T):
    x, y, vx, vy, psi= make_demonstrations(vehicle_no)
    Trajectory = []
    seg_x = x
    print(len(seg_x))
    seg_y = y
    seg_vx = vx
    seg_vy = vy
    seg_psi = psi

    #look ahead ateps of 50
    t_steps = np.linspace(0, 1, len(seg_x))
    # T = 8

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
    return Trajectory


def get_observation_seg_for_testing(x, y, vx, vy, psi, T, interval):
    Trajectory = []
    for i in range(0, len(x), interval):
        seg_x = x[i:(i + interval)]
        #print(len(seg_x))
        seg_y = y[i:(i + interval)]
        seg_vx = vx[i:(i + interval)]
        seg_vy = vy[i:(i + interval)]
        seg_psi = psi[i:(i + interval)]

        #look ahead ateps of 50
        t_steps = np.linspace(0, 1, len(seg_x))
        # T = 8

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
    return Trajectory

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



######################### new segment reconstruction policy ##############
def reconstruct_seg_goal(observation_data, mean_ww, sigma_ww, prior_seg=[]):
    # each segment into T steps

    observation_data = np.array(observation_data)
    # observation_x = observation_data[0:T*D:D]
    # observation_y = observation_data[1:T*D:D]
    #print(observation_data.shape)
    start_point = observation_data[0:D]
    goal_point= observation_data[-D:]
    psi_obv = np.copy(big_psi[-D:])
    psi_obv_g = np.copy(big_psi[-D:])
    psi_obv_s = np.copy(big_psi[0:D])
    psi_obv_sg = np.vstack((psi_obv_s, psi_obv_g))
    obj_traj =  np.hstack((start_point, goal_point))
    #print(goal_point.shape)
    #print(psi_obv.shape)

    # k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
    # mean_ww = mean_ww + np.dot(k_seg, (goal_point - np.dot(psi_obv, mean_ww)))
    # sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
    k_seg = np.dot(np.dot(sigma_ww, psi_obv_sg.T), np.linalg.inv(np.dot(np.dot(psi_obv_sg, sigma_ww), psi_obv_sg.T)))
    mean_ww = mean_ww + np.dot(k_seg, (obj_traj - np.dot(psi_obv_sg, mean_ww)))
    sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv_sg, sigma_ww))
    #trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_pos_ww),big_psi.T)).diagonal())* 1
    inferred_traj = np.dot(big_psi, mean_ww) #+ trained_traj_upper

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


def compute_class(data, all_phi, mean_weights, sigma_weights):
    #data is the observation (x,y,vx,vy,psi)
    #segment is original training data (x, y, vx, vy, psi)
    #can be used for the finding out the mean trajectory from the class or from the separate data
    obs_len = 20#50#100
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
    return mean_weights[index], sigma_weights[index], index

# ###################new reconstruction with the KF points ##############
# inferred_test_traj = []
# def_mean_pos = mean_pos_ww.copy()
# def_pos_sigma = sigma_pos_ww.copy()
# for i in range (len(seg_ranges_obv)):
#     current_test_traj= seg_ranges_obv[i]
#     current_inferred_traj = []
#     #plt.figure(10)
#
#     mean_pos_ww = def_mean_pos
#     sigma_pos_ww = def_pos_sigma
#     for j in range (len (current_test_traj)):
#         current_segment = current_test_traj[j]
#
#         # #check is positive or negative class
#         if current_segment[-D]>0 :
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




if __name__ == "__main__":
    # pulls one state vector
    # x, y, vx, vy, psi= make_demonstrations(17)
    all_x, all_y, all_vx, all_vy, all_psi = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=100)
    active_x, active_y, active_vx, active_vy, active_psi, active_int, active_b, active_e =  extract_dataset_traj_active("Scenario4", False, [3], [5], data_lim=100, track_id_list = [37, 47, 77]) # [37,47,77]
    active_2x, active_2y, active_2vx, active_2vy, active_2psi, active_2int, active_2b, active_2e =  extract_dataset_traj_active("Scenario4", False, [3], [5], data_lim=100, track_id_list = [41, 46, 60]) # [41,46,60]
    # all_x, all_y, all_vx, all_vy, all_psi= generate_demo_traj()
    indices = np.arange(len(all_x))
    np.random.shuffle(indices)
    train_x, train_y, train_vx, train_vy, train_psi = [], [], [], [], []
    test_x, test_y, test_vx, test_vy, test_psi = [], [], [], [], []
    for i in range(len(indices)):

        train_x.append(all_x[indices[i]])
        train_y.append(all_y[indices[i]])
        train_vx.append(all_vx[indices[i]])
        train_vy.append(all_vy[indices[i]])
        train_psi.append(all_psi[indices[i]])



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

    ###############################################################

    ###########################################################original segmentation block#############################################
    # seg_ranges = []
    # for i in range(len(train_x)):
    #     #print(train_vx[i]-train_vy[i])
    #     seg_ranges.append(state_segmentation(train_x[i], train_y[i], train_psi[i], train_vx[i], train_vy[i], plot=True, seg_whole= False, plot_states= True))

    ###############################################################
    # algorithm 2
    big_psi, dt = generate_psi_dt()
    # clustering
    list_pos_ww = []
    list_neg_ww = []

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

        # if train_x[i][-1] > 0:
        #     list_pos_ww.extend(list_ww) # list extended
        #     #list_pos_ww.append(np.ones((400))+j)
        #     #j+= 1
        # else:
        #     list_neg_ww.extend(list_ww)

    # print(ww.shape)
    # list_psi = np.array(list_psi)
    # list_traj = np.array(list_traj)
    # list_pos_ww = np.array(list_pos_ww) # make an array of the list
    # print(list_pos_ww.shape)
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

    # sigma_pos_ww_t = list_pos_ww.var(0)
    # sigma_pos_ww = np.identity(sigma_pos_ww_t.shape[0])

    #     tempmean = np.expand_dims(list_pos_ww[demo_idx]-mean_pos_ww,1)
    #     temp_var = np.dot(tempmean,tempmean.T)
    #     weights_var += temp_var
    #
    # weights_var= weights_var/ (list_pos_ww.shape[0]-1)
    # weights_var_diag = np.diagonal(weights_var)
    # trained_traj_upper = (np.dot(big_psi,  np.sqrt(weights_var_diag))* 1.96)+ trained_traj

    plot_wts_mn_cv = False
    if plot_wts_mn_cv:
        print("Variance shape: ", sigma_pos_ww.shape)
        # trained_traj_upper = (np.dot(big_psi,  mean_pos_ww + np.sqrt(sigma_pos_ww.diagonal()))* 1)#+ trained_traj
        trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_pos_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
        # plt.figure()
        # plt.plot(mean_pos_ww, label="mean")
        # #plt.plot(mean_pos_ww+ np.sqrt(sigma_pos_ww.diagonal()), label="mean+sd")
        # plt.plot( np.sqrt(sigma_pos_ww.diagonal()),  label="sd" )
        # plt.legend()
        # plt.show()

        # trained_traj_lower = np.dot(big_psi, lower_mean)

        plt.figure()
        color_l = ["r", "b", "g", "y", "cyan", "m", "lime"]
        for j in range(len(list_pos_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_pos_ww[j])
            color = color_l[clusters[j]]
            # plt.plot(traj_all[0:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            plt.plot(traj_all[0:T * D:D], "--", color=color, linewidth=2.0, alpha=0.4)
            # plt.show()
        # plt.plot(trained_traj[0:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
        # plt.plot(np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj[0:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
        # plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj[0:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of X state segments")
        plt.show()
        plt.figure()
        for j in range(len(list_pos_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_pos_ww[j])
            plt.plot(traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj[1:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj[1:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj[1:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Y state segments")
        plt.show()
        plt.figure()
        for j in range(len(list_pos_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_pos_ww[j])
            plt.plot(traj_all[2:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj[2:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[2:T * D:D]) + trained_traj[2:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[2:T * D:D]) + trained_traj[2:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Angle state segments")
        plt.show()
        plt.figure()
        for j in range(len(list_pos_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_pos_ww[j])
            plt.plot(traj_all[3:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj[3:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[3:T * D:D]) + trained_traj[3:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[3:T * D:D]) + trained_traj[3:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of VX state segments")
        plt.show()
        plt.figure()
        for j in range(len(list_pos_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_pos_ww[j])
            plt.plot(traj_all[4:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj[4:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[4:T * D:D]) + trained_traj[4:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[4:T * D:D]) + trained_traj[4:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of VY state segments")
        plt.show()

        plt.figure()
        for j in range(len(list_pos_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_pos_ww[j])
            plt.plot(traj_all[0:T * D:D], traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj[0:T * D:D], trained_traj[1:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj[0:T * D:D],
                 np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj[1:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj[0:T * D:D],
                 -np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj[1:T * D:D], color="lime", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.show()

        # xe = np.linspace(0,1,20)
        # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        # fig.suptitle('Mean and Variance of the Primtives for Each State')
        # ax1.plot(trained_traj[0:T * D:D],  color="b", label="Primtive x", linewidth=2.0)
        # #ax1.errorbar(xe, trained_traj[0:T * D:D], yerr= yerr1,  color="b", label="Primtive x", linewidth=2.0)
        # ax2.plot(trained_traj[1:T * D:D], color="b", label="Primtive y", linewidth=2.0)
        # ax3.plot(trained_traj[2:T * D:D], color="b", label="Primitive psi", linewidth=2.0)
        # ax4.plot(trained_traj[3:T * D:D], color="b", label="Primitive vx", linewidth=2.0)
        # ax5.plot(trained_traj[4:T * D:D], color="b", label="Primitive vy", linewidth=2.0)
        #
        # #ax1.plot( trained_traj_upper[0:T * D:D],  ".", color="r", label="Primitive var x upper", linewidth=2.0)
        # # ax1.plot( trained_traj_lower[0:T * D:D], ".", color="r", label="Primitive var x lower", linewidth=2.0)
        # # ax2.plot( trained_traj_upper[1:T * D:D],  ".", color="r", label="Primitive var y upper", linewidth=2.0)
        # # ax2.plot( trained_traj_lower[1:T * D:D], ".", color="r", label="Primitive var y lower", linewidth=2.0)
        # # ax3.plot( trained_traj_upper[2:T * D:D],  ".", color="r", label="Primitive var theta upper", linewidth=2.0)
        # # ax3.plot( trained_traj_lower[2:T * D:D], ".", color="r", label="Primitive var theta lower", linewidth=2.0)
        # # ax4.plot( trained_traj_upper[3:T * D:D],  ".", color="r", label="Primitive vx var upper", linewidth=2.0)
        # # ax4.plot( trained_traj_lower[3:T * D:D], ".", color="r", label="Primitive vx var lower", linewidth=2.0)
        # # ax5.plot( trained_traj_upper[4:T * D:D],  ".", color="r", label="Primitive vy var upper", linewidth=2.0)
        # # ax5.plot( trained_traj_lower[4:T * D:D], ".", color="r", label="Primitive vy var lower", linewidth=2.0)
        #
        # ax1.legend()
        # ax2.legend()
        # ax3.legend()
        # ax4.legend()
        # ax5.legend()
        # plt.show()

        # --- COMMENTED OUT NEGATIVE CLUSTERS

    plot_wts_mn_cluster = False
    if plot_wts_mn_cluster:
        trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_ent_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_ent_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_ent_ww[j])
            plt.plot(traj_all[0:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_ent[0:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_ent[0:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_ent[0:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 0")
        plt.show()
        plt.figure()

        trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_ent_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_ent_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_ent_ww[j])
            plt.plot(traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_ent[1:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_ent[1:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_ent[1:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 0")
        plt.show()
        plt.figure()

        trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_mid_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_mid_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_mid_ww[j])
            plt.plot(traj_all[0:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_mid[0:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_mid[0:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_mid[0:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 1")
        plt.show()
        plt.figure()
        trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_mid_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_mid_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_mid_ww[j])
            plt.plot(traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_mid[1:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_mid[1:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_mid[1:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 1")
        plt.show()
        plt.figure()
        trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_exit_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_exit_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_exit_ww[j])
            plt.plot(traj_all[0:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_exit[0:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_exit[0:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D]) + trained_traj_exit[0:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 2")
        plt.show()
        plt.figure()
        trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_exit_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
        for j in range(len(list_exit_ww)):
            # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
            traj_all = np.dot(big_psi, list_exit_ww[j])
            plt.plot(traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
            # plt.show()
        plt.plot(trained_traj_exit[1:T * D:D], color="b", label="Primtive ", linewidth=2.0)
        plt.plot(np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_exit[1:T * D:D], color="g", label="Upper SD ",
                 linewidth=2.0)
        plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D]) + trained_traj_exit[1:T * D:D], color="m", label="Lower SD ",
                 linewidth=2.0)
        plt.legend()
        plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 2")
        plt.show()
        plt.figure()

    ##### for two cars, just find the start of the ego vehicle
    base_timestamp1 = 0
    base_timestamp2 = 0

    #checking only for ego car
    ptime_start = active_b[0]
    active_end = active_e[0]

    # primitive second car
    # primirive first car
    timesteps_car1 = list(np.arange(active_b[0], active_e[0], 100))
    timesteps_car2 = list(np.arange(active_2b[0], active_2e[0], 100))
    # print(timesteps_car1)

    game_frenet = False
    calc_frenet = False

    car1_states_inferred_x = []
    car2_states_inferred_x = []
    car1_states_inferred_y = []
    car2_states_inferred_y = []
    obs_traj = []
    traj_p_10 = []
    first_time_index = -1
    prevlookahead = 0
    prevlookaheadseg = 0
    phase_prev = 0.0
    prev_index = 1000 #large number
    while ptime_start < active_end:  # primitive end time
        hascar1 = False  # ego car
        hascar2 = False
        in_game = False

        if ptime_start in timesteps_car1:
            time_index1 = timesteps_car1.index(ptime_start)
            position_1 = (active_x[0][time_index1] ** 2 + active_y[0][time_index1] ** 2) ** (1 / 2)
            velocity_1 = (active_vx[0][time_index1] ** 2 + active_vy[0][time_index1] ** 2) ** (1 / 2)
            hascar1 = True

        if ptime_start in timesteps_car2:
            time_index2 = timesteps_car2.index(ptime_start)
            position_2 = (active_2x[0][time_index2] ** 2 + active_2y[0][time_index2] ** 2) ** (1 / 2)
            velocity_2 = (active_2vx[0][time_index2] ** 2 + active_2vy[0][time_index2] ** 2) ** (1 / 2)
            hascar2 = True

        # print(hascar1, " has car ",  hascar2)

        nodes_removed = 0
        # when both car are present #TODO: but have to start primitive when one car enters the intersections

        if hascar1 and hascar2:
            car_length = 5.0  # we chose maximum
            TTC = np.abs((position_2 - position_1 - car_length) / (velocity_1 - velocity_2))
            dist = abs(position_2 - position_1)

            if dist< 5:

                if first_time_index< 0:
                    first_time_index= time_index1
                    first_time_index2= time_index2
                horizon = 3  # time horizon
                #total_prediction = total_prediction -1
                ts = 0.1  # 100 ms
                #game

                in_game = True

                action_set = [[-0.1, -1], [0, -1], [0.1, -1], [-0.1, 0], [0, 0], [0.1, 0], [-0.1, 1], [0, 1],
                              [0.1, 1]]

                car_state1 = [active_x[0][time_index1], active_y[0][time_index1], active_vx[0][time_index1],
                              active_vy[0][time_index1], active_psi[0][time_index1]]
                car_state2 = [active_2x[0][time_index2], active_2y[0][time_index2], active_2vx[0][time_index2],
                              active_2vy[0][time_index2], active_2psi[0][time_index2]]
                # primitive_index = int()

                primitive1 = [traj_p_10[primitive_index*D],traj_p_10[(primitive_index*D)+1]]

                treedict = {}  # empty dictionary
                root_node = data_node("root")
                root_node.isroot = True
                root_node.level = 0
                root_node.previous_states = [car_state1, car_state2]
                root_node.current_states = [car_state1, car_state2]
                root_node.parent = "root"
                root_node.cost = [0]
                treedict["root"] = root_node
                # children node initialized for the tree

                # build the cost matrix
                haschildren = True
                current_nodes_list = [root_node]
                min_cost1 = 100000
                best_node = None

                while haschildren:
                    if len(current_nodes_list) == 0:
                        # Done for all children
                        haschildren = False
                        continue

                    current_node = current_nodes_list[-1]
                    current_children_list = current_node.children
                    level = current_node.level
                    if current_node.finish_count >= len(action_set):
                        # This child branch is done for level below it, update parent level and remove it
                        treedict[current_node.parent].finish_count += 1
                        if current_node.level <horizon:
                            nodes_removed = nodes_removed+ 1
                            print("Remove node at level: ", current_node.level)
                            #print("Total nodes removed", nodes_removed)
                        current_nodes_list = current_nodes_list[:-1]
                        continue

                    if level < horizon:
                        #building the tree
                        if len (current_children_list) == 0:
                            for i in range(len(action_set)):
                                name = str(level + 1) + "_" + str(i)
                                child_node = data_node(name)
                                child_node.level = level + 1
                                child_node.current_action = [action_set[i]]
                                child_node.parent = current_node.name
                                treedict[name] = child_node
                                current_node.children.append(child_node)

                        for child_index in range(len(current_children_list)):
                            child_node= current_children_list[child_index]
                            child_node.previous_states = treedict[child_node.parent].current_states
                            car_state1 = child_node.previous_states[0]
                            car_state2 = child_node.previous_states[1]

                            pred_car_1 = dynamics(car_state1,child_node.current_action[0])
                            pred_car_2 = constant_velocity_model(car_state2)
                            child_node.current_states = [pred_car_1, pred_car_2]
                            car_distance = ((pred_car_1[0] - pred_car_2[0]) ** 2 + (
                                        pred_car_1[1] - pred_car_2[1]) ** 2) ** (1 / 2)
                            # self distance from primitive
                            primitive1 = [traj_p_10[(primitive_index+child_node.level) * D], traj_p_10[((primitive_index+child_node.level) * D) + 1]]
                            change_pred1 = np.sqrt(((pred_car_1[0] - primitive1[0]) ** 2 + (pred_car_1[1] - primitive1[1])**2))
                            cost1 = (change_pred1 - 1 / car_distance)
                            parent_cost = treedict[current_node.parent].cost

                            child_node.cost = [parent_cost[0] + cost1]

                            if level< (horizon-1):
                                current_nodes_list.append(child_node)
                            else:
                                #print(child_node.cost)
                                if child_node.cost[0] < min_cost1:
                                    best_node = child_node
                                    min_cost1 =  child_node.cost
                        if level == (horizon-1):
                            treedict[current_node.parent].finish_count += 1
                            current_nodes_list = current_nodes_list[:-1]
                while True:
                    print("Level ", best_node.level)
                    print("State ", best_node.current_states)
                    print("Cost: ", best_node.cost)
                    print("Actions: ", best_node.current_action)
                    print("-" * 20)
                    # car_states_inferred.append(best_node.current_states)
                    car1_states_inferred_x.append(best_node.current_states[0][0])
                    car2_states_inferred_x.append(best_node.current_states[1][0])
                    car1_states_inferred_y.append(best_node.current_states[0][1])
                    car2_states_inferred_y.append(best_node.current_states[1][1])
                    if best_node.name == "root":
                        break
                    best_node = treedict[best_node.parent]
                # break
                # print("car states inferred 0", car_states_inferred[0])
                # print("car states inferred 1", car_states_inferred[1])
                print("car states inferred 0", car1_states_inferred_x, " ", car1_states_inferred_y)
                print("car states inferred 1", car2_states_inferred_x, " ", car2_states_inferred_y)

                plt.figure()
                plt.plot(car1_states_inferred_x, car1_states_inferred_y, 'rx',
                         label='GT inferred ego vehicle trajectory')

                # plt.plot(active_x[0][time_index1 : time_index1+ horizon], active_y[0][ time_index1 : time_index1+ horizon], 'g')
                # plt.plot(active_2x[0][time_index2 : time_index2+ horizon], active_2y[0][time_index2 : time_index2+ horizon],'y')

                plt.plot(active_x[0][0: time_index1 + horizon],
                         active_y[0][0: time_index1 + horizon], 'g', label= 'ego vehicle')
                plt.plot(active_2x[0][0 : time_index2+ horizon], active_2y[0][0: time_index2+ horizon],'y', label= 'other vehicle')
                plt.plot(traj_p_10[0:(primitive_index + 3) * 5:5], traj_p_10[1:(primitive_index + 3) * 5:5], 'b.',
                         label='primitive segment')
                # plt.plot(active_x[0][first_time_index : time_index1+ 3], active_y[0][first_time_index: time_index1+ horizon], 'g.',  label='observed ego interacting vehicle')
                plt.plot(active_x[0][time_index1: time_index1 + 4],
                         active_y[0][time_index1: time_index1 + 4], 'g.',
                         label='observed ego interacting vehicle')
                plt.plot(active_2x[0][first_time_index2 : time_index2+ 3], active_2y[0][first_time_index2: time_index2+ horizon], 'y.')
                plt.plot(car2_states_inferred_x, car2_states_inferred_y, 'm.')
                plt.xlabel('distance (m)')
                plt.ylabel('distance (m)')
                plt.legend()
                plt.show()
                GT_pm_accuracy = True
                if GT_pm_accuracy:
                    diff_x = Reverse(car1_states_inferred_x[-4:]) - active_x[0][time_index1: time_index1 + 4]
                    diff_y = Reverse(car1_states_inferred_y[-4:]) - active_y[0][time_index1: time_index1 + 4]

                    RMSE_x = np.sqrt(sum(diff_x ** (2)) / 4)
                    RMSE_y = np.sqrt(sum(diff_y ** (2)) / 4)
                    print(" RMSE accuracy", RMSE_x, RMSE_y)

                # exit()

                ptime_start = ptime_start + 300  # i #
                primitive_index = primitive_index+ horizon

                #with all possible action set find the minimum action set when you regard other vehicle as obstacle



                #have the future trajectories in traj_p#

                print("game")
            else:
                in_game = False

        if not in_game:
            primitive_index = 0
            #obs_traj = []
            obs_data_len = 10

            for j in range(prevlookahead, (prevlookahead+obs_data_len)):
                obs_traj.append(active_x[0][j])
                obs_traj.append(active_y[0][j])
                obs_traj.append(active_psi[0][j])
                obs_traj.append(active_vx[0][j])
                obs_traj.append(active_vy[0][j])

            prevlookahead = prevlookahead+ obs_data_len
            prevlookaheadseg= prevlookaheadseg+obs_data_len

            obs_traj_a = np.array(obs_traj)
            observation_x = obs_traj_a[0:obs_traj_a.shape[0]:D]
            observation_y = obs_traj_a[1:obs_traj_a.shape[0]:D]
            #compute ent and sigma
            seg_mean_ww, seg_sigma_ww, index =  compute_class(obs_traj_a, big_psi, mean_weights_data, sigma_weights_data)

            print("data chosen=", index)

            if prev_index != index:
                phase_prev = 0.0

            prev_index = index

            _, inferred_traj, phase, time_now = compute_phi_index_obs_mdtw(obs_traj_a, big_psi, seg_mean_ww, seg_sigma_ww, obs_data_len, phase_prev, use_KF=True, use_lookahead=False)
            phase_prev = phase

            traj_p_10 = inferred_traj[time_now * D: (time_now+ obs_data_len)*D]
            #traj_p_10 = inferred_traj[(prevlookaheadseg-obs_data_len) * D: prevlookaheadseg * D]

            plt.plot(inferred_traj[0:len(inferred_traj): D], inferred_traj[1: len(inferred_traj): D], 'bo',
                     label='inferred all')
            plt.plot(inferred_traj[0:prevlookaheadseg * D: D], inferred_traj[1:prevlookaheadseg * D: D], 'go',
                     label='inferred obs')
            plt.plot(active_x[0], active_y[0], 'g.')
            plt.plot(observation_x, observation_y, 'rx')
            plt.legend()
            plt.show()

            future_pred= True

            if future_pred:
                plt.plot(traj_p_10[0:len(traj_p_10): D], traj_p_10[1: len(traj_p_10): D], 'ro',
                         label='inferred next horizon')
                plt.plot(active_x[0][prevlookahead:prevlookahead+ obs_data_len], active_y[0][prevlookahead:prevlookahead+ obs_data_len], 'bx', label = "future traj")
                plt.xlabel("distance (m)")
                plt.ylabel ("distance (m)")

                future_pred_accuracy = True
                if future_pred_accuracy:
                    if len(traj_p_10[0:len(traj_p_10): D]) == len(active_x[0][prevlookahead:prevlookahead+ obs_data_len]):
                        diff_x = traj_p_10[0:len(traj_p_10): D] - active_x[0][prevlookahead:prevlookahead+ obs_data_len]
                        diff_y = traj_p_10[1: len(traj_p_10): D] - active_y[0][prevlookahead:prevlookahead+ obs_data_len]

                        RMSE_x = np.sqrt(sum(diff_x ** (2)) / 4)
                        RMSE_y = np.sqrt(sum(diff_y ** (2)) / 4)
                        print(" RMSE accuracy", RMSE_x, RMSE_y)

                plt.legend()
                plt.show()



            if phase_prev == 1.0:
                prev_index= 1000 #large number
                obs_traj = []
                prevlookaheadseg = 0
            ptime_start = ptime_start + 100*obs_data_len


