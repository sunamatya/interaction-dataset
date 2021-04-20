import numpy as np
import matplotlib.pyplot as plt
from example_tester import generate_demo_traj
from example_tester import extract_dataset_traj
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

#pulls one state vector
#x, y, vx, vy, psi= make_demonstrations(17)
all_x, all_y, all_vx, all_vy, all_psi= extract_dataset_traj("Scenario4", False, [1], [4], data_lim=100)
#all_x, all_y, all_vx, all_vy, all_psi= generate_demo_traj()
indices = np.arange(len(all_x))
np.random.shuffle(indices)
train_x, train_y, train_vx, train_vy, train_psi = [], [], [], [], []
test_x, test_y, test_vx, test_vy, test_psi = [], [], [], [], []
for i in range(len(indices)):
    if i < len(indices)*0.9: # percent of data
        train_x.append(all_x[indices[i]])
        train_y.append(all_y[indices[i]])
        train_vx.append(all_vx[indices[i]])
        train_vy.append(all_vy[indices[i]])
        train_psi.append(all_psi[indices[i]])
        #print(np.sum(all_vx[indices[i]]- all_vy[indices[i]]))
    else:
        test_x.append(all_x[indices[i]])
        test_y.append(all_y[indices[i]])
        test_vx.append(all_vx[indices[i]])
        test_vy.append(all_vy[indices[i]])
        test_psi.append(all_psi[indices[i]])

#pull individual trajectory


plot_figure = False
#number of data points
if plot_figure:
    n_samples = len(vx)
    print(n_samples)

    plt.figure(1)
    plt.plot(vx)
    plt.title('velocity x')

    plt.figure(2)
    plt.plot(vy)
    plt.title('velocity y')

    plt.figure(3)
    plt.plot(psi)
    plt.title('angle')
    plt.show()

#initilization
#data_mean = np.array[]

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


all_states_seg = []
all_states_traj =[]
for i in range(len(train_x)):
    all_states_seg_temp, all_states_traj_temp  = (state_seg_v2(train_x[i], train_y[i], train_psi[i], train_vx[i], train_vy[i], plot=True, seg_whole=False,
                       plot_states=True))
    all_states_seg.extend(all_states_seg_temp)
    all_states_traj.append(all_states_traj_temp)

all_states_seg = np.array(all_states_seg)
gmm = GaussianMixture(n_components=3, reg_covar=1, covariance_type="full", verbose=1)  # testing with 10, 5
gmm.fit(all_states_seg)

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
###########################################################GMM segmentation block#############################################
seg_ranges = state_seg_cluster_gmm(gmm, all_states_traj, plot = False)

###############################################################


###########################################################original segmentation block#############################################
# seg_ranges = []
# for i in range(len(train_x)):
#     #print(train_vx[i]-train_vy[i])
#     seg_ranges.append(state_segmentation(train_x[i], train_y[i], train_psi[i], train_vx[i], train_vy[i], plot=True, seg_whole= False, plot_states= True))

###############################################################

#algorithm 2
dk = np.linspace(0,1,K)
dt = np.linspace(0,1,T)
#dt = np.arange(1, T, 1)
#dt = np.array(list(range(1, T+1)))

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

def weight_gen(seg_range,x,y,vx,vy,psi,big_psi, plot = False, calc_accuracy = False, plot_states= True):
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
        #t_steps = np.linspace(0, T, t_data)

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

#clustering
list_pos_ww = []
list_neg_ww = []

list_ent_ww= []
list_mid_ww = []
list_exit_ww = []

j = 1
for i in range(len(seg_ranges)):
    list_ww = weight_gen(seg_ranges[i], train_x[i], train_y[i], train_vx[i], train_vy[i], train_psi[i], big_psi, plot = False, calc_accuracy=False, plot_states= False)

    for j in range(len( list_ww)):
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



#print(ww.shape)
# list_psi = np.array(list_psi)
#list_traj = np.array(list_traj)
# list_pos_ww = np.array(list_pos_ww) # make an array of the list
# print(list_pos_ww.shape)
list_ent_ww = np.array(list_ent_ww)
list_mid_ww = np.array(list_mid_ww)
list_exit_ww = np.array(list_exit_ww)




######################################################---------------------GMM testing------------------------############################################3
# gmm = GaussianMixture(n_components= 6, reg_covar= 1, covariance_type = "full", verbose = 1) # testing with 10, 5
# gmm.fit(list_pos_ww)
# clusters = gmm.predict(list_pos_ww)
# print (np.unique(clusters))
# list_ww_0 = []
# list_ww_1 = []
# list_ww_2 = []
# list_ww_3 = []
# list_ww_4 = []
# list_ww_5 = []
#
# for j in range(len(list_pos_ww)):
#     # ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
#     if clusters[j] == 0:
#         list_ww_0.append(list_pos_ww[j])
#     elif clusters[j] == 1:
#         list_ww_1.append(list_pos_ww[j])
#     elif clusters[j] == 2:
#         list_ww_2.append(list_pos_ww[j])
#     elif clusters[j] == 3:
#         list_ww_3.append(list_pos_ww[j])
#     elif clusters[j] == 4:
#         list_ww_4.append(list_pos_ww[j])
#     elif clusters[j] == 5:
#         list_ww_5.append(list_pos_ww[j])
#
# list_ww_0 = np.array(list_ww_0)
# list_ww_1 = np.array(list_ww_1)
# list_ww_2 = np.array(list_ww_2)
# list_ww_3 = np.array(list_ww_3)
# list_ww_4 = np.array(list_ww_4)
# list_ww_5 = np.array(list_ww_5)
# def plot_seg_mean(list_ww_temp, idx):
#     for j in range(len(list_ww_temp)):
#         #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
#         traj_all = np.dot(big_psi, list_ww_temp[j])
#
#         plt.plot(traj_all[0:T * D:D], traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0, alpha = 0.4)
#         plt.plot(traj_all[0], traj_all[1], 'og')
#         plt.plot(traj_all[-D], traj_all[-D+1], 'ob')
#     mean_0_ww = list_ww_temp.mean(0)
#     trained_traj = np.dot(big_psi, mean_0_ww)
#     plt.plot(trained_traj[0:T * D:D], trained_traj[1:T * D:D], "--", color="b", linewidth=2.0)
#     plt.plot(trained_traj[0], trained_traj[1], 'og')
#     plt.plot(trained_traj[-D], trained_traj[-D + 1], 'ob')
#     plt.legend()
#     plt.title("Observation of Cluster {} ".format(idx))
#     plt.show()
#
# plot_seg_mean(list_ww_0, 0)
# plot_seg_mean(list_ww_1, 1)
# plot_seg_mean(list_ww_2, 2)
# plot_seg_mean(list_ww_3, 3)
# plot_seg_mean(list_ww_4, 4)
# plot_seg_mean(list_ww_5, 5)



# Get mean and sigma for weights
# find th mean of the list
#mean_pos_ww = list_pos_ww.mean(0)
mean_ent_ww = list_ent_ww.mean(0)
mean_mid_ww = list_mid_ww.mean(0)
mean_exit_ww = list_exit_ww.mean(0)

#trained_traj = np.dot(big_psi, mean_pos_ww)
trained_traj_ent= np.dot(big_psi,mean_ent_ww)
trained_traj_mid= np.dot(big_psi,mean_mid_ww)
trained_traj_exit= np.dot(big_psi,mean_exit_ww)

def get_primitive_info():
    return trained_traj_ent, trained_traj_mid, trained_traj_exit


# #sigma_pos_ww_t = list_pos_ww.var(0)
# sigma_pos_ww_t = list_pos_ww.var(0)
# sigma_pos_ww = np.identity(sigma_pos_ww_t.shape[0])
# np.fill_diagonal(sigma_pos_ww, sigma_pos_ww_t)
#sigma_pos_ww= np.cov(list_pos_ww.T)

#new covariance metrics for the vehicle
# weight_var_init = 0.2*0.2
# weights_covar = np.ones((D*K, D*K))* weight_var_init
# weight_mean = list_pos_ww[0]
#
#
# # print("weight_var", weights_var)
# # print("weight_mean", temp_mean)
# # for demo_idx in range(list_pos_ww.shape[0]):
# #
# #     state =list_pos_ww[demo_idx]
# #     weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
# #     temp = np.expand_dims(state - weight_mean, 1)
# #     weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
# # mean_pos_ww = weight_mean
# # sigma_pos_ww = weights_covar
#############################################initializing for 3 clusters#############
weight_var_init = 0.2*0.2
weights_covar_ent = np.ones((D*K, D*K))* weight_var_init
weight_mean_ent = list_ent_ww[0]
weights_covar_mid = np.ones((D*K, D*K))* weight_var_init
weight_mean_mid = list_mid_ww[0]
weights_covar_exit = np.ones((D*K, D*K))* weight_var_init
weight_mean_exit = list_exit_ww[0]
weight_mean = list_ent_ww[0]
weights_covar = np.ones((D*K, D*K))* weight_var_init
for demo_idx in range(list_ent_ww.shape[0]):

    state =list_ent_ww[demo_idx]
    weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
    temp = np.expand_dims(state - weight_mean, 1)
    weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
mean_ent_ww = weight_mean
sigma_ent_ww = weights_covar

weight_mean = list_mid_ww[0]
weights_covar = np.ones((D*K, D*K))* weight_var_init
for demo_idx in range(list_mid_ww.shape[0]):

    state =list_mid_ww[demo_idx]
    weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
    temp = np.expand_dims(state - weight_mean, 1)
    weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
mean_mid_ww = weight_mean
sigma_mid_ww = weights_covar

weight_mean = list_exit_ww[0]
weights_covar = np.ones((D*K, D*K))* weight_var_init
for demo_idx in range(list_exit_ww.shape[0]):

    state =list_exit_ww[demo_idx]
    weight_mean = (weight_mean * demo_idx + state) / (demo_idx + 1)  # eq3
    temp = np.expand_dims(state - weight_mean, 1)
    weights_covar = (demo_idx / (demo_idx + 1)) * weights_covar + (demo_idx / (demo_idx + 1) ** 2) * temp * temp.T  # eq4
mean_exit_ww = weight_mean
sigma_exit_ww = weights_covar





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
    #trained_traj_upper = (np.dot(big_psi,  mean_pos_ww + np.sqrt(sigma_pos_ww.diagonal()))* 1)#+ trained_traj
    trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_pos_ww),big_psi.T)).diagonal())* 1 # + trained_traj
    # plt.figure()
    # plt.plot(mean_pos_ww, label="mean")
    # #plt.plot(mean_pos_ww+ np.sqrt(sigma_pos_ww.diagonal()), label="mean+sd")
    # plt.plot( np.sqrt(sigma_pos_ww.diagonal()),  label="sd" )
    # plt.legend()
    # plt.show()

    #trained_traj_lower = np.dot(big_psi, lower_mean)

    plt.figure()
    color_l = ["r", "b", "g", "y", "cyan", "m", "lime"]
    for j in range(len(list_pos_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_pos_ww[j])
        color= color_l[clusters[j]]
        # plt.plot(traj_all[0:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        plt.plot(traj_all[0:T * D:D], "--", color=color, linewidth=2.0, alpha= 0.4)
        #plt.show()
    #plt.plot(trained_traj[0:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    #plt.plot(np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj[0:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    #plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj[0:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of X state segments")
    plt.show()
    plt.figure()
    for j in range(len(list_pos_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_pos_ww[j])
        plt.plot(traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj[1:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj[1:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj[1:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of Y state segments")
    plt.show()
    plt.figure()
    for j in range(len(list_pos_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_pos_ww[j])
        plt.plot(traj_all[2:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj[2:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[2:T * D:D])+ trained_traj[2:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[2:T * D:D])+ trained_traj[2:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of Angle state segments")
    plt.show()
    plt.figure()
    for j in range(len(list_pos_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_pos_ww[j])
        plt.plot(traj_all[3:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj[3:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[3:T * D:D])+ trained_traj[3:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[3:T * D:D])+ trained_traj[3:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of VX state segments")
    plt.show()
    plt.figure()
    for j in range(len(list_pos_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_pos_ww[j])
        plt.plot(traj_all[4:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj[4:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[4:T * D:D])+ trained_traj[4:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[4:T * D:D])+ trained_traj[4:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of VY state segments")
    plt.show()

    plt.figure()
    for j in range(len(list_pos_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_pos_ww[j])
        plt.plot(traj_all[0:T * D:D], traj_all[1:T * D:D], "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj[0:T * D:D], trained_traj[1:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj[0:T * D:D], np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj[1:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj[0:T * D:D], -np.sqrt(trained_traj_upper[1:T * D:D])+trained_traj[1:T * D:D],  color="lime", label="Lower SD ", linewidth=2.0)
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

plot_wts_mn_cluster = True
if plot_wts_mn_cluster:
    trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_ent_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
    for j in range(len(list_ent_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_ent_ww[j])
        plt.plot(traj_all[0:T * D:D],  "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj_ent[0:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj_ent[0:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj_ent[0:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 0")
    plt.show()
    plt.figure()

    trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_ent_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
    for j in range(len(list_ent_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_ent_ww[j])
        plt.plot(traj_all[1:T * D:D],  "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj_ent[1:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj_ent[1:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj_ent[1:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 0")
    plt.show()
    plt.figure()

    trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_mid_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
    for j in range(len(list_mid_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_mid_ww[j])
        plt.plot(traj_all[0:T * D:D],  "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj_mid[0:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj_mid[0:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj_mid[0:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 1")
    plt.show()
    plt.figure()
    trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_mid_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
    for j in range(len(list_mid_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_mid_ww[j])
        plt.plot(traj_all[1:T * D:D],  "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj_mid[1:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj_mid[1:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj_mid[1:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 1")
    plt.show()
    plt.figure()
    trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_exit_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
    for j in range(len(list_exit_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_exit_ww[j])
        plt.plot(traj_all[0:T * D:D],  "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj_exit[0:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj_exit[0:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[0:T * D:D])+ trained_traj_exit[0:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of X state segments for Cluster 2")
    plt.show()
    plt.figure()
    trained_traj_upper = ((np.dot(np.dot(big_psi, sigma_exit_ww), big_psi.T)).diagonal()) * 1  # + trained_traj
    for j in range(len(list_exit_ww)):
        #ax1.plot(trained_traj[0:T * D:D], color="b", label="Primtive x", linewidth=2.0)
        traj_all = np.dot(big_psi, list_exit_ww[j])
        plt.plot(traj_all[1:T * D:D],  "--", color="#ff6a6a", linewidth=2.0)
        #plt.show()
    plt.plot(trained_traj_exit[1:T * D:D],  color="b", label="Primtive ", linewidth=2.0)
    plt.plot(np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj_exit[1:T * D:D],  color="g", label="Upper SD ", linewidth=2.0)
    plt.plot(-np.sqrt(trained_traj_upper[1:T * D:D])+ trained_traj_exit[1:T * D:D],  color="m", label="Lower SD ", linewidth=2.0)
    plt.legend()
    plt.title("Observation, Mean and Standard Deviation of Y state segments for Cluster 2")
    plt.show()
    plt.figure()


#
#
# list_neg_ww = np.array(list_neg_ww)
# # Get mean and sigma for weights
# mean_neg_ww = list_neg_ww.mean(0)
# sigma_neg_ww= np.cov(list_neg_ww.T)
# # sigma_neg_ww_t = list_neg_ww.var(0)
# # sigma_neg_ww = np.identity(sigma_neg_ww_t.shape[0])
# #np.fill_diagonal(sigma_ww, sigma_ww_t)
# # print("mean: ", mean_pos_ww.shape)
# # print("sigma: ", sigma_pos_ww.shape)


# inf_traj = np.dot(big_psi, mean_ww)
# inf_x = inf_traj[0:T*D:D]
# inf_y = inf_traj[1:T*D:D]
# fig=plt.figure()
# plt.plot(inf_x, inf_y, "--", color="#ff6a6a", linewidth=2.0)
# plt.show()

# Get gain
# k_segs = []
# for seg_idx in range(len(seg_range)):
#     k_seg = np.dot(np.dot(sigma_ww, big_psi.T),np.linalg.inv(np.dot(np.dot(big_psi,sigma_ww),big_psi.T)))
#     k_segs.append(np.copy(k_seg))
#     print("K: ", k_seg.shape)
# k_seg = np.dot(np.dot(sigma_ww, big_psi.T),np.linalg.inv(np.dot(np.dot(big_psi,sigma_ww),big_psi.T)))
# k_segs.append(k_seg)
# k_segs = np.array(k_segs)

# Get new mean and sigma for new observation
# Must get the new observation
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
        #t_steps = np.linspace(0, T, t_data)
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


# Must get correct segment for the observation as well, for taking gain (k_segs) and weights (list_psi)
# for i in range(len(test_x)):
#     get_observation_seg_for_testing(test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i], T, 10)

# seg_ranges_obv = []
# for i in range(len(test_x)):
#      seg_test_obs = state_segmentation(test_x[i], test_y[i],  test_psi[i], test_vx[i], test_vy[i], plot=False, seg_whole= False)
#      seg_ranges_obv.append(seg_2_traj(seg_test_obs, test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i]))


#observation_data =  get_observation_for_vehicle_whole(18, T)
#observation_data =  get_observation_for_vehicle(18, T)

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



###################previous reconstruction with the KF points ##############
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
#plt.legend()
#plt.show()

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


###################new reconstruction GMM with the KF points ##############
seg_ranges_obv = []
all_states_seg_test=[]
all_states_traj_test = []
for i in range(len(test_x)):
     all_states_seg_temp, all_states_traj_temp = (state_seg_v2(test_x[i], test_y[i], test_psi[i], test_vx[i], test_vy[i], plot=False, seg_whole=False,
                     plot_states=False))
     all_states_seg_test.extend(all_states_seg_temp)
     all_states_traj_test.append(all_states_traj_temp)

     # seg_test_obs = state_segmentation(test_x[i], test_y[i],  test_psi[i], test_vx[i], test_vy[i], plot=False, seg_whole= False)
     # seg_ranges_obv.append(seg_2_traj(seg_test_obs, test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i]))

seg_ranges_test = state_seg_cluster_gmm(gmm, all_states_traj_test, plot=False)

for i in range(len(seg_ranges_test)):
    seg_ranges_obv.append(seg_2_traj(seg_ranges_test[i], test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i]))



inferred_test_traj = []
RMSE_ent= []
RMSE_mid = []
RMSE_exit = []
for i in range (len(seg_ranges_obv)):
    current_test_traj= seg_ranges_obv[i]
    current_inferred_traj = []
    #plt.figure(10)


    for j in range (len (current_test_traj)):
        current_segment = current_test_traj[j]
        current_cluster = int(seg_ranges_test[i][j]['class'])

        if current_cluster == 0 :
            inferred_traj, _, _ = reconstruct_seg_goal(current_segment, np.copy(mean_ent_ww), np.copy(sigma_ent_ww))
        elif current_cluster == 1:
            inferred_traj, _, _ = reconstruct_seg_goal(current_segment, np.copy(mean_mid_ww), np.copy(sigma_mid_ww))
        elif current_cluster == 2:
            inferred_traj, _, _ = reconstruct_seg_goal(current_segment, np.copy(mean_exit_ww), np.copy(sigma_exit_ww))

        #
        #
        # current_inferred_traj.append(inferred_traj)

        inferred_x = inferred_traj[0:inferred_traj.shape[0]:D]
        inferred_y = inferred_traj[1:inferred_traj.shape[0]:D]
        inferred_psi = inferred_traj[2:inferred_traj.shape[0]:D]
        inferred_vx = inferred_traj[3:inferred_traj.shape[0]:D]
        inferred_vy = inferred_traj[4:inferred_traj.shape[0]:D]
        # primitive_traj= big_psi* mean_pos_ww
        # primitive_x = primitive_traj[0:current_segment.shape[0]:D]
        # primitive_y = primitive_traj[0:current_segment.shape[0]:D]
        observation_x = current_segment[0:current_segment.shape[0]:D]
        observation_y = current_segment[1:current_segment.shape[0]:D]
        observation_psi = current_segment[2:current_segment.shape[0]:D]
        observation_vx = current_segment[3:current_segment.shape[0]:D]
        observation_vy = current_segment[4:current_segment.shape[0]:D]

        calc_accuracy = True
        if calc_accuracy:
            diff_x =  inferred_x- observation_x
            diff_y =  inferred_y - observation_y
            diff_psi = inferred_psi- observation_psi
            diff_vx =  inferred_vx- observation_vx
            diff_vy =  inferred_vy- observation_vy

            RMSE_x = np.sqrt(sum(diff_x**(2))/T)
            RMSE_y = np.sqrt(sum(diff_y**(2))/T)
            RMSE_psi = np.sqrt(sum(diff_psi ** (2))/T)
            RMSE_vx = np.sqrt(sum(diff_vx**(2))/T)
            RMSE_vy= np.sqrt(sum(diff_vy ** (2))/T)
            print("Segment ", current_cluster, " RMSE accuracy", RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy)

            if current_cluster == 0:
                RMSE_ent.append([RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy])
            elif current_cluster == 1:
                RMSE_mid.append([RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy])
            elif current_cluster == 2:
                RMSE_exit.append([RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy])



        obv_colors = ['r','b', 'g']
        inf_colors = ['m', 'c', 'limegreen']
        plt.plot(observation_x, observation_y, "--", color=obv_colors[current_cluster], label="Observation" + str(current_cluster), linewidth=2.0)
        plt.plot(inferred_x, inferred_y, ".", color=inf_colors[current_cluster], label="Inferred"+ str(current_cluster), linewidth=3.0)
        plt.legend()

    plt.show()
    inferred_test_traj.append(current_inferred_traj)

#plt.legend()
#plt.show()

RMSE_ent = np.array(RMSE_ent)
RMSE_mid = np.array(RMSE_mid)
RMSE_exit = np.array(RMSE_exit)

mean_RMSE_ent = np.mean(RMSE_ent, 0)
mean_RMSE_mid = np.mean(RMSE_mid, 0)
mean_RMSE_exit = np.mean(RMSE_exit, 0)
print("Mean error in cluster 0 ", mean_RMSE_ent )
print("Mean error in cluster 1 ", mean_RMSE_mid )
print("Mean error in cluster 2 ", mean_RMSE_exit )































