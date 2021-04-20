import numpy as np
import matplotlib.pyplot as plt
from example_tester import generate_demo_traj
from example_tester import extract_dataset_traj
import scipy.interpolate
import sys
import csv


sys.path.append('C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset/data')
np.random.seed(123)

# constants for the primitive
# T = 20 #number of timesteps
# K = 8 # number of basis function acc to the paper
T = 20 #number of timesteps
K = 80 # number of basis function acc to the paper
# T = 100 #number of timesteps
# K = 25 # number of basis function acc to the paper
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
all_x, all_y, all_vx, all_vy, all_psi= extract_dataset_traj("Scenario4", False, [3], [5], data_lim=100)
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
def state_segmentation(x, y, psi, vx, vy, plot = True, seg_whole = False):
    var = np.zeros(D)
    state = np.array((x[0], y[0],  psi[0], vx[0], vy[0]))
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
        temp2 = np.expand_dims(state[2:3],1) - cond_new_orientation_mean
        d_mo = np.matmul(temp2.T, np.matmul(np.linalg.pinv(cond_orientation_var),temp2))

        #algorithm 1
        dp_emax = 5.0
        do_Mmax = np.pi/4

        # initiate segmentation library
        if not (seg_whole):
            if (d_ep>dp_emax and d_mo>do_Mmax):
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

    # check segmentation
    #print(seg_range)
    if plot:
        seg_colors=["b","r","g"]
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
            #fig = plt.figure()
            plt.plot(traj_x, traj_y, ".", color=seg_colors[seg_idx%3], label="Segmented "+str(seg_idx), linewidth=2.0)
            fig.suptitle('Segmented Trajectory')
            plt.xlabel('X')
            plt.ylabel('Y')

        plt.legend()
        plt.show()
    return seg_range

seg_ranges = []
for i in range(len(train_x)):
     seg_ranges.append(state_segmentation(train_x[i], train_y[i], train_psi[i], train_vx[i], train_vy[i], plot=False, seg_whole= False))


#algorithm 2
dk = np.linspace(0,1,K)
dt = np.linspace(0,1,T)
#primitive_mean = dt.mean() #1
#primitive_variance = 0.2
#primitive_variance = 5
primitive_variance = np.linspace(0.2,10, 50)
# list_psi = []
#list_traj = []
#list_ww = []


# equation 12 and 13
big_psi_list = []
for jj in range(primitive_variance.shape[0]):
    big_psi = np.zeros((D * T, D * K))

    for ii in range(T):  # number of sequence
        for kk in range(K):
            b_t_x = get_GausRBF(dt[ii], dk[kk], primitive_variance[jj], D)
            # b_t_y = get_GausRBF(z_y[ii], primitive_mean, primitive_variance, D)
            # b_t_o = get_GausRBF(z_psi[ii], primitive_mean, primitive_variance, D)
            # b_t_vx = get_GausRBF(z_vx[ii], primitive_mean, primitive_variance, D)
            # b_t_vy = get_GausRBF(z_vy[ii], primitive_mean, primitive_variance, D)
            small_psi = np.identity(D)
            # np.fill_diagonal(small_psi, [b_t_x, b_t_y, b_t_o, b_t_vx, b_t_vy])
            np.fill_diagonal(small_psi, [b_t_x, b_t_x, b_t_x, b_t_x, b_t_x])
            # np.fill_diagonal(small_psi, [b_t_x])
            big_psi[ii * D:(ii + 1) * D, kk * D:(kk + 1) * D] = np.copy(small_psi)
    #print("Big Psi shape: ", big_psi.shape)
    big_psi_list.append(np.copy(big_psi))

def weight_gen(seg_range,x,y,vx,vy,psi,big_psi, plot = False, calc_accuracy = False):
    #equation 11
    list_ww = []

    alphas = [1e-9, 1e-11, 1e-13]
    sum_RMSE = np.zeros((primitive_variance.shape[0] * len(alphas), 7))
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
        #alpha = 1e-11 #from promp papaer from france
        #alpha = 1e-7
        #alpha = 0
        list_RMSE = []
        for i in range(len(big_psi_list)):
            for j in range(len(alphas)):
                alpha = alphas[j]
                big_psi_t = big_psi_list[i]
                ww =get_Ridge_Refression (big_psi_t, Trajectory, alpha)
                # list_psi.append(np.copy(big_psi))
                #list_traj.append(np.copy(Trajectory))
                #list_ww.append(np.copy(ww))

                # TESTER

                traj_x = Trajectory[0:T * D:D]
                traj_y = Trajectory[1:T * D:D]  # np.arange(traj_x.shape[0])#
                traj_psi = Trajectory[2:T * D:D]
                traj_vx = Trajectory[3:T * D:D]
                traj_vy = Trajectory[4:T * D:D]

                new_traj = np.dot(big_psi_t, ww)
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
                    #print("Primitive variance", primitive_variance, " RMSE accuracy", RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy)
                    list_RMSE.append([RMSE_x, RMSE_y, RMSE_psi, RMSE_vx, RMSE_vy, primitive_variance[i],alphas[j]])

        list_RMSE= np.array(list_RMSE)
        sum_RMSE[:,0:5] += list_RMSE[:,0:5]
        sum_RMSE[:,5:7] = list_RMSE[:,5:7]



        if plot:
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
    return sum_RMSE

#clustering
sum_RMSE = np.zeros((primitive_variance.shape[0]*3, 7))
totalsegment= 0
for i in range(len(seg_ranges)):
    list_RMSE = weight_gen(seg_ranges[i], train_x[i], train_y[i], train_vx[i], train_vy[i], train_psi[i], big_psi, plot = False, calc_accuracy=True)
    sum_RMSE[:,0:5] += list_RMSE[:,0:5]
    totalsegment += len(seg_ranges[i])
mean_RMSE = sum_RMSE/ totalsegment
min_all= np.min(mean_RMSE,0)
min_all_pos = np.argmin(mean_RMSE,0)
minimum_x = mean_RMSE[min_all_pos[0],0]
min_x_var = list_RMSE[min_all_pos[0], 5]
min_x_alpha = list_RMSE[min_all_pos[0], 6]
print("Minimum x primitive variance =",  min_x_var, " Minimum x regression lambda =",  min_x_alpha," Minimum x RMSE =",  minimum_x )
minimum_y = mean_RMSE[min_all_pos[1],1]
min_y_var = list_RMSE[min_all_pos[1], 5]
min_y_alpha = list_RMSE[min_all_pos[1], 6]
print("Minimum y primitive variance =",  min_y_var, " Minimum y regression lambda =",  min_y_alpha," Minimum y RMSE =",  minimum_y )
minimum_psi = mean_RMSE[min_all_pos[2],2]
min_psi_var = list_RMSE[min_all_pos[2], 5]
min_psi_alpha = list_RMSE[min_all_pos[2], 6]
print("Minimum psi primitive variance =",  min_psi_var, " Minimum psi regression lambda =",  min_psi_alpha," Minimum psi RMSE =",  minimum_psi )
minimum_vx = mean_RMSE[min_all_pos[3],3]
min_vx_var = list_RMSE[min_all_pos[3], 5]
min_vx_alpha = list_RMSE[min_all_pos[3], 6]
print("Minimum vx primitive variance =",  min_vx_var, " Minimum vx regression lambda =",  min_vx_alpha," Minimum vx RMSE =",  minimum_vx )
minimum_vy = mean_RMSE[min_all_pos[4],4]
min_vy_var = list_RMSE[min_all_pos[4], 5]
min_vy_alpha = list_RMSE[min_all_pos[4], 6]
print("Minimum vy primitive variance =",  min_vy_var, " Minimum vy regression lambda =",  min_vy_alpha," Minimum vy RMSE =",  minimum_vy )

mean_RMSE[:,5:7]= list_RMSE[:,5:7]
np.savetxt("gridsearch.csv", mean_RMSE, delimiter=",")

# with open("gridsearch.csv", 'w') as wid:
#     for i in range(mean_RMSE.shape[0]):
#         wid.write(mean_RMSE)



#
# #print(ww.shape)
# # list_psi = np.array(list_psi)
# #list_traj = np.array(list_traj)
# list_pos_ww = np.array(list_pos_ww)
# # Get mean and sigma for weights
# mean_pos_ww = list_pos_ww.mean(0)
# sigma_pos_ww_t = list_pos_ww.var(0)
# sigma_pos_ww = np.identity(sigma_pos_ww_t.shape[0])
#
#                                                                # --- COMMENTED OUT NEGATIVE CLUSTERS
# '''
# list_neg_ww = np.array(list_neg_ww)
# # Get mean and sigma for weights
# mean_neg_ww = list_neg_ww.mean(0)
# sigma_neg_ww_t = list_neg_ww.var(0)
# sigma_neg_ww = np.identity(sigma_neg_ww_t.shape[0])
# # print("mean: ", mean_pos_ww.shape)
# # print("sigma: ", sigma_pos_ww.shape)
# '''
#
# # inf_traj = np.dot(big_psi, mean_ww)
# # inf_x = inf_traj[0:T*D:D]
# # inf_y = inf_traj[1:T*D:D]
# # fig=plt.figure()
# # plt.plot(inf_x, inf_y, "--", color="#ff6a6a", linewidth=2.0)
# # plt.show()
#
# # Get gain
# # k_segs = []
# # for seg_idx in range(len(seg_range)):
# #     k_seg = np.dot(np.dot(sigma_ww, big_psi.T),np.linalg.inv(np.dot(np.dot(big_psi,sigma_ww),big_psi.T)))
# #     k_segs.append(np.copy(k_seg))
# #     print("K: ", k_seg.shape)
# # k_seg = np.dot(np.dot(sigma_ww, big_psi.T),np.linalg.inv(np.dot(np.dot(big_psi,sigma_ww),big_psi.T)))
# # k_segs.append(k_seg)
# # k_segs = np.array(k_segs)
#
# # Get new mean and sigma for new observation
# # Must get the new observation
# def get_observation_for_vehicle(vehicle_no, T):
#     x, y, vx, vy, psi= make_demonstrations(vehicle_no)
#     Trajectory = []
#     for i in range(0, len(x), 50):
#         seg_x = x[i:(i + 50)]
#         print(len(seg_x))
#         seg_y = y[i:(i + 50)]
#         seg_vx = vx[i:(i + 50)]
#         seg_vy = vy[i:(i + 50)]
#         seg_psi = psi[i:(i + 50)]
#
#         #look ahead ateps of 50
#         t_steps = np.linspace(0, 1, len(seg_x))
#         # T = 8
#
#         path_gen_x = scipy.interpolate.interp1d(t_steps, seg_x)
#         path_gen_y = scipy.interpolate.interp1d(t_steps, seg_y)
#         path_gen_vx = scipy.interpolate.interp1d(t_steps, seg_vx)
#         path_gen_vy = scipy.interpolate.interp1d(t_steps, seg_vy)
#         path_gen_psi = scipy.interpolate.interp1d(t_steps, seg_psi)
#
#         z_x = np.zeros(T)
#         z_y = np.zeros(T)
#         z_vx = np.zeros(T)
#         z_vy = np.zeros(T)
#         z_psi = np.zeros(T)
#         for t in range(T):
#             z_x[t] = path_gen_x(dt[t])
#             z_y[t] = path_gen_y(dt[t])
#             z_vx[t] = path_gen_vx(dt[t])
#             z_vy[t] = path_gen_vy(dt[t])
#             z_psi[t] = path_gen_psi(dt[t])
#             Trajectory.append(z_x[t])
#             Trajectory.append(z_y[t])
#             Trajectory.append(z_psi[t])
#             Trajectory.append(z_vx[t])
#             Trajectory.append(z_vy[t])
#     return Trajectory
#
#
# def get_observation_for_vehicle_whole(vehicle_no, T):
#     x, y, vx, vy, psi= make_demonstrations(vehicle_no)
#     Trajectory = []
#     seg_x = x
#     print(len(seg_x))
#     seg_y = y
#     seg_vx = vx
#     seg_vy = vy
#     seg_psi = psi
#
#     #look ahead ateps of 50
#     t_steps = np.linspace(0, 1, len(seg_x))
#     # T = 8
#
#     path_gen_x = scipy.interpolate.interp1d(t_steps, seg_x)
#     path_gen_y = scipy.interpolate.interp1d(t_steps, seg_y)
#     path_gen_vx = scipy.interpolate.interp1d(t_steps, seg_vx)
#     path_gen_vy = scipy.interpolate.interp1d(t_steps, seg_vy)
#     path_gen_psi = scipy.interpolate.interp1d(t_steps, seg_psi)
#
#     z_x = np.zeros(T)
#     z_y = np.zeros(T)
#     z_vx = np.zeros(T)
#     z_vy = np.zeros(T)
#     z_psi = np.zeros(T)
#     for t in range(T):
#         z_x[t] = path_gen_x(dt[t])
#         z_y[t] = path_gen_y(dt[t])
#         z_vx[t] = path_gen_vx(dt[t])
#         z_vy[t] = path_gen_vy(dt[t])
#         z_psi[t] = path_gen_psi(dt[t])
#         Trajectory.append(z_x[t])
#         Trajectory.append(z_y[t])
#         Trajectory.append(z_psi[t])
#         Trajectory.append(z_vx[t])
#         Trajectory.append(z_vy[t])
#     return Trajectory
#
#
# def get_observation_seg_for_testing(x, y, vx, vy, psi, T, interval):
#     Trajectory = []
#     for i in range(0, len(x), interval):
#         seg_x = x[i:(i + interval)]
#         #print(len(seg_x))
#         seg_y = y[i:(i + interval)]
#         seg_vx = vx[i:(i + interval)]
#         seg_vy = vy[i:(i + interval)]
#         seg_psi = psi[i:(i + interval)]
#
#         #look ahead ateps of 50
#         t_steps = np.linspace(0, 1, len(seg_x))
#         # T = 8
#
#         path_gen_x = scipy.interpolate.interp1d(t_steps, seg_x)
#         path_gen_y = scipy.interpolate.interp1d(t_steps, seg_y)
#         path_gen_vx = scipy.interpolate.interp1d(t_steps, seg_vx)
#         path_gen_vy = scipy.interpolate.interp1d(t_steps, seg_vy)
#         path_gen_psi = scipy.interpolate.interp1d(t_steps, seg_psi)
#
#         z_x = np.zeros(T)
#         z_y = np.zeros(T)
#         z_vx = np.zeros(T)
#         z_vy = np.zeros(T)
#         z_psi = np.zeros(T)
#         for t in range(T):
#             z_x[t] = path_gen_x(dt[t])
#             z_y[t] = path_gen_y(dt[t])
#             z_vx[t] = path_gen_vx(dt[t])
#             z_vy[t] = path_gen_vy(dt[t])
#             z_psi[t] = path_gen_psi(dt[t])
#             Trajectory.append(z_x[t])
#             Trajectory.append(z_y[t])
#             Trajectory.append(z_psi[t])
#             Trajectory.append(z_vx[t])
#             Trajectory.append(z_vy[t])
#     return Trajectory
#
# def seg_2_traj(seg_range,x,y,vx,vy,psi):
#     list_traj =[]
#     for seg_idx in range(len(seg_range)):
#     #for seg_idx in range(1):
#         Trajectory = []
#         segment= seg_range[seg_idx]
#         data_r = segment["data_range"]
#         t_start = data_r[0]
#         t_end = data_r[1]
#         t_data = t_end-t_start+1
#         t_steps = np.linspace(0, 1, t_data)
#         #x, y, vx, vy, psi
#         seg_x = x[t_start:t_end+1]
#         #print("seg_x shape: ",len(seg_x))
#         seg_y = y[t_start:t_end+1]
#         seg_vx = vx[t_start:t_end+1]
#         seg_vy = vy[t_start:t_end+1]
#         seg_psi = psi[t_start:t_end+1]
#
#         path_gen_x = scipy.interpolate.interp1d(t_steps, seg_x)
#         path_gen_y = scipy.interpolate.interp1d(t_steps, seg_y)
#         path_gen_vx = scipy.interpolate.interp1d(t_steps, seg_vx)
#         path_gen_vy = scipy.interpolate.interp1d(t_steps, seg_vy)
#         path_gen_psi = scipy.interpolate.interp1d(t_steps, seg_psi)
#
#         z_x = np.zeros(T)
#         z_y = np.zeros(T)
#         z_vx = np.zeros(T)
#         z_vy = np.zeros(T)
#         z_psi = np.zeros(T)
#         for t in range(T):
#             z_x[t] = path_gen_x(dt[t])
#             z_y[t] = path_gen_y(dt[t])
#             z_vx[t] = path_gen_vx(dt[t])
#             z_vy[t] = path_gen_vy(dt[t])
#             z_psi[t] = path_gen_psi(dt[t])
#             Trajectory.append(z_x[t])
#             Trajectory.append(z_y[t])
#             Trajectory.append(z_psi[t])
#             Trajectory.append(z_vx[t])
#             Trajectory.append(z_vy[t])
#
#         Trajectory = np.array(Trajectory)
#         list_traj.append(Trajectory)
#     return list_traj
#
#
# # Must get correct segment for the observation as well, for taking gain (k_segs) and weights (list_psi)
# # for i in range(len(test_x)):
# #     get_observation_seg_for_testing(test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i], T, 10)
#
# seg_ranges_obv = []
# for i in range(len(test_x)):
#      seg_test_obs = state_segmentation(test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i], plot=True, seg_whole= False)
#      seg_ranges_obv.append(seg_2_traj(seg_test_obs, test_x[i], test_y[i], test_vx[i], test_vy[i], test_psi[i] ))
#
#
# #observation_data =  get_observation_for_vehicle_whole(18, T)
# #observation_data =  get_observation_for_vehicle(18, T)
#
# def reconstruct_seg(observation_data, mean_ww, sigma_ww, prior_seg=[] ):
#     # each segment into T steps
#
#     observation_data = np.array(observation_data)
#     observation_x = observation_data[0:T*D:D]
#     observation_y = observation_data[1:T*D:D]
#     # observation_traj = np.vstack((observation_x,observation_y))
#
#     span = D
#     len_observation = int(observation_data.shape[0]/span)
#     if not len_observation == T:
#         print("Observed data length: ", len(observation_data))
#     prior = 2
#     lookahead = 2
#     # for the beginning of the segment
#     # if not len(prior_seg) == 0:
#     #     observation = prior_seg
#     #     psi_obv = np.copy(big_psi[-2*span:0])
#     #     k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
#     #     mean_ww = mean_ww + np.dot(k_seg, (observation - np.dot(psi_obv, mean_ww)))
#     #     sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
#
#
#     # for the rest of the trajectory
#     for i in range (prior,len_observation):
#         # Update K, mean, sigma, psi_obv
#         #observation = observation_data[0:(i + 1) * span]
#         observation = observation_data[(i-prior)*span:i*span]
#         #psi_obv = np.copy(big_psi[0:(i+1)*span])
#         psi_obv = np.copy(big_psi[(i-prior)*span:i*span])
#
#         k_seg = np.dot(np.dot(sigma_ww, psi_obv.T), np.linalg.inv(np.dot(np.dot(psi_obv, sigma_ww), psi_obv.T)))
#         mean_ww = mean_ww + np.dot(k_seg, (observation - np.dot(psi_obv, mean_ww)))
#         sigma_ww = sigma_ww - np.dot(k_seg, np.dot(psi_obv, sigma_ww))
#
#         #Infer for new trajectory
#         # inferred_traj = np.dot(big_psi, mean_ww)
#         # inferred_x = inferred_traj[0:inferred_traj.shape[0]:5]
#         # inferred_y = inferred_traj[1:inferred_traj.shape[0]:5]
#         # inferred = np.vstack((inferred_x, inferred_y))
#
#
#         # plt.plot(observation[0:observation.shape[0]:5], ".", color="#ff6a6a", label="Observation", linewidth=2.0)
#         # plt.plot(inferred[0],inferred[1], color = "#85d87f", label = "Inferred", linewidth = 3.0)
#         # plt.show()
#         inferred_traj = np.dot(big_psi, mean_ww)
#         inferred_traj = inferred_traj[0:(i+lookahead)*span]
#
#         #inferred at each step
#         inferred_x = inferred_traj[0:inferred_traj.shape[0]:D]
#         inferred_y = inferred_traj[1:inferred_traj.shape[0]:D]
#         plt.plot(observation_x, observation_y, "--", color="#ff6a6a", label="Observation", linewidth=2.0)
#         plt.plot(inferred_x, inferred_y, ".", color = "#85d87f", label = "Inferred", linewidth = 2.0)
#         plt.legend()
#         plt.show()
#
#     return inferred_traj, mean_ww, sigma_ww
#
# inferred_test_traj = []
# for i in range (len(seg_ranges_obv)):
#     current_test_traj= seg_ranges_obv[i]
#     current_inferred_traj = []
#     #plt.figure(10)
#
#     for j in range (len (current_test_traj)):
#         current_segment = current_test_traj[j]
#         #check is positive or negative class
#         if current_segment[-D]>0 :
#             if j ==0:
#                 inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg(current_segment,mean_pos_ww, sigma_pos_ww)
#             else:
#                 inferred_traj, mean_pos_ww, sigma_pos_ww = reconstruct_seg(current_segment, mean_pos_ww, sigma_pos_ww, prior_seg= current_segment[-2*D: 0])
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
#         plt.show()
#         inferred_test_traj.append(current_inferred_traj)
# #plt.legend()
# #plt.show()


































