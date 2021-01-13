import numpy as np
import matplotlib.pyplot as plt

import sys
import csv
#sys.path.append('C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset/data')        --Sunny specific
# C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset
def make_demonstrations():
    name = 'vehicle_tracks_000.csv'
    with open('../data/' + name, 'r') as csvfile:
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
            if row[0]== 2: # vehicle 2 data for testing
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
    return np.exp((-0.5*(z_t-mean_k)*(z_t-mean_k))/sigma) / (np.sqrt(np.power(2*np.pi,D) * sigma))

def get_Ridge_Refression (X, Y, alpha):
    I = np.identity(len(X))
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha * I), X.T), Y)
    return w

'''====================================================
    MAIN
===================================================='''

#pulls one state vector
x, y, vx, vy, psi= make_demonstrations()
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

#state clustering

var = np.zeros(5)
state = np.array((x[0], y[0],  psi[0], vx[0], vy[0]))
data_mean = state
#data_covar = np.fill_diagonal(np.identity(5), var)
data_covar = np.zeros((5, 5))

#list for data
seg_range = []
current_range_start = 0
print(len(x))
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
    cond_new_pos_mean = cond_pos_mean+ np.matmul (np.matmul(np.concatenate((cond_po, cond_pv),1), np.linalg.inv(data_covar[2:5, 2:5])), temp[2:5, 0:1])
    #eq8
    cond_pos_var = cond_pp - np.matmul (np.matmul(np.concatenate((cond_po, cond_pv),1), np.linalg.inv(data_covar[2:5, 2:5])), data_covar[2:5, 0:2])

    #orientation
    inv_temp_mat_1 = np.concatenate((cond_pp, cond_pv), 1)
    inv_temp_mat_2 = np.concatenate((cond_vp, cond_vv), 1)
    inv_temp_mat = np.concatenate((inv_temp_mat_1, inv_temp_mat_2), 0)
    mean_data_temp = np.concatenate((temp[0:2, 0:1], temp[3:5, 0:1]), 0)
    cond_new_orientation_mean = cond_orientation_mean + np.matmul(np.matmul(np.concatenate((cond_op, cond_ov),1), np.linalg.inv(inv_temp_mat)), mean_data_temp)
    cond_orientation_var = cond_oo - np.matmul (np.matmul(np.concatenate((cond_op, cond_ov),1), np.linalg.inv(inv_temp_mat)), np.concatenate((cond_po, cond_vo), 0))

    #velocity
    inv_temp_mat_1 = np.concatenate((cond_pp, cond_po), 1)
    inv_temp_mat_2 = np.concatenate((cond_op, cond_oo), 1)
    inv_temp_mat = np.concatenate((inv_temp_mat_1, inv_temp_mat_2), 0)
    cond_new_velocity_mean = cond_velocity_mean+ np.matmul (np.matmul(np.concatenate((cond_vp, cond_vo),1), np.linalg.inv(inv_temp_mat)), temp[0:3, 0:1])
    cond_velocity_var = cond_vv - np.matmul (np.matmul(np.concatenate((cond_vp, cond_vo),1), np.linalg.inv(inv_temp_mat)), np.concatenate((cond_pv, cond_ov),0))

    #euclidian distance system # eq9 and eq10 np.expand_dims(state-data_mean, 1)
    temp1 = np.expand_dims(state[0:2],1) - cond_new_pos_mean
    d_ep = np.sqrt(np.matmul(temp1.T, temp1))
    temp2 = np.expand_dims(state[2:3],1) - cond_new_orientation_mean
    d_mo = np.matmul(temp2.T, np.matmul(np.linalg.inv(cond_orientation_var),temp2))

    #algorithm 1
    dp_emax = 8.0
    do_Mmax = np.pi

    # initiate segmentation library -> @TODO?

    if (d_ep>dp_emax and d_mo>do_Mmax):
        #dictionary to save data
        cluster_info = {}
        cluster_info["data_range"] =[current_range_start, demo_idx]
        cluster_info["states_info_mean"] = data_mean
        cluster_info["states_info_var"] = data_covar
        seg_range.append(cluster_info)

        #reinitialize mean
        data_covar = np.zeros((5, 5))
        data_mean = state
        current_range_start = demo_idx+1
# --- end for

if current_range_start< demo_idx:
    cluster_info = {}
    cluster_info["data_range"] = [current_range_start, demo_idx]
    cluster_info["states_info_mean"] = data_mean
    cluster_info["states_info_var"] = data_covar
    seg_range.append(cluster_info)

# check segmentation
print(seg_range)

#algorithm 2
# constants
T = 2
K = 2
D = 5
primitive_mean = 1
primitive_variance = 0.1
dt = np.linspace(0,1,T)
Big_psi= np.zeros((D*T, D*K))
import scipy.interpolate
#eq11
for seg_idx in range(len(seg_range)):
    segment= seg_range[seg_idx]
    data_r = segment["data_range"]
    t_start = data_r[0]
    t_end = data_r[1]
    t_data = t_end-t_start+1
    t_steps = np.linspace(0, 1, t_data)
    #x, y, vx, vy, psi
    seg_x = x[t_start:t_end+1]
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

    #equation 12 and 13


    # for each segmentation
    # for each state
    # for each gaussian function
    #breaks = np.array(np.where(z_x[0] != z_x[0]))[0]
    #dmp_sets = []
    #generate psi, make equation 13
    for ii in range(T): #number of sequence
        for kk in range(K):
            b_t_x = get_GausRBF(z_x[ii], primitive_mean, primitive_variance, D)
            b_t_y = get_GausRBF(z_y[ii], primitive_mean, primitive_variance, D)
            b_t_o = get_GausRBF(z_psi[ii], primitive_mean, primitive_variance, D)
            b_t_vx = get_GausRBF(z_vx[ii], primitive_mean, primitive_variance, D)
            b_t_vy = get_GausRBF(z_vy[ii], primitive_mean, primitive_variance, D)
            small_psi = np.identity(5)
            np.fill_diagonal(small_psi, [b_t_x, b_t_y, b_t_o, b_t_vx, b_t_vy])
            Big_psi[ii*D:(ii+1)*D, kk*D:(kk+1)*D]= np.copy(small_psi)

    print(Big_psi.shape)
































