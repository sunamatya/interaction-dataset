import pandas as pd
import numpy as np
# Plotting Packages
import matplotlib.pyplot as plt
import seaborn as sbn
# Configuring Matplotlib
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
savefig_options = dict(format="png", dpi=300, bbox_inches="tight")
import scipy.interpolate

#computational packages
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from example_tester import extract_dataset_traj
from example_tester import extract_dataset_traj_active



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


def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    #distances = compute_euclidean_distance_matrix(x, y)
    distances = compute_euclidean_distance_matrix_2D(x,y)

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


K = 80
T = 100
D = 1
def get_GausRBF_alpha(z_t,mean_k,sigma,D, alpha):
    return np.exp((-0.5*(alpha*z_t-mean_k)*(alpha* z_t-mean_k))*(1./sigma)) / (np.sqrt(np.power(2*np.pi,D) * sigma))

def generate_psi_dt_ts(T, alpha):
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
            b_t_x = get_GausRBF_alpha(dt[ii], dk[kk], primitive_variance, D, alpha)
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

# def tester_old():
#     x = [3, 1, 2, 2, 1]
#     y = [2, 0, 0, 3, 3, 1, 0]
#     distance, warp_path = fastdtw(x, y, dist=euclidean)
#
#     cost_matrix = compute_accumulated_cost_matrix(x, y)
#
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax = sbn.heatmap(cost_matrix, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
#     ax.invert_yaxis()
#
#     # Get the warp path in x and y directions
#     path_x = [p[0] for p in warp_path]
#     path_y = [p[1] for p in warp_path]
#
#     # Align the path from the center of each cell
#     path_xx = [x+0.5 for x in path_x]
#     path_yy = [y+0.5 for y in path_y]
#
#     ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.2)
#
#
#
#     fig, ax = plt.subplots(figsize=(14, 10))
#
#     # Remove the border and axes ticks
#     fig.patch.set_visible(False)
#     ax.axis('off')
#
#     for [map_x, map_y] in warp_path:
#         ax.plot([map_x, map_y], [x[map_x], y[map_y]], '--k', linewidth=4)
#
#     ax.plot(x, '-ro', label='x', linewidth=4, markersize=20, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
#     ax.plot(y, '-bo', label='y', linewidth=4, markersize=20, markerfacecolor='skyblue', markeredgecolor='skyblue')
#     ax.set_title("DTW Distance", fontsize=28, fontweight="bold")
#
#     #fig.savefig("ex1_dtw_distance.png", **savefig_options)
#     plt.show()
#
#     #fig.savefig("ex1_heatmap.png", **savefig_options)
#
#     res = [min(i) for i in zip(*cost_matrix)][len(x)-1]
#     r, c = np.where(cost_matrix == res)
#
#     print ("phase =", (r+1)/(len(y)))
#
#     print(distance)
#     print(warp_path)

def data_teter_1d():
    all_x, all_y, all_vx, all_vy, all_psi = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=100)

    #y = all_x[1][0::10] # demonstrated trajectory
    y = all_x[1]
    x = all_x[2][1:40] #observed trajectory
    t_obs = 40
    plt.plot(x, 'g')
    #plt.plot(y)
    #plt.show()
    data_len_x = len(x)
    data_len_y = len(y)
    print(data_len_x)
    print(data_len_y)
    t_steps = np.linspace(0, 1, data_len_x)
    path_gen_x = scipy.interpolate.interp1d(t_steps, x)
    dt = np.linspace(0, 1, T)
    Trajectory = []
    z_x = np.zeros(T)
    for t in range(T):
        z_x[t] = path_gen_x(dt[t])
        Trajectory.append(z_x[t])
    Trajectory= np.array(Trajectory)
    cost_matrix1 = compute_accumulated_cost_matrix(x, y)
    cost_matrix2 = compute_accumulated_cost_matrix(Trajectory, y)
    res = [min(i) for i in zip(*cost_matrix1)][len(x)-1]
    r, c = np.where(cost_matrix1 == res)
    phase = (r+1)/(len(y))
    print ("phase1 =", (r+1)/(len(y)))
    print("total time if phase is constant = ",  t_obs/phase)

    res = [min(i) for i in zip(*cost_matrix2)][T-1]
    r, c = np.where(cost_matrix2 == res)
    phase = (r+1)/(len(y))
    print ("phase2 =", (r+1)/(len(y)))

    print("total time if phase is constant = ",  t_obs/phase)

    plt.plot(Trajectory, 'ro')
    plt.plot(y, 'bo')
    plt.show()

    # big_psi_ent_mean, _ = generate_psi_dt_ts(100, phase)
    # trained_traj_ent = np.dot(big_psi_ent_mean, mean_ent_ww)

    # x = np.array([1, 2, 3, 3, 7])
    # y = np.array([1, 2, 2, 2, 2, 2, 2, 4])


if __name__ == "__main__":
    all_x, all_y, all_vx, all_vy, all_psi = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=100)
    active_x, active_y, active_vx, active_vy, active_psi, active_int, active_b, active_e = extract_dataset_traj_active(
        "Scenario4", False, [3], [5], data_lim=100, track_id_list=[37, 47, 77])  # [37,47,77]

    #y = all_x[1][0::10] # demonstrated trajectory
    y = [all_x[1], all_y[1]]
    #x = [all_x[3][0:20], all_y[3][0:20]] #observed trajectory #non interacting trajectory
    x = [active_x[0][0:5], active_y[0][0:5]]



    t_obs = 5 #10
    plt.figure()
    plt.plot(x[0],x[1], 'g', label = 'Observed trajectory')
    plt.plot(y[0], y[1], 'r', label = 'Saved trajectory')
    plt.xlabel('x distance(m)')
    plt.ylabel('y distance (m)')
    plt.legend()
    plt.show()
    data_len_x = len(x[0])
    data_len_y = len(y[0])
    print(data_len_x)
    print(data_len_y)

    cost_matrix1 = compute_accumulated_cost_matrix_2D(x, y)

    res = [min(i) for i in zip(*cost_matrix1)][len(x[0])-1]
    r, c = np.where(cost_matrix1 == res)
    phase = (r+1)/(len(y[0]))
    print ("phase1 =", phase)
    print("total time if phase is constant = ",  t_obs/phase)

    phase_prev = 0.0
    #testing phase based on previous obervation
    for j in range (0, len(all_x[3])-5, 5):
       # x = [all_x[3][0:j+20], all_y[3][0:j+20]]  # observed trajectory #non interacting trajectory
        x = [active_x[0][0:j+5], active_y[0][0:j+5]]  # observed trajectory #interacting trajectory

        cost_matrix1 = compute_accumulated_cost_matrix_2D(x, y)
        res = [min(i) for i in zip(*cost_matrix1)][len(x[0]) - 1]
        r, c = np.where(cost_matrix1 == res)
        phase = (r + 1) / (len(y[0]))
        #print(r)

        print("phase passed =", phase)
        print("total time if phase is constant = ", t_obs / (phase-phase_prev))
        phase_prev = phase
        plt.figure()
        plt.plot(x[0], x[1], 'g', label='Observed trajectory')
        plt.plot(y[0], y[1], 'r', label='Saved trajectory')
        plt.xlabel('x distance(m)')
        plt.ylabel('y distance (m)')
        plt.legend()
        plt.show()

