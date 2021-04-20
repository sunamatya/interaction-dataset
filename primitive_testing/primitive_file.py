import numpy as np
import matplotlib.pyplot as plt
import os, sys
import csv
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from python.utils import dataset_reader

from sklearn.mixture import GaussianMixture

from example_tester import extract_dataset_traj_active
from example_tester import extract_dataset_traj
from GT_primitive_primary import state_seg_v2
from GT_primitive_primary import state_seg_cluster_gmm
from GT_primitive_primary import generate_psi_dt
from GT_primitive_primary import weight_gen
from frenetframe import get_tn_dp
from frenetframe import frenet
from frenetframe import mapper

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




def get_primitive_info( ):
    active_x, active_y, active_vx, active_vy, active_psi, active_int, active_b, active_e = extract_dataset_traj_active(
        "Scenario4", True, [3], [5], data_lim=1000)
    for i in range(len(active_b)):
        #print(active_x[i][1:10])
        interval = np.array(active_e[i])- np.array(active_b[i])
        ts = list(range(0, interval+ 100, 100))
        ts_act = list(range(active_b[i], np.array(active_e[i]+100), 100))
        #pseudo code
        plt.plot(np.array(ts_act), active_vy[i], 'b')  # , label='interacting velocity')
        for k,v in active_int[i].items():
            data_ts=[]
            data_x = []
            for j in range(len(v)):
                try:
                    data_ts.append(v[j])
                    data_x.append(active_vy[i][ts_act.index(v[j])])
                except:
                    print(v[j], ts_act.index(v[j]))

                plt.plot(data_ts, data_x, 'r')
    plt.show()

        # get_ pair_data(interacting_all)
    return active_int


def dynamics(X, u):
    sx, sy, v_x, v_y, ang = X[0], X[1], X[2], X[3], X[4]
    dt = 0.1

    ang_new = ang + u[0] *dt
    vx_new = v_x + u[1] * np.cos(ang)*dt
    vy_new = v_y + u[1] * np.sin(ang)* dt
    sx_new = sx + vx_new * dt
    sy_new = sy + vy_new * dt

    return [sx_new, sy_new, vx_new, vy_new, ang_new]

def dynamics_frenet(X, u):
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    X = np.matmul(A, X) + np.matmul(B, u)
    return X


def get_primitive(train_x, train_y, train_psi, train_vx, train_vy):
    T = 20  # number of timesteps
    K = 80  # number of basis function acc to the paper
    D = 5  # dimension of data
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

    return mean_ent_ww, mean_mid_ww, mean_exit_ww, sigma_ent_ww, sigma_mid_ww, sigma_exit_ww, gmm

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
    plt.plot(all_x[1], all_y[1], 'g', label = 'non interacting vehicle')
    plt.plot(all_x2[0], all_y2[0], 'y', label = 'other non interacting vehicle')
    plt.legend()

    plt.show()


    # generate primitive information



    #predifined data pair
    base_timestamp1 = 0
    base_timestamp2 = 0
    if active_b[0]<= active_2b[0]:
        ptime_start= active_b[0]
        active_start = active_2b[0]
        base_timestamp1 = ptime_start
        base_timestamp2 = active_start
    else:
        ptime_start = active_2b[0]
        active_start = active_b[0]
        base_timestamp1 = active_start
        base_timestamp2 = ptime_start

    if active_e[0] >= active_2e[0]:
        ptime_end = active_e[0]
        active_end = active_2e[0]
    else:
        ptime_end = active_2e[0]
        active_end = active_e[0]


    #first car enters ( primitive)
    #second car enters (primitive ) (calculating TTC)
    #when TTC (game)
    #primitive second car
    #primirive first car
    timesteps_car1= list(np.arange(active_b[0], active_e[0], 100))
    timesteps_car2 = list(np.arange(active_2b[0], active_2e[0], 100))
   # print(timesteps_car1)

    game_frenet = False
    calc_frenet = False



    while ptime_start< active_end: # primitive end time
        hascar1= False
        hascar2= False

        if ptime_start in timesteps_car1:
            time_index1 = timesteps_car1.index(ptime_start)
            position_1 = (active_x[0][time_index1] ** 2 + active_y[0][time_index1] ** 2) ** (1 / 2)
            velocity_1 = (active_vx[0][time_index1] ** 2 + active_vy[0][time_index1] ** 2) ** (1 / 2)
            hascar1= True

        if ptime_start in timesteps_car2:
            time_index2 = timesteps_car2.index(ptime_start)
            position_2 = (active_2x[0][time_index2] ** 2 + active_2y[0][time_index2] ** 2) ** (1 / 2)
            velocity_2 = (active_2vx[0][time_index2] ** 2 + active_2vy[0][time_index2] ** 2) ** (1 / 2)
            hascar2= True

        print(hascar1, " has car ",  hascar2)
        ptime_start = ptime_start + 100
        nodes_removed = 0
        # when both car are present #TODO: but have to start primitive when one car enters the intersections

        if hascar1 and hascar2:
            car_length = 5.0 # we chose maximum
            TTC = np.abs((position_2- position_1-car_length)/ (velocity_1- velocity_2))


            print(TTC)
            #total_prediction= 10
            car1_states_inferred_x = []
            car2_states_inferred_x = []
            car1_states_inferred_y = []
            car2_states_inferred_y = []

            car_states_true= []

            if TTC < 5.0:
                horizon = 3  # time horizon
                #total_prediction = total_prediction -1
                ts = 0.1  # 100 ms
                if game_frenet:
                    # if calc_frenet:
                    #     #calculate the frenet frame for the cars
                    #     # take the baseline primitive trajectory to generate the frenet frame trajectory
                    #     # calculate l,s from it
                    #     x, y, s_c, vx, vy = get_tn_dp(all_x2[0][time_index2: len(all_x2[0])],
                    #                                   all_y2[0][time_index2: len(all_x2[0])])
                    #     x1, y1, s_c1, vx1, vy1 = get_tn_dp(all_x[0][time_index1: len(all_x[0])],
                    #                                        all_y[0][time_index1: len(all_x[0])])
                    #     tester_mapper = mapper()
                    #     tester_mapper.map_waypoints_x = x
                    #     tester_mapper.map_waypoints_y = y
                    #     tester_mapper.map_waypoints_s = s_c
                    #     tester_mapper.map_waypoints_dx = vx
                    #     tester_mapper.map_waypoints_dy = vy
                    #
                    #     tester_frenet = frenet()
                    #     d_test = []
                    #     s_test = []
                    #
                    #     tester_mapper1 = mapper()
                    #     tester_mapper1.map_waypoints_x = x1
                    #     tester_mapper1.map_waypoints_y = y1
                    #     tester_mapper1.map_waypoints_s = s_c1
                    #     tester_mapper1.map_waypoints_dx = vx1
                    #     tester_mapper1.map_waypoints_dy = vy1
                    #
                    #     tester_frenet1 = frenet()
                    #     d_test1 = []
                    #     s_test1 = []
                    #
                    #     # print(len(all_x2[0]))
                    #     # for i in range(1, len(all_x2[0])-1):
                    #     for i in range(time_index2 + 1, len(all_x2[0]) - 1):
                    #         tester_frenet = tester_mapper.get_frenet(all_x2[0][i], all_y2[0][i], all_psi2[0][i])
                    #         # print(tester_frenet.s)
                    #         s_test.append(tester_frenet.s)
                    #
                    #     # for i in range(1, len(all_x1[0])-1):
                    #     for i in range(time_index1 + 1, len(all_x[0]) - 1):
                    #         tester_frenet1 = tester_mapper1.get_frenet(all_x[0][i], all_y[0][i], all_psi[0][i])
                    #         # print(tester_frenet.s)
                    #         s_test1.append(tester_frenet1.s)
                    #
                    #     plt.plot(s_test, 'rx')
                    #     plt.plot(s_test1, 'bo')
                    #     plt.show()

                        calc_frenet = False
                    # calculate the ls for the actual trajectory (car, state)

            #         action_set = [-2, -1, 0, 1]
            #
            #         # car_state1 = [active_x[0][time_index1], active_y[0][time_index1], active_vx[0][time_index1],
            #         #               active_vy[0][time_index1], active_psi[0][time_index1]]
            #         # car_state2 = [active_2x[0][time_index2], active_2y[0][time_index2], active_2vx[0][time_index2],
            #         #               active_2vy[0][time_index2], active_2psi[0][time_index2]]
            #
            #         car_state1 =
            #         car_state2 =
            #
            #         cost_matrix = np.zeros((5, 4, 4, 2))
            #         state_matrix_car_1= np.zeros((6,4,5))
            #         state_matrix_car_2 = np.zeros((6,4,4,5))
                else:
                    action_set = [[-0.1, -1], [0, -1], [0.1, -1], [-0.1, 0], [0, 0], [0.1, 0], [-0.1, 1], [0, 1],
                                  [0.1, 1]]
                    #action_set = [[-0.5, -5], [0, -5], [0.5, -5]]
                    #action_set = [[-0.1, -1], [0, -1], [0.1, -1], [-0.1, 1], [0, 1], [0.1, 1]]
                    car_state1 = [active_x[0][time_index1], active_y[0][time_index1], active_vx[0][time_index1],
                                  active_vy[0][time_index1], active_psi[0][time_index1]]
                    car_state2 = [active_2x[0][time_index2], active_2y[0][time_index2], active_2vx[0][time_index2],
                                  active_2vy[0][time_index2], active_2psi[0][time_index2]]
                    # primitive1 = [all_x[1][time_index1: time_index1+ horizon ], all_y[1][time_index1 : time_index1+ horizon]]
                    # primitive2 = [all_x2[0][time_index2 : time_index2+ horizon], all_y2[0][time_index2 : time_index2+ horizon]]

                    primitive1 = [all_x[1][time_index1], all_y[1][time_index1]]
                    primitive2 = [all_x2[0][time_index2], all_y2[0][time_index2]]

                    # cost_matrix = np.zeros((5, 9, 9, 2))
                    # state_matrix_car_1= np.zeros((6,9,5))
                    # state_matrix_car_2 = np.zeros((6,9,9,5))

                # state_matrix_car_1[0,:]= car_state1
                # state_matrix_car_2[0,:,:]= car_state2

                treedict= {} #empty dictionary
                root_node = data_node("root")
                root_node.isroot = True
                root_node.level = 0
                root_node.previous_states = [car_state1, car_state2]
                root_node.current_states = [car_state1, car_state2]
                root_node.parent = "root"
                root_node.cost = [0,0]
                treedict["root"]= root_node
                #children node initialized for the tree


                #build the cost matrix
                haschildren= True
                current_nodes_list = [root_node]
                min_cost1, min_cost2 = 100000, 100000
                best_node = None

                while haschildren:
                    if len(current_nodes_list) == 0:
                        # Done for all children
                        haschildren = False
                        continue

                    current_node = current_nodes_list[-1]
                    current_children_list = current_node.children
                    level = current_node.level
                    if current_node.finish_count >= len(action_set)*len(action_set):
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
                                for j in range(len(action_set)):
                                    name = str(level+1) + "_" + str(i) + "_" + str(j)
                                    child_node = data_node(name)
                                    child_node.level = level+1
                                    child_node.current_action = [action_set[i], action_set[j]]
                                    child_node.parent = current_node.name
                                    treedict[name] = child_node
                                    current_node.children.append(child_node)


                        for child_index in range(len(current_children_list)):
                            child_node= current_children_list[child_index]
                            child_node.previous_states = treedict[child_node.parent].current_states
                            car_state1 = child_node.previous_states[0]
                            car_state2 = child_node.previous_states[1]

                            pred_car_1 = dynamics(car_state1,child_node.current_action[0])
                            pred_car_2 = dynamics(car_state2, child_node.current_action[1])

                            child_node.current_states = [pred_car_1, pred_car_2]

                            car_distance = ((car_state1[0] - car_state2[0]) ** 2 - (
                                        car_state1[1] - car_state2[1]) ** 2) ** (1 / 2)
                            # self distance from primitive
                            change_pred1 = ((car_state1[0] - primitive1[0]) ** 2 - (car_state1[1] - primitive1[1])) ** (0.5)
                            change_pred2 = ((car_state2[0] - primitive2[0]) ** 2 - (car_state2[1] - primitive2[1])) ** (0.5)

                            #find cost
                            # update child's cost and states
                            cost1 = (change_pred1 - 1 / car_distance)
                            cost2 = (change_pred2 - 1 / car_distance)


                            parent_cost = treedict[current_node.parent].cost

                            child_node.cost = [parent_cost[0]+cost1, parent_cost[1]+cost2]
                            # print("parent: ", current_node.parent)
                            # print("Level ", child_node.level)
                            # print("State ", child_node.current_states)
                            # print("Cost: ", child_node.cost)
                            # print("Actions: ", child_node.current_action)
                            # print("-" * 20)

                            if level< (horizon-1):
                                current_nodes_list.append(child_node)
                            else:
                                # print(child_node.cost)
                                if child_node.cost[0] < min_cost1 and child_node.cost[1] < min_cost2:
                                    best_node = child_node
                                    min_cost1, min_cost2 = child_node.cost

                        if level == (horizon-1):
                            treedict[current_node.parent].finish_count += 1
                            current_nodes_list = current_nodes_list[:-1]

                while True:
                    print("Level ", best_node.level)
                    print("State ", best_node.current_states)
                    print("Cost: ", best_node.cost)
                    print("Actions: ", best_node.current_action)
                    print("-"*20)
                    #car_states_inferred.append(best_node.current_states)
                    car1_states_inferred_x.append(best_node.current_states[0][0])
                    car2_states_inferred_x.append(best_node.current_states[1][0])
                    car1_states_inferred_y.append(best_node.current_states[0][1])
                    car2_states_inferred_y.append(best_node.current_states[1][0])
                    if best_node.name=="root":
                        break
                    best_node = treedict[best_node.parent]
                #break
                #print("car states inferred 0", car_states_inferred[0])
                #print("car states inferred 1", car_states_inferred[1])
                print("car states inferred 0", car1_states_inferred_x, " ", car1_states_inferred_y)
                print("car states inferred 1", car2_states_inferred_x, " ", car2_states_inferred_y)

                plt.figure()
                plt.plot(car1_states_inferred_x, car1_states_inferred_y, 'r')
                plt.plot(car2_states_inferred_x, car2_states_inferred_y, 'b')
                plt.plot(active_x[0][time_index1 : time_index1+ horizon], active_y[0][ time_index1 : time_index1+ horizon], 'g')
                plt.plot(active_2x[0][time_index2 : time_index2+ horizon], active_2y[0][time_index2 : time_index2+ horizon],'y')
                plt.show()

                exit()

            ptime_start = ptime_start + 100# i #



