import numpy as np
import matplotlib.pyplot as plt
#from example_tester import generate_demo_traj
from primitive_testing.example_tester import extract_dataset_traj_active
import scipy.interpolate
import sys
import csv
import math

class frenet():
    s = 0
    d = 0

class mapper():
    def __init__(self):
        self.map_waypoints_x = []
        self.map_waypoints_y =[]
        self.map_waypoints_s = []
        self.map_waypoints_dx = []
        self.map_waypoints_dy =[]




    def get_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))

    def closest_way_point(self, x, y):
        closestlen= 1000000
        closest_waypoint = 0
        for i in range (len(self.map_waypoints_x)-1):
            map_x = self.map_waypoints_x[i]
            map_y = self.map_waypoints_y[i]
            dist = self.get_distance(x, y, map_x, map_y)
            if dist< closestlen:
                closestlen = dist
                closest_waypoint = i

        return closest_waypoint

    def NextWayPoint(self, x, y, theta):
        closestWaypoint = self.closest_way_point(x,y)
        dx = self.map_waypoints_dx[closestWaypoint]
        dy = self.map_waypoints_dy[closestWaypoint]

        heading = math.atan2(dy,dx)+ np.pi/2
        #print("dy, dx, atan2", dy, " ", dx, " ", heading)
        angle = np.abs(theta-heading)
        if (angle > np.pi / 4):
            closestWaypoint +=1
            if (closestWaypoint == len(self.map_waypoints_dx) - 1):
                closestWaypoint = 0

        return closestWaypoint




    def cyclic_index(self, next_wp):
        if (next_wp == 0):
            return len(self.map_waypoints_x)-2
        else:
            return next_wp-1



    def get_frenet(self, x, y, theta):
        next_wp = self.NextWayPoint(x, y, theta)
        prev_wp = self.cyclic_index(next_wp)

        n_x = self.map_waypoints_x[next_wp] - self.map_waypoints_x[prev_wp]
        n_y = self.map_waypoints_y[next_wp] - self.map_waypoints_y[prev_wp]
        x_x = x - self.map_waypoints_x[prev_wp]
        x_y = y - self.map_waypoints_y[prev_wp]

        proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y

        f1 = frenet()
        f1.d = self.get_distance(x_x, x_y, proj_x, proj_y)


        center_x = 1000 - self.map_waypoints_x[prev_wp]
        center_y = 2000 - self.map_waypoints_y[prev_wp]
        centerToPos = self.get_distance(center_x, center_y, x_x, x_y)
        centerToRef = self.get_distance(center_x, center_y, proj_x, proj_y)

        if (centerToPos <= centerToRef):
            f1.d *= -1
        f1.s = 0

        for i in range(prev_wp):
            f1.s += self.get_distance(self.map_waypoints_x[i], self.map_waypoints_y[i], self.map_waypoints_x[i + 1], self.map_waypoints_y[i + 1])

        f1.s += self.get_distance(0, 0, proj_x, proj_y)

        return f1;

    def deg2rad (x):
        return x*np.pi/180

    def rad2deg(x):
        return x*180/np.pi


def get_tn(all_x2, all_y2):
    x = []
    y = []
    vx = []
    vy = []
    s_c= [0]
    for i in range(len(all_x2[0])-1):
    #for i in range(len(all_x2[0]) - 1):
        # if i == 0:
        #     s_c.append(0)
        # calculate s
        y_2 = all_y2[0][i+1]
        y_1 = all_y2[0][i]
        x_2 = all_x2[0][i+1]
        x_1 = all_x2[0][i]
        y_diff = (y_2 - y_1)
        x_diff = (x_2 - x_1)
        x.append(x_1)
        y.append(y_1)
        s_l = np.sqrt(x_diff*x_diff + y_diff *y_diff)
        s_c_temp = s_c[i]+s_l
        s_c.append(s_c_temp)

        # calculate normals
        n_1 = y_diff/s_l
        n_2 =- x_diff/s_l
        print(n_1)
        vx.append(n_1)
        vy.append(n_2)

    return x, y, s_c, vx, vy

def get_tn_dp(all_x2, all_y2):
    x = []
    y = []
    vx = []
    vy = []
    s_c= [0]
    for i in range(len(all_x2)-1):
    #for i in range(len(all_x2[0]) - 1):
        # if i == 0:
        #     s_c.append(0)
        # calculate s
        y_2 = all_y2[i+1]
        y_1 = all_y2[i]
        x_2 = all_x2[i+1]
        x_1 = all_x2[i]
        y_diff = (y_2 - y_1)
        x_diff = (x_2 - x_1)
        x.append(x_1)
        y.append(y_1)
        s_l = np.sqrt(x_diff*x_diff + y_diff *y_diff)
        s_c_temp = s_c[i]+s_l
        s_c.append(s_c_temp)

        # calculate normals
        n_1 = y_diff/s_l
        n_2 =- x_diff/s_l
        print(n_1)
        vx.append(n_1)
        vy.append(n_2)

    return x, y, s_c, vx, vy

def test_original_transformation():
    x = []
    y = []
    s = []
    vx = []
    vy = []
    ang = [0]
    s_c= [0]
    ######################### Data from online paper###################
    with open('C:\\Users\\samatya.ASURITE\\Desktop\\primitives\\game_theory\\highway_map.csv', 'r') as vehicletesttrack:
        csvReader = csv.reader(vehicletesttrack, delimiter=' ', quotechar='|')
        for row in csvReader:
            #content = list(row[i] for i in included_cols)
            lx = row[0]
            ly = row[1]
            lo = row[2]
            lvx = row[3]
            lvy = row[4]
            x.append(float(lx))
            y.append(float(ly))
            s.append(float(lo))
            vx.append(float(lvx))
            vy.append(float(lvy))

    # THIS IS FOR THE DIRECT DATA FROM THE HIGHWAY_WAP
    for i in range(len(x)-1):
        # if i == 0:
        #     s_c.append(0)
        # calculate s
        y_2 = y[i+1]
        y_1 = y[i]
        x_2 = x[i+1]
        x_1 = x[i]
        y_diff = (y_2 - y_1)
        x_diff = (x_2 - x_1)
        s_l = np.sqrt(x_diff*x_diff + y_diff *y_diff)
        s_c_temp = s_c[i]+s_l
        s_c.append(s_c_temp)

        # calculate normals
        n_1 = y_diff/s_l
        n_2 =- x_diff/s_l
        #print(x[i], " ", y[i], " ", s_c[i], " ", n_1, " ", n_2)
        print(n_2, " ", vy[i])
        ang_temp = math.atan2(y_diff, x_diff)
        ang.append(ang_temp)

    plt.plot(x, y, 'rx')
    #plt.ylabel('some numbers')
    #plt.show()

    tester_mapper = mapper()
    tester_mapper.map_waypoints_x = x
    tester_mapper.map_waypoints_y = y
    tester_mapper.map_waypoints_s = s
    tester_mapper.map_waypoints_dx = vx
    tester_mapper.map_waypoints_dy = vy

    tester_frenet = frenet()
    d_test = []
    s_test = []
    for i in range (1, len (x)-1):
        tester_frenet = tester_mapper.get_frenet(x[i],y[i],ang[i])
        #print(tester_frenet.s)
        s_test.append(tester_frenet.s)
    plt.plot(s_test)
    plt.show()
    #     s_test = np.append(tester_frenet.s.values())
    #     d_test = np.append(tester_frenet.d)
    #
    plt.plot(s_test)




if __name__ == "__main__":

    #test for the original code for the round vehicle trajectory to frenet frame
    #test_original_transformation()


    # testing for two vehiles

    all_x2, all_y2, all_vx2, all_vy2, all_psi2, all_int, all_2b, all_2e = extract_dataset_traj_active("Scenario4", True, [3], [5], data_lim= 100, track_id_list = [37]) #37- 41 are pair

    #all references to 1 are the second vehicle that has been added for secong vehicle and the testing
    all_x1, all_y1, all_vx1, all_vy1, all_psi1, all_int1, all_1b, all_1e = extract_dataset_traj_active("Scenario4", True, [3], [5], data_lim= 100, track_id_list = [41]) #37- 41 are pair

    #predifined data pair
    if all_1b[0]<= all_2b[0]:
        ptime_start= all_1b[0]
        active_start = all_2b[0]
    else:
        ptime_start = all_2b[0]
        active_start = all_1b[0]

    if all_1e[0] >= all_2e[0]:
        ptime_end = all_1e[0]
        active_end = all_2e[0]
    else:
        ptime_end = all_2e[0]
        active_end = all_1e[0]

    print("active start and active end", ptime_start, active_end)


    #first car enters ( primitive)
    #second car enters (primitive ) (calculating TTC)
    #when TTC (game)
    #primitive second car
    #primirive first car
    timesteps_car1= list(np.arange(all_1b[0], all_1e[0], 100))
    timesteps_car2 = list(np.arange(all_2b[0], all_2e[0], 100))

    while ptime_start< active_end: # primitive end time
        hascar1= False
        hascar2= False

        if ptime_start in timesteps_car1:
            time_index1 = timesteps_car1.index(ptime_start)
            position_1 = (all_x1[0][time_index1] ** 2 + all_y1[0][time_index1] ** 2) ** (1 / 2)
            velocity_1 = (all_vx1[0][time_index1] ** 2 + all_vy1[0][time_index1] ** 2) ** (1 / 2)
            hascar1= True

        if ptime_start in timesteps_car2:
            time_index2 = timesteps_car2.index(ptime_start)
            position_2 = (all_x2[0][time_index2] ** 2 + all_y2[0][time_index2] ** 2) ** (1 / 2)
            velocity_2 = (all_vx2[0][time_index2] ** 2 + all_vy2[0][time_index2] ** 2) ** (1 / 2)
            hascar2= True


        print(hascar1, " has car ",  hascar2)
        ptime_start = ptime_start + 100
        if hascar1 and hascar2:
            car_length = 5.0 # we chose maximum
            TTC = np.abs((position_2- position_1-car_length)/ (velocity_1- velocity_2))
            print(TTC)

            if TTC < 10.0:

                x = []
                y = []
                s = []
                vx = []
                vy = []
                ang = [0]
                s_c= [0]

                x1 = []
                y1 = []
                s1 = []
                vx1 = []
                vy1 = []
                ang1 = [0]
                s_c1= [0]
                ################################
                # x, y, s_c, vx, vy = get_tn(all_x2[0][time_index2:len(all_x2[0])], all_y2[0][time_index2: len(all_x2[0])])
                # x1, y1, s_c1, vx1, vy1 = get_tn(all_x1[0][time_index1 :len(all_x1[0])], all_y1[0][time_index1: len(all_x1[0])])

                # x, y, s_c, vx, vy = get_tn_dp(all_x2[0][time_index2:len(all_x2[0])], all_y2[0][time_index2: len(all_x2[0])])
                # x1, y1, s_c1, vx1, vy1 = get_tn_dp(all_x1[0][time_index1 :len(all_x2[0])], all_y1[0][time_index1: len(all_x2[0])])

                x, y, s_c, vx, vy = get_tn_dp(all_x2[0][time_index2: len(all_x2[0])], all_y2[0][time_index2: len(all_x2[0])])
                x1, y1, s_c1, vx1, vy1 = get_tn_dp(all_x1[0][time_index1 : len(all_x1[0])], all_y1[0][time_index1: len(all_x1[0])])

                plt.plot(all_x2, all_y2, 'rx')
                plt.plot(all_x1, all_y1, 'bo')
                plt.show()

                ###############################frenet frame from the data points##########################3
                tester_mapper = mapper()
                tester_mapper.map_waypoints_x = x
                tester_mapper.map_waypoints_y = y
                tester_mapper.map_waypoints_s = s_c
                tester_mapper.map_waypoints_dx = vx
                tester_mapper.map_waypoints_dy = vy

                tester_frenet = frenet()
                d_test = []
                s_test = []

                tester_mapper1 = mapper()
                tester_mapper1.map_waypoints_x = x1
                tester_mapper1.map_waypoints_y = y1
                tester_mapper1.map_waypoints_s = s_c1
                tester_mapper1.map_waypoints_dx = vx1
                tester_mapper1.map_waypoints_dy = vy1

                tester_frenet1 = frenet()
                d_test1 = []
                s_test1 = []

                #print(len(all_x2[0]))
                #for i in range(1, len(all_x2[0])-1):
                for i in range(time_index2+1, len(all_x2[0]) - 1):
                    tester_frenet = tester_mapper.get_frenet(all_x2[0][i],all_y2[0][i],all_psi2[0][i])
                    #print(tester_frenet.s)
                    s_test.append(tester_frenet.s)

                #for i in range(1, len(all_x1[0])-1):
                for i in range(time_index1+1, len(all_x1[0]) - 1):
                    tester_frenet1 = tester_mapper1.get_frenet(all_x1[0][i],all_y1[0][i],all_psi1[0][i])
                    #print(tester_frenet.s)
                    s_test1.append(tester_frenet1.s)


                plt.plot(s_test, 'rx')
                plt.plot(s_test1, 'bo')
                plt.show()
                #     s_test = np.append(tester_frenet.s.values())
                #     d_test = np.append(tester_frenet.d)
                #
                #plt.plot(s_test)













