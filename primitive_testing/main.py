#from game.autonomous_vehicle import AutonomousVehicle
#from game.sim_draw import Sim_Draw
#from game.sim_data import Sim_Data
import pickle
import os
#import pygame as pg
import datetime
#from trajectory_interaction import get_primitive_info
from primitive_file import get_primitive_info

#generate
class Main():
    def __init__(self):
        # self.P = "Roundabout name"
        self.car_1 = AutonomousVehicle(secnario_parameter = self.P,
                                       loss_style = "reactive", #note to change the reactive one
                                       who=1 )#ego vehicle
        self.car_2 = AutonomousVehicle(scenario_parameter = self.P,
                                     loss_style="reactive",  # note to change the reactive one
                                     who=0)  # ego vehicle)
        # # assign another car
        #
        # self.car_1.other_car = self.car_2
        # self.car_2.other_car = self.car_1
        # self.car_1.states_o = self.car_2.states
        # self.car_2.states_o = self.car_1.states

        #dummy tester
        self.time_stamp_ms_first = 0
        self.time_stamp_ms_last = 100

        self.vehicle_runtime = self.time_stamp_ms_first
        self.running = True


        self.trial()
        self.dist_threshold = 10




    def trial(self):
        T = 20  # number of timesteps
        K = 80  # number of basis fun
        D = 5  # dimension of data
        while self.running:
            mean_traj_1, mean_traj_2, mean_traj_3 = get_primitive_info(car_1, car_2)
            mp_x = mean_traj_1[0:T * D:D]
            mp_y = mean_traj_1[1: T*D: D]
            # distance = (x_1-pred_x_2)**2+ (y_1-pred_y_2)**2)**(1/2)
            # if (distance < self.dist_threshold):
            #     Cost_F_1 =

            if (self.vehicle_runtime > self.time_stamp_ms_last):
                self.running = False

            #print(self.vehicle_runtime)

            self.vehicle_runtime = self.vehicle_runtime+1





if __name__ == "__main__":
    Main()