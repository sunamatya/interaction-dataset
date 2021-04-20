"""
python 3.6 and up is required
records agent info
"""
import numpy as np
import scipy
from sim_data import DataUtil
import pygame as pg
import dynamics


class AutonomousVehicle:
    """
    States:
            X-Position
            Y-Position
    """
    def __init__(self, sim, env, par, inference_model, decision_model, i):
        self.sim = sim
        self.env = env  # environment parameters
        self.car_par = par  # car parameters
        self.inference_model = inference_model
        self.decision_model = decision_model
        self.id = i

        # Initialize variables
        self.state = self.car_par["initial_state"]  # state is cumulative
        self.intent = self.car_par["par"]
        self.action = self.car_par["initial_action"]  # action is cumulative
        self.trajectory = []
        self.planned_actions_set = []
        self.planned_trajectory_set = []
        self.initial_belief = self.get_initial_belief(self.env.car_par[1]['belief'][0], self.env.car_par[0]['belief'][0],
                                                      self.env.car_par[1]['belief'][1], self.env.car_par[0]['belief'][1],
                                                      weight=0.8)  # note: use params from the other agent's belief
        # Initialize prediction variables
        self.predicted_intent_all = []
        self.predicted_intent_other = []
        self.predicted_intent_self = []
        self.predicted_policy_other = []
        self.predicted_policy_self = []
        "for recording predicted state from inference"
        # TODO: check this predicted action: to match the time steps we put 0 initially, but fixes?
        #self.predicted_actions_other = [0]  # assume initial action of other agent = 0
        #self.predicted_actions_self = [0]
        self.predicted_actions_other = []  # assume initial action of other agent = 0
        self.predicted_actions_self = []
        self.predicted_states_self = []
        self.predicted_states_other = []
        self.min_speed = 0.1
        self.max_speed = 30

    def update(self, sim):
        other = sim.agents[:self.id]+sim.agents[self.id+1:]  # get all other agents
        frame = sim.frame

        # take a snapshot of the state at beginning of the frame before agents update their states
        snapshot = sim.snapshot()  # snapshot = agent.copy() => snapshot taken before updating

        # perform inference
        inference = self.inference_model.infer(snapshot, sim)
        DataUtil.update(self, inference)

        # planning
        plan = self.decision_model.plan()

        # update state
        action = plan["action"]
        if self.sim.decision_type[self.id] == "baseline" \
                or self.sim.decision_type[self.id] == "baseline2" \
                or self.sim.decision_type[self.id] == "non-empathetic"\
                or self.sim.decision_type[self.id] == "empathetic":  #TODO: need to be able to check indivisual type
            action = action[self.id]
            plan = {"action": action}
        DataUtil.update(self, plan)
        print("chosen action", action)
        self.dynamics(action)

    def dynamics(self, action):  # Dynamic of cubic polynomial on velocity
        # TODO: add steering
        # define the discrete time dynamical model

        def f_environment(x, u, dt): # x, y, theta, velocity
            sx, sy, vx, vy = x[0], x[1], x[2], x[3]
            if self.id == 0 or self.id == 1:
                vx_new = vx
                vy_new = vy + u * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
                if vy_new < self.min_speed:
                    vy_new = self.min_speed
                else:
                    vy_new = max(min(vy_new, self.max_speed), self.min_speed)
                sx_new = sx
                sy_new = sy + (vy + vy_new) * dt * 0.5
            else:
                vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
                vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
                sx_new = sx + (vx + vx_new) * dt * 0.5
                sy_new = sy + (vy + vy_new) * dt * 0.5
            print("ID:", self.id, "action:", u, "old vel:", vx, vy, "new vel:", vx_new, vy_new)
            return sx_new, sy_new, vx_new, vy_new

        # if self.env.name == "merger":
        #     self.state.append(f_environment(self.state[-1], action, self.sim.dt))

        # else:  # using dynamics defined in dynamics.py, for easier access
        #     self.state.append(dynamics.dynamics_1d(self.state[-1], action, self.sim.dt, self.min_speed, self.max_speed))
        # def f_environment(x, u, dt): # x, y, theta, velocity
        #     sx, sy, theta, vy = x[0], x[1], x[2], x[3]
        #     if self.id == 0 or self.id == 1:
        #         vy_new = vy + u[1] * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
        #         if vy_new < self.min_speed:
        #             vy_new = self.min_speed
        #         else:
        #             vy_new = max(min(vy_new, self.max_speed), self.min_speed)
        #         sx_new = sx + (vy + vy_new) * dt *np.cos(theta)
        #         sy_new = sy + (vy + vy_new) * dt *np.sin(theta)
        #         theta_new = theta + u[0]
        #     else:
        #         vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
        #         vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
        #         sx_new = sx + (vx + vx_new) * dt * 0.5
        #         sy_new = sy + (vy + vy_new) * dt * 0.5
        #         theta_new = theta + u[0]
        #     print("ID:", self.id, "action:", u[0],"," ,u[1], "old vel:", vy, "new vel:", vy_new, "angle", theta_new)
        #     return sx_new, sy_new, theta_new, vy_new
        # if self.env.name == "merger":
        #     self.state.append(f_environment(self.state[-1], action, self.sim.dt))
        # else:
            #self.state.append(f(self.state[-1], action, self.sim.dt))

        def f_environment_sc(x, u, dt): # x, y, heading, velocity steering, velocity 
            sx, sy, theta, delta, vy = x[0], x[1], x[2], x[3], x[4]
            L = 3 # length of the vehicle 
            if self.id == 0 or self.id == 1:
                vy_new = vy + u[1] * dt #* vy / (np.linalg.norm([vx, vy]) + 1e-12)
                delta_new = delta + u[0] * dt
                if vy_new < self.min_speed:
                    vy_new = self.min_speed
                else:
                    vy_new = max(min(vy_new, self.max_speed), self.min_speed)
                sx_new = sx + (vy_new) * dt *np.cos(theta)
                sy_new = sy + (vy_new) * dt *np.sin(theta)
                theta_new = theta + vy_new/L*np.tan(delta_new) *dt
            else:
                vx_new = vx + u * dt * vx #/ (np.linalg.norm([vx, vy]) + 1e-12)
                vy_new = vy + u * dt * vy #/ (np.linalg.norm([vx, vy]) + 1e-12)
                sx_new = sx + (vx + vx_new) * dt * 0.5
                sy_new = sy + (vy + vy_new) * dt * 0.5
                theta_new = theta + u[0]
            print("ID:", self.id, "action:", u[0],"," ,u[1], "old vel:", vy, "new vel:", vy_new, "angle", theta_new)
            return sx_new, sy_new, theta_new, delta_new, vy_new
        if self.env.name == "merger":
            self.state.append(f_environment_sc(self.state[-1], action, self.sim.dt))
        else:
            # self.state.append(f(self.state[-1], action, self.sim.dt))
            self.state.append(dynamics.dynamics_1d(self.state[-1], action, self.sim.dt, self.min_speed, self.max_speed))

        return

    def get_initial_belief(self, theta_h, theta_m, lambda_h, lambda_m, weight):
        """
        Obtain initial belief of the params
        :param theta_h:
        :param theta_m:
        :param lambda_h:
        :param lambda_m:
        :param weight:
        :return:
        """
        # TODO: given weights for certain param, calculate the joint distribution (p(theta_1), p(lambda_1) = 0.8, ...)
        theta_list = self.sim.theta_list
        lambda_list = self.sim.lambda_list
        beta_list = self.sim.beta_set

        if self.sim.inference_type[1] == 'empathetic':
            # beta_list = beta_list.flatten()
            belief = np.ones((len(beta_list), len(beta_list)))
            for i, beta_h in enumerate(beta_list):  # H: the rows
                for j, beta_m in enumerate(beta_list):  # M: the columns
                    if beta_h[0] == theta_h:  # check lambda
                        belief[i][j] *= weight
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_h[1] == lambda_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    if beta_m[0] == theta_m:  # check lambda
                        belief[i][j] *= weight
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                        if beta_m[1] == lambda_m:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)

                    # if beta_h == [lambda_h, theta_h] and beta_m == [lambda_m, theta_m]:
                    #     belief[i][j] = weight
                    # else:
                    #     belief[i][j] = 1

        # TODO: not in use! we only use the game theoretic inference
        else:  # get belief on H agent only
            belief = np.ones((len(lambda_list), len(theta_list)))
            for i, lamb in enumerate(lambda_list):
                for j, theta in enumerate(theta_list):
                    if lamb == lambda_h:  # check lambda
                        belief[i][j] *= weight
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
                    else:
                        belief[i][j] *= (1 - weight) / (len(lambda_list) - 1)
                        if theta == theta_h:  # check theta
                            belief[i][j] *= weight
                        else:
                            belief[i][j] *= (1 - weight) / (len(theta_list) - 1)
        # THIS SHOULD NOT NEED TO BE NORMALIZED!
        # print(belief, np.sum(belief))
        assert round(np.sum(belief)) == 1
        return belief


# dummy class
class dummy(object):
    pass


if __name__ == '__main__':
    Car1 = {"initial_state": [[200, 200, 0, 0, 0]], "par": 1, "initial_action":1}
    from environment import *
    env = Environment("merger")
    sim = dummy()
    sim.dt = 1
    #def __init__(self, sim, env, par, inference_model, decision_model, i):
    test_car = AutonomousVehicle(sim, env, Car1, "baseline", "baseline", 0)
    i = 1
    while i<20:
        test_car.dynamics([1, 1])
        i+= 1


