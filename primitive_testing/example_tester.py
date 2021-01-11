import numpy as np
import matplotlib.pyplot as plt
# containter for the data

def generate_demo_traj():
    big_x_traj = []
    big_y_traj = []
    big_vx_traj = []
    big_vy_traj = []
    big_psi_traj = []
    psi = np.linspace(-np.pi / 4, np.pi / 4, 15)
    ts = 0.1
    v = 1.0
    L = 4
    # x = 0
    # y = 0
    # theta = 0
    # demonstaration of the data
    for demo_psi in range(len(psi)):
        x = 0
        y = 0
        theta = 0
        x_traj = []
        y_traj = []
        vx_traj = []
        vy_traj = []
        psi_traj = []

        for t in range(100):
            #print(psi[demo_psi])
            theta_dot = (v/L) * np.tan(psi[demo_psi])
            theta = theta + ts* theta_dot
            x_dot = v* np.cos(theta)
            y_dot = v* np.sin(theta)
            x = x + ts * x_dot
            y = y + ts* y_dot

            x_traj.append(x)
            y_traj.append(y)
            vx_traj.append(x_dot)
            vy_traj.append(y_dot)
            psi_traj.append(theta)

        big_x_traj.append(np.array(x_traj))
        big_y_traj.append(np.array(y_traj))
        big_vx_traj.append(np.array(vx_traj))
        big_vy_traj.append(np.array(vy_traj))
        big_psi_traj.append(np.array(psi_traj))

        #
        # plt.figure(1)
        # plt.plot(x_traj, y_traj)
        x = 0
        y = 0
        theta = 0
        x_traj = []
        y_traj = []
        vx_traj = []
        vy_traj = []
        psi_traj = []

        for t in range(100):
            #print(psi[demo_psi])
            theta_dot = -(v/L) * np.tan(psi[demo_psi])
            theta = theta + ts* theta_dot
            x_dot = -v* np.cos(theta)
            y_dot = - v* np.sin(theta)
            x = x + ts * x_dot
            y = y + ts* y_dot

            x_traj.append(x)
            y_traj.append(y)
            vx_traj.append(x_dot)
            vy_traj.append(y_dot)
            psi_traj.append(theta)

        # plt.plot(x_traj, y_traj)
        big_x_traj.append(np.array(x_traj))
        big_y_traj.append(np.array(y_traj))
        big_vx_traj.append(np.array(vx_traj))
        big_vy_traj.append(np.array(vy_traj))
        big_psi_traj.append(np.array(psi_traj))

    # #
    # plt.title('tester_plot')
    # plt.show()


    return(big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)

all_x, all_y, all_vx, all_vy, all_psi= generate_demo_traj()


