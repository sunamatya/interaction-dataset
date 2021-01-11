import numpy as np
import matplotlib.pyplot as plt

import sys
import csv
sys.path.append('C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset/data')
def make_demonstrations():
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
            if row[0]== 17: # vehicle 2 data for testing
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

x, y, vx, vy, psi= make_demonstrations()
plot_figure = True
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
