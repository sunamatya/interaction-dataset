import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy.interpolate
import csv
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

if __name__ == "__main__":
    #interacting_all = get_primitive_info()
    #track_file_name = open('primitive_traj1.csv', 'r')
    #file = csv.DictReader(track_file_name)
    traj = []
    with open("primitive_traj1.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
    #traj = []
        for col in csv_reader:
            print(col[0])
            traj.append(col[0])

    traj_len = int(len(traj)/5)
    D = 5
    plt.plot(traj[0:traj_len*D:D],traj[1:traj_len*D:D])
    plt.show()