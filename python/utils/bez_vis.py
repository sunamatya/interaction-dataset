#!/usr/bin/env python

import argparse
import os
import sys
import pickle
import numpy as np
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pickle 
from utils import bezier
from utils import dataset_reader
import inspect
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import utils 



def get_track_file_path(file_number, scenario_name):
    # Create save path for trackectory dict 
    error_string = ""
    if __name__ == "__main__":
        track_dir = "../../recorded_trackfiles"
    else:
        track_dir = "../recorded_trackfiles"
    scenario_dir = track_dir + "/" + scenario_name
    track_file_prefix = "vehicle_tracks_"
    track_file_ending = ".csv"
    track_file_name = scenario_dir + "/" + track_file_prefix + str(file_number).zfill(3) + track_file_ending
    if not os.path.isdir(track_dir):
        error_string += "Did not find track file directory \"" + track_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)

    return track_file_name

def get_traj_file_path(file_number, scenario_name):
    # Create save path for trajectory dict 
    error_string = ""
    if __name__ == "__main__":
        traj_dir = "../../trajectory_files"
    else:
        traj_dir = "../trajectory_files"
    scenario_dir = traj_dir + "/" + scenario_name
    traj_file_prefix = "vehicle_tracks_"
    traj_file_ending = "_trajs.pickle"
    traj_file_name = scenario_dir + "/" + traj_file_prefix + str(file_number).zfill(3) + traj_file_ending
    if not os.path.isdir(traj_dir):
        error_string += "Did not find traj file directory \"" + traj_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)

    return traj_file_name

def load_trajs(file_name):
    with open(file_name, 'rb') as handle:
        tracks = pickle.load(handle)
    return tracks



if __name__ == "__main__":
    with open("../../maps/metadata_dict.pickle", 'rb') as handle:
        map_meta_dict = pickle.load(handle,fix_imports=True)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("traj_file_number", type=int, help="Number of the track file (int)", nargs="?")
    parser.add_argument("--id", type=int, help="Number of the track file (int)", default=[-1], nargs=1)
    parser.add_argument('-p',"--points", action='store_true', help="Show original points from the dataset")
    parser.add_argument('-a', '--all', action='store_true', help="Iterate through all track files") # not implemented
    parser.add_argument("--enter", nargs="+", type=int, help="Specify which entrances specifically", default=[])
    parser.add_argument("--exit", nargs="+", type=int,help="Specify which entrances specifically", default=[])
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if args.traj_file_number is None:
        raise IOError("You must specify a file number. Type --help for help.")

    file_path = get_traj_file_path(args.traj_file_number, args.scenario_name)
    traj_dict = load_trajs(file_path)   
    map_meta  = map_meta_dict[args.scenario_name]

    if args.id[0] < 0:
        # Multiple trajectories
        # clean data 
        if args.enter == []:
            enter = [i for i in range(len(map_meta.entrances))]
        else:
            enter = args.enter
        if args.exit == []:
            exits = [i for i in range(len(map_meta.exits))]
        else:
            exits = args.exit
        # Plot values 
        for car in traj_dict.values():
                if not car.error:
                    if (car.entrance_id in enter) and (car.exit_id in exits):
                        txvals, tyvals = bezier.bezier_curve(car.traj_bez)
                        txvals = txvals[30:975]
                        tyvals = tyvals[30:975]
                        plt.plot(txvals, tyvals, "red", linewidth=1.8, alpha=.08)
    else:
        print("plotting car %d trajectory..." % (args.id[0]))
        # Show single trajectory with one
        if args.points:
            # load original data and plot points
            print("loading original points...")
            xy_points= [[],[]]
            track_file_name = get_track_file_path(args.traj_file_number, args.scenario_name)
            track_dictionary = dataset_reader.read_tracks(track_file_name)

            try:
                car = track_dictionary[args.id[0]]
            except KeyError:
                print("Error: Invalid car id -- car %d not found" % (args.id[0])) 
                exit()

            for state in car.motion_states:
                xy_points[0] = np.append(xy_points[0], car.motion_states[state].x)
                xy_points[1] = np.append(xy_points[1], car.motion_states[state].y)
            plt.plot(xy_points[0],xy_points[1],'bo', markersize=2)
        
        # plot car traj
        try:
            txvals, tyvals = bezier.bezier_curve(traj_dict[args.id[0]].traj_bez)
            txvals         = txvals[30:975]
            tyvals         = tyvals[30:975]
        except KeyError:
            print("Error: Invalid car id -- car %d not found" % (args.id[0])) 
            exit()
        plt.plot(txvals, tyvals, "red", linewidth=1.8)
    
    plt.xlim(min(map_meta.xlim),max(map_meta.xlim))
    plt.ylim(min(map_meta.ylim),max(map_meta.ylim))
    plt.show()


