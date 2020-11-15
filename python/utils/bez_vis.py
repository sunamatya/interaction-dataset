#!/usr/bin/env python

import argparse
import os
import sys
import pickle 
import bezier
import inspect
import matplotlib.pyplot as plt


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import utils 

def get_traj_file_path(file_number, scenario_name):
    # Create save path for trajectory dict 
    error_string = ""
    traj_dir = "../../trajectory_files"
    maps_dir = "../maps"
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("traj_file_number", type=int, help="Number of the track file (int)", nargs="?")
    parser.add_argument("track_id", type=int, help="Number of the track file (int)", nargs="?")
    parser.add_argument('-a', '--all', action='store_true', help="Iterate through all track files")
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if args.traj_file_number is None:
        raise IOError("You must specify a file number. Type --help for help.")

    file_path = get_traj_file_path(args.traj_file_number, args.scenario_name)
    traj_dict = load_trajs(file_path)   

    bez = traj_dict[args.track_id].traj_bez
    xvals, yvals = bezier.bezier_curve(bez, nTimes=1000)
    plt.plot(xvals,yvals)
    plt.show()




main()