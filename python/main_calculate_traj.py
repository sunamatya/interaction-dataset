#!/usr/bin/env python

import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from utils import dataset_reader
from utils import dataset_types
from utils import map_vis_lanelet2
from utils import tracks_vis
from utils import dict_utils
from utils import bezier



def get_track_dict(file_number, scenario_name):
    # check folders and files
    error_string = ""
    tracks_dir = "../recorded_trackfiles"
    scenario_dir = tracks_dir + "/" + scenario_name
    track_file_prefix = "vehicle_tracks_"
    track_file_ending = ".csv"
    track_file_name = scenario_dir + "/" + track_file_prefix + str(file_number).zfill(3) + track_file_ending
    if not os.path.isdir(tracks_dir):
        error_string += "Did not find track file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if not os.path.isfile(track_file_name):
        error_string += "Did not find track file \"" + track_file_name + "\"\n"
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)

    # load the tracks
    print("Loading track %d..." % file_number)
    track_dictionary = None
    pedestrian_dictionary = None
    track_dictionary = dataset_reader.read_tracks(track_file_name)

    return track_dictionary



def calc_file_traj(file_number, scenario_name):

    track_dict = get_track_dict(file_number, scenario_name)
    xy_points = [[],[]]
    for car in track_dict.values():
        if car.track_id == 4:
            for state in car.motion_states:
                xy_points[0] = np.append(xy_points[0], car.motion_states[state].x)
                xy_points[1] = np.append(xy_points[1], car.motion_states[state].y)
            bez = bezier.bezier_points(xy_points,5)
            return bez

                     




if __name__ == "__main__":

    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", nargs="?")
    parser.add_argument('-a', '--all', action='store_true', help="Iterate through all track files")  
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if not args.all and args.track_file_number is None:
        raise IOError("You must specify a track number or --all. Type --help for help") 
    if args.all and args.track_file_number is not None:
        raise IOError("You cannot use -a/--all with a specific track number. Type --help for help") 
    
    
    if args.all:
        args.track_file_number = 0
        # iterate through all trajectory files
        # NOT BUILT
        calc_file_traj(args.track_file_number, args.scenario_name)

    else:
        calc_file_traj(args.track_file_number, args.scenario_name)

   

