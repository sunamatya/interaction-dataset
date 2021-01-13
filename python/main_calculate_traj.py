#!/usr/bin/env python

import argparse
import os
import sys
import time
import math
import pickle
import operator
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np

from interaction_calculations import calculate_interactions
from utils import dataset_reader
from utils import dataset_types
from utils import map_vis_lanelet2
from utils import tracks_vis
from utils import dict_utils
from utils import bezier

global SCENARIO
SCENARIO = "test"
Traj =  dataset_types.Traj
MapMeta = dataset_types.MapMeta
N1 = [[1009.6,1015.8],[1014.6,1021.9]]
X4 = [[971.3,991.6],[978.2,997.0]]
with open("../maps/metadata_dict.pickle", 'rb') as handle:
    map_meta_dict = pickle.load(handle,fix_imports=True)


def update_progress(progress, id):
    print("\r Curr_id: %s Total Progress: [%10s] %0.2f percent" % (str(id), ("#"*math.floor((progress*10))), progress*100), end='')

def line_through_box(xpoints, ypoints, box):
    # Needs major edits :: for testing purposes only
    pt1 = box[0]
    pt2 = box[1]

    xRange = [min(pt1[0],pt2[0]), max(pt1[0],pt2[0])]
    yRange = [min(pt1[1],pt2[1]), max(pt1[1],pt2[1])]

    N=len(xpoints)
    for i in range(N):
        if (xRange[0] < xpoints[i] and xpoints[i] < xRange[1]):
            if (yRange[0] < ypoints[i] and ypoints[i] < yRange[1]):
                return True 

    return False

def get_boundary_ids(xpoints, ypoints, scenario_name):
    meta = map_meta_dict[scenario_name]
    entrance_id = None
    exit_id = None
    for i in range(len(meta.entrances)):
        if line_through_box(xpoints, ypoints, meta.entrances[i]):
            entrance_id = i
            break
    for j in range(len(meta.exits)):
        if line_through_box(xpoints, ypoints, meta.exits[j]):
            exit_id = j
            break
    
    return entrance_id, exit_id

def get_track_file_path(file_number, scenario_name):
    # Return track file associated 
    tracks_dir = "../recorded_trackfiles"
    scenario_dir = tracks_dir + "/" + scenario_name
    track_file_prefix = "vehicle_tracks_"
    track_file_ending = ".csv"
    track_file_name = scenario_dir + "/" + track_file_prefix + str(file_number).zfill(3) + track_file_ending

    return track_file_name


def get_traj_file_path(file_number, scenario_name):
    # Create save path for trajectory dict 
    error_string = ""
    traj_dir = "../trajectory_files"
    maps_dir = "../maps"
    scenario_dir = traj_dir + "/" + scenario_name
    traj_file_prefix = "vehicle_tracks_"
    traj_file_ending = "_trajs.pickle"
    traj_file_name = scenario_dir + "/" + traj_file_prefix + str(file_number).zfill(3) + traj_file_ending
    if not os.path.isdir(traj_dir):
        error_string += "Did not find traj file directory \"" + traj_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        try:
            print("Creating new scenario directory at: " + scenario_dir)
            os.makedirs(scenario_dir)
        except OSError as e:
            if e.errno != errno.EXIST:
                raise
        #error_string += "Did not find scenario directory \"" + scenario_dir + "\"\n"
    if error_string != "":
        error_string += "Type --help for help."
        raise IOError(error_string)

    return traj_file_name


def get_track_dict(file_number, scenario_name):
    # check folders and files
    error_string = ""
    track_file_name = get_track_file_path(file_number, scenario_name)
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

 
def calculate_traj(car):
    # worker to calculate trajectory
    xy_points   = [[],[]]
    velocities  = [[],[]]
    curr_traj   = Traj(car)
    for state in car.motion_states:
        curr_traj.x_vals = np.append(curr_traj.x_vals, car.motion_states[state].x)
        curr_traj.y_vals = np.append(curr_traj.y_vals, car.motion_states[state].y)
        curr_traj.vx_vals = np.append(curr_traj.vx_vals, car.motion_states[state].vx)
        curr_traj.vy_vals = np.append(curr_traj.vy_vals, car.motion_states[state].vy)
        curr_traj.psi_vals = np.append(curr_traj.vy_vals, car.motion_states[state].psi_rad)

    err, curr_traj.traj_bez = bezier.bezier_points([curr_traj.x_vals, curr_traj.y_vals])
    curr_traj.bez_xvals, curr_traj.bez_yvals = bezier.bezier_curve(curr_traj.traj_bez)
    if err < 0:
        curr_traj.error = True
    
    print("...Car %d finished" % (car.track_id))
    return curr_traj


def calc_file_traj(file_number, scenario_name, recalculate):
    ''' 
        Args:
            file_number   - (int) vehicle_tracks file number 
            scenario_name - (str) 'Scenario#' where # is the scenario number
        
        Output:
            No output -- saves traj's in pickle file in ../trajectory_files/scenario_name/
            vehicle_tracks_000.csv --> vehicle_tracks_000_trajs.pickle
    '''
    track_dict  = get_track_dict(file_number, scenario_name)
    traj_file   = get_traj_file_path(file_number, scenario_name)
    traj_dict   = dict()
    tik         = time.time()

    if os.path.isfile(traj_file): 
        if not recalculate:
            print("[Traj] traj file already exists")
            return

    print("Calculating Trajectories for vehicle_track_%s" % (str(file_number).zfill(3)))
    # calculate all trajectories from a vehicle_track_file with multi-processes
    executor = concurrent.futures.ProcessPoolExecutor(10)
    futures = [executor.submit(calculate_traj, car) for car in track_dict.values()]
    concurrent.futures.wait(futures)

    for fut in futures:
        traj_Obj = fut.result()
        traj_Obj.entrance_id, traj_Obj.exit_id = get_boundary_ids(traj_Obj.bez_xvals, traj_Obj.bez_yvals, scenario_name)
        #print("id: " + str(traj_Obj.track_id) +", N: "+str(traj_Obj.entrance_id)+", X: "+str(traj_Obj.exit_id)+"\n")
        traj_dict[traj_Obj.track_id] = traj_Obj
    
    print("Calculating Interactions")
    traj_dict = calculate_interactions(track_dict, traj_dict)
 
    tok          = time.time()
    elapsed_time = tok - tik
    print("Trajectory calculation complete in %f seconds" % (elapsed_time))
    print("Saving trajectories...")
    save_traj_file(traj_dict, traj_file)



def save_traj_file(traj_dict, file_name):
    '''
        Save trajectory file in pickle format 
    '''
    if not file_name.endswith('.pickle'):
        IOError("Error -- file name not a .pickle file in save_traj_file\n")
    with open(file_name, 'wb') as handle:
        pickle.dump(traj_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    # provide data to be visualized
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", nargs="?")
    parser.add_argument('-a', '--all', action='store_true', help="Iterate through all track files")  
    parser.add_argument('-r', '--recalculate', action='store_true', help="Recalculate the traj file if it already exists")
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if not args.all and args.track_file_number is None:
        raise IOError("You must specify a track number or --all. Type --help for help") 
    if args.all and args.track_file_number is not None:
        raise IOError("You cannot use -a/--all with a specific track number. Type --help for help") 
    SCENARIO = args.scenario_name
    
    if args.all:    
        # iterate through all trajectory files
        file_number = 0
        track = get_track_file_path(file_number, args.scenario_name)
        while os.path.isfile(track):
            calc_file_traj(file_number, args.scenario_name, args.recalculate)
            file_number+=1
            track = get_track_file_path(file_number, args.scenario_name)
    else:
        #calculate specific traj file
        calc_file_traj(args.track_file_number, args.scenario_name, args.recalculate)


