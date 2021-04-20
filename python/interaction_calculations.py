import time
import math
import pickle
import numpy as np
import concurrent.futures

from utils import dataset_reader
from utils import dataset_types
from utils import dict_utils


DISTANCE_TH = 10.0   # Distance threshold that classifies an interaction

def init_active_cars(active_cars, iter, timestamp_min, track_dictionary):
    for key, track in dict_utils.get_item_iterator(track_dictionary):
        if track.time_stamp_ms_first == timestamp_min:
            active_cars.append(track)

def distance_formula(car1, car2, timestamp):
    try:
        x1 = car1.motion_states[timestamp].x
        y1 = car1.motion_states[timestamp].y
        x2 = car2.motion_states[timestamp].x
        y2 = car2.motion_states[timestamp].y
        return np.sqrt((x1-x2)**2+(y1-y2)**2)
    except Exception as e:
        return 999999999

def update_active_cars(active_cars, curr_car_id, car_iter, track_dictionary,timestamp):
    car = track_dictionary[curr_car_id[0]]
    # DEBUG print("Current active car: %d" % (car.track_id))
    for track in active_cars:
        if track.time_stamp_ms_last < timestamp:
            active_cars.remove(track)

    while car.time_stamp_ms_first == timestamp:
        active_cars.append(car)
        try:
            curr_car_id = next(car_iter)
        except Exception as e:
            break
        car = track_dictionary[curr_car_id[0]]
        # DEBUG print("Looking at next car: %d" % (car.track_id))
    
    # DEBUG print("updated active cars: %d" % (len(active_cars)))
    return curr_car_id, active_cars

def add_interaction(track1, track2, traj_dictionary, timestamp):
    if track1 in traj_dictionary[track2].interact_with.keys():
        interaction_list = traj_dictionary[track2].interact_with[track1]
    else:
        interaction_list = []
    interaction_list.append(timestamp)
    traj_dictionary[track2].interact_with[track1] = interaction_list



def check_distances(active_cars, traj_dictionary, timestamp):
    # DEBUG print("CHECKING DISTANCES: length %d" % (len(active_cars)))
    for i in range(len(active_cars)-1):
        car1 = active_cars[i]
        for j in range(i+1, len(active_cars)):
            car2 = active_cars[j]
            d = distance_formula(car1, car2, timestamp)
            # DEBUG print("--Comparing %d and %d at distance: %f" % (car1.track_id, car2.track_id, d))
            if d <= DISTANCE_TH:
                traj_dictionary[car1.track_id].interaction = True
                traj_dictionary[car2.track_id].interaction = True
                add_interaction(car1.track_id, car2.track_id, traj_dictionary, timestamp)
                add_interaction(car2.track_id, car1.track_id, traj_dictionary, timestamp)




def calculate_interactions(track_dictionary, traj_dictionary):
    timestamp_min = 1e9
    timestamp_max = 0
    timestamp = 0
    active_cars = []
    curr_car_id = None

    # Get min and max timestamps
    if track_dictionary is not None:                                        
        for key, track in dict_utils.get_item_iterator(track_dictionary):
            timestamp_min = min(timestamp_min, track.time_stamp_ms_first)
            timestamp_max = max(timestamp_max, track.time_stamp_ms_last) 
    # DEBUG print("Max: %f\nMin: %f" % (timestamp_max, timestamp_min))
    # initialize
    init_active_cars(active_cars, track_dictionary, timestamp_min, track_dictionary)
    car_iter = dict_utils.get_item_iterator(track_dictionary)
    
    # move iterator past active cars 
    for x in range(len(active_cars)+1):
       curr_car = next(car_iter)

    # run calculation loop
    timestamp = timestamp_min
    while timestamp <= timestamp_max:
        # DEBUG print("Time: %d" % (timestamp))
        check_distances(active_cars, traj_dictionary, timestamp)
        timestamp += 100
        curr_car, active_cars = update_active_cars(active_cars, curr_car, car_iter, track_dictionary, timestamp)

    return traj_dictionary
    # return traj dict? or what
    

# FOR TESTING ONLY 
if __name__ == "__main__":
    track_file_name = "../recorded_trackfiles/Scenario4/vehicle_tracks_001.csv"
    track_dictionary = dataset_reader.read_tracks(track_file_name)
    
    file_path = "../trajectory_files/Scenario4/vehicle_tracks_001_trajs.pickle"
    with open(file_path, 'rb') as handle:
            traj_dict = pickle.load(handle)

    for key,val in traj_dict.items():
        if not hasattr(val, 'interact_with'):
            val.interact_with = {}

    traj_dict = calculate_interactions(track_dictionary, traj_dict)

    output_file_path = "../trajectory_files/Scenario4/vehicle_tracks_001_trajs_int.pickle"
    #output_file_path = "../trajectory_files/Scenario4/vehicle_tracks_001_trajs_active.pickle"
    with open(output_file_path, 'wb') as handle:
        pickle.dump(traj_dict, handle, pickle.HIGHEST_PROTOCOL)

    print(len(traj_dict))

    for key, val in traj_dict.items():
        print("Track ", key, ": ", val.interact_with)
