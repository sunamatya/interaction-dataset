[TOC]

# Python Scripts for the INTERACTION dataset and the INTERPRET Competition

* these scripts assist you to visualize the INTERACTION dataset and to process the INTERACTION Dataset into segments that can be directly used for training, validation and test in the INTERPRET challenge.
* for details about and access to the dataset, visit https://interaction-dataset.com/
* for details about the INTERPRET challenge, visit http://challenge.interaction-dataset.com/prediction-challenge/intro
* If you want to visualize the INTERACTION dataset, please refer to the github at https://github.com/interaction-dataset/interaction-dataset
*  `python3` is supported

## Required Python Packages
* `csv`: for reading the csv track files
* to work with the map:
  * either `lanelet2` for most convenient map usage
    * see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details
  * or
    * `pyproj`
    * `xml`
* `numpy`, `os`,  `sys`,  `functools`, `shutil` and`argparse` for processing the data
* `math`, `numpy`,`time` and `matplotlib` for visualizing the scenarios
* use `test_imports.py` to test whether you have all necessary packages installed

## Dataset Trajectory Calculation
* **SEE FOLDER STRUCTURE BEFORE CALCULATING TRAJECTORIES** : [link](doc/folder-structure.md)
* Each scenario corresponds to a specific number (e.x. DR_CHN_Merging_ZS == `Scenario7`)
  * copy/download the correct data into the correct folders according to [scenario-definitions.md](docs/scenario-definitions.md)
* Trajectories for each car/scenario can be calculated using a polynomial fit
* Calculated trajectories are saved in `trajectory_files/<scenario_name>/<vehicle_tracks_<XXX>_trajs.pickle>`
*If the trajectory file already exists it will not be recalculated to minimize calculation time
  * can be overriden by the using the `-r` flag
### Important
  * Calculations use manually defined map metadata: initialize the metadata file
    * run `./create_map_meta` from folder `python` before calculating/visualizing anything
### Usage
* to calculate a specific track file's trajectories
  * run `./main_calculate_traj.py <scenario_name> <trackfile_number>` from folder `python` to calculate trackfile trajectories
* to calculate all trajectory files
  * run `./main_calculate_traj.py <scenario_name> --all` from folder `python` to calculate scenario trajectories
  * run `./main_calculate_traj.py <scenario_name> --all -r` from folder `python` to recalculate scenario trajectories
    * recalculate all scenario trajectory files

## Dataset Visualization

### Usage 

* copy/download the INTERACTION drone data into the right place
  * copy/download the track files into the folder `recorded_trackfiles`, keep one folder per scenario, as in your download
  * copy/download the maps into the folder `maps`
  * your folder structure should look like in [folder-structure.md](doc/folder-structure.md)
* to visualize the data
  * run `./main_visualize_data.py <scenario_name> <trackfile_number (default=0)>` from folder `python` to visualize scenarios
* if you only want to load and work with the track files
  * run `./main_load_track_file.py <tracks_filename>` from folder `python` to load tracks
* if you want to show trajectories use the `--traj` flag
  * run `./main_visualize_data.py <scenario_name> <trackfile_number (default=0)> --traj`
* if you want to restrict trajectories to cars between certain enterances/exists use --enter or --exit 
  * run `./main_visualize_data.py <scenario_name> <trackfile_number (default=0)> --traj --enter <int,int,...> --exit <int,int,...>`
  * omitting --enter or --exit sets the value to all entrances/exits.  
  * ex. Show trajectories that only enter from entrance 3
    * run `./main_visualize_data.py <scenario_name> <trackfile_number (default=0)> --traj --enter 3`

### Trajectory Visualization

* if you want to view ONLY the trajectories in more detail
  * run `./bez_vis.py <scenario_name> <trackfile_number (default=0)>` from the `utils` folder
* entrance and exit restrictions can be done using the same syntax as before
  * run `./bez_vis.py <scenario_name> <trackfile_number (default=0)> --enter <int,int,...> --exit <int,int,...>`
* if you want to view ONE car trajectory only use `--id`
  * run `./bez_vis.py <scenario_name> <trackfile_number (default=0)> --id <car_id>`
  * use the `--points` flag to show the original dataset points on top of the car's trajectory
    * run `./bez_vis.py <scenario_name> <trackfile_number (default=0)> --id <car_id> --points`
### Test Usage without Dataset

* to test the visualization
  * run `./main_visualize_data.py .TestScenarioForScripts`
* to test loading the data
  * run `./main_load_track_file.py ../recorded_trackfiles/.TestScenarioForScripts/vehicle_tracks_000.csv`

## Prepare Training Dataset for the INTERPRET Challenge

- copy/download the INTERACTION drone data into the right place
  - copy/download the track files into the folder `recorded_trackfiles`, keep one folder per scenario, as in your download
  - copy/download the maps into the folder `maps`
  - copy/download the suggested split .txt file ( validation-set-list_INTERACTION-dataset_v1.txt ) for the train and validation set into the folder `recorded_trackfiles`
  - your folder structure should look like in [folder-structure.md](doc/folder-structure.md)
- to split all track files into train and validation subfolders
  - run `./split_train_val_script.py <path_to_the_suggested_split_txt_file> <path_to_the_recorded_trackfiles> ` from folder `python` to split the trackfiles in the corresponding scenarios into two subfolders  `train`  and  `val` . (Internal note: the original csv files in the folder will be moved into the two newly created subfolders)
- to prepare the data into segments that can be directly utilized for the INTERPRET prediction
  - run `./segment_data.py <options> <path_to_folder_or_file_from_scenario_folder> number_of_frames_in_segments gap_frames_between_segments` from folder `python`
    - If users choose  `option=default`,  all trackfiles in both the  `train`  and  `val` subfolders in all scenarios will be segmented. The segmented csv files will be saved in newly created subfolders  `segmented` in corresponding train and validation folders. In the meanwhile, a folder `sorted` will also be generated in the  `train`  and  `val` subfolders which contains the csv files that are sorted based on the frame_id instead of the track_id in the file.
    - If users choose  `option=dir `, and specify the  <path_to_folder_or_file_from_scenario_folder> as a folder, then all csv files under this folder will be segmented, saved in a newly created subfolder `segmented`. If users choose  `option=file `, and specify ` <path_to_folder_or_file_from_scenario_folder>` as a .csv file, then the csv file will be segmented, saved in the subfolder  `segmented`.
    - All the segmented track files will follow the names of the original csv files.
- to split all track files and prepare all data into segments
  - from the  `python` folder, run `./split_train_val_script.py <path_to_the_suggested_split_txt_file> <path_to_the_recorded_trackfiles> `
  - from the `python` folder, run `./segment_data.py default <path_to_recorded_track_files> number_of_frames_in_segments gap_frames_between_segments` to split all the train and validation sets for all scenarios and segment all track files.  The first parameter `number_of_frames_in_segments ` determines how many frames (including both historical and predicted frames) are contained in each training case, and users can decide the length of historical frames and predicted frames as they need. The second ` gap_frames_between_segments` defines the gap between two successive test cases with a same  ` agent`. Typically, you should set   ` gap_frames_between_segments > number_of_frames_in_segments `  to avoid overlaps between different test cases. Moreover, in the segmented files, there is one extra column compared to the original track files, i.e., the  ` agent_role` column. If ` agent_role="agent"`, then that agent will be the one whose future trajectories need to be predicted, and all surronding vehicles will be marked as ` others`.

Note: The segmentation process will take a while, so please be patient.



