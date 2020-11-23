# Folder structure

For every scenario with name `<scenario_name>`, the scripts expect the map with name `<scenario_name>.osm` to be in folder `maps` and the track files to be in a subfolder with name `<scenario_name>` within folder `recorded_trackfiles`.

```
.
├── README.md
├── doc
│   └── ...
├── python
│   └── ...
├── maps
│   ├── metadata_dict
│   ├── Scenario1.osm
│   └── Scenario2.osm
├── trajectory_files
│   ├── Scenario1
│   │   ├── vehicle_tracks_000_trajs.pickle (generated file)
│   │   ├── vehicle_tracks_001_trajs.pickle (generated file)
│   │   └── ...
│   └── Scenario2
│       ├── vehicle_tracks_000_trajs.pickle (generated file)
│       └── vehicle_tracks_000_trajs.pickle (generated file)
│       └── ...
└── recorded_trackfiles
		├── validation-set-list_INTERACTION-dataset_v1.txt 
    ├── Scenario1
    │   ├── vehicle_tracks_000.csv
    │   └── pedestrian_tracks_000.csv
    		└── ...
    └── Scenario2
        ├── vehicle_tracks_000.csv
        └── pedestrian_tracks_000.csv
        └── ...

```
