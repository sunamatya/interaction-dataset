#!/usr/bin/env python

'''
    Create metadata dictionary 
'''

import pickle 
from utils import boundary_boxes as bboxes 
from utils import dataset_types 

filepath = "../maps/metadata_dict.pickle"
meta_dict = dict()

'''
* Scenario1  := DR_CHN_Roundabout_LN
* Scenario2  := DR_DEU_Roundabout_OF
* Scenario3  := DR_USA_Roundabout_EP
* Scenario4  := DR_USA_Roundabout_FT
* Scenario5  := DR_USA_Roundabout_SR
'''

# Scenario1 = DR_CHN_Merging_ZS
map_data                = dataset_types.MapMeta(1)
map_data.scenario_name  = "DR_CHN_Roundabout_LN"
map_data.entrances      = bboxes.N1 
map_data.exits          = bboxes.X1
map_data.xlim           = bboxes.xlim1
map_data.ylim           = bboxes.ylim1
meta_dict["Scenario1"]  = map_data

# Scenario2 = DR_DEU_Roundabout_OF
map_data                = dataset_types.MapMeta(2)
map_data.scenario_name  = "DR_DEU_Roundabout_OF"
map_data.enterances     = bboxes.N2 
map_data.exits          = bboxes.X2
map_data.xlim           = bboxes.xlim2
map_data.ylim           = bboxes.ylim2
meta_dict["Scenario2"]  = map_data

# Scenario3 = DR_USA_Roundabout_EP
map_data                = dataset_types.MapMeta(3)
map_data.scenario_name  = "DR_USA_Roundabout_EP"
map_data.enterances     = bboxes.N3 
map_data.exits          = bboxes.X3
map_data.xlim           = bboxes.xlim3
map_data.ylim           = bboxes.ylim3
meta_dict["Scenario3"]  = map_data

# Scenario4 = DR_USA_Roundabout_FT
map_data                = dataset_types.MapMeta(4)
map_data.scenario_name  = "DR_USA_Roundabout_FT"
map_data.enterances     = bboxes.N4
map_data.exits          = bboxes.X4
map_data.xlim           = bboxes.xlim4
map_data.ylim           = bboxes.ylim4
meta_dict["Scenario4"]  = map_data

# Scenario4 = DR_USA_Roundabout_SR
map_data                = dataset_types.MapMeta(5)
map_data.scenario_name  = "DR_USA_Roundabout_SR"
map_data.enterances     = bboxes.N5
map_data.exits          = bboxes.X5
map_data.xlim           = bboxes.xlim5
map_data.ylim           = bboxes.ylim5
meta_dict["Scenario5"]  = map_data


with open(filepath, 'wb') as handle:
    pickle.dump(meta_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)