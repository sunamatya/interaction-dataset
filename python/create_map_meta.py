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
map_data.entrances     = bboxes.N1 
map_data.exits          = bboxes.X1
map_data.xlim           = bboxes.xlim1
map_data.ylim           = bboxes.ylim1
meta_dict["Scenario1"]  = map_data

'''
# Scenario2 = DR_CHN_Merging_ZS
map_data                = MapMeta(2)
map_data.scenario_name  = "DR_CHN_Roundabout_LN"
map_data.enterances     = bboxes.N2 
map_data.exits          = bboxes.X2
map_data.xlim           = bboxes.xlim2
map_data.ylim           = bboxes.ylim2
meta_dict["Scenario2"]  = map_data
'''

with open(filepath, 'wb') as handle:
    pickle.dump(meta_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)