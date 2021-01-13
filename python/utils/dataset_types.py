#!/usr/bin/env python


DELTA_TIMESTAMP_MS = 100  # similar throughout the whole dataset

class MapMeta:
    def __init__(self, id):
        self.scenario_id = id        # int
        self.scenario_name = None    # str 
        self.entrances = None        # list of list of tuples
        self.exits = None            # list of list of tuples
        self.xlim = None             # tuple
        self.ylim = None             # tuple
        
class Traj:
    def __init__(self, car):
        # assert isinstance(id, int)
        self.track_id = car.track_id
        self.agent_type = car.agent_type
        self.length = car.length
        self.width = car.width
        self.time_stamp_ms_first = car.time_stamp_ms_first
        self.time_stamp_ms_last = car.time_stamp_ms_last
        self.interaction = False
        self.error = False
        self.entrance_id = None
        self.exit_id = None 
        self.traj_bez = None
        self.bez_xvals = None
        self.bez_yvals = None
        self.x_vals = []
        self.y_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.psi_vals = []
        # self.dx   --deviation in x    

    def __str__(self):
        string = "Track: track_id=" + str(self.track_id) + ", agent_type=" + str(self.agent_type) + \
               ", length=" + str(self.length) + ", width=" + str(self.width) + \
               ", interaction=" + str(self.interaction) + ", bezier_point_count=" + str(len(self.traj_bez)) + \
               ", entrance_id=" +str(self.entrance_id) + ", exit_id=" + str(self.exit_id) + "\n"

class MotionState:
    def __init__(self, time_stamp_ms):
        assert isinstance(time_stamp_ms, int)
        self.time_stamp_ms = time_stamp_ms
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.psi_rad = None

    def __str__(self):
        return "MotionState: " + str(self.__dict__)


class Track:
    def __init__(self, id):
        # assert isinstance(id, int)
        self.track_id = id
        self.agent_type = None
        self.length = None
        self.width = None
        self.time_stamp_ms_first = None
        self.time_stamp_ms_last = None
        self.motion_states = dict()

    def __str__(self):
        string = "Track: track_id=" + str(self.track_id) + ", agent_type=" + str(self.agent_type) + \
               ", length=" + str(self.length) + ", width=" + str(self.width) + \
               ", time_stamp_ms_first=" + str(self.time_stamp_ms_first) + \
               ", time_stamp_ms_last=" + str(self.time_stamp_ms_last) + \
               "\n motion_states:"
        for key, value in sorted(self.motion_states.items()):
            string += "\n    " + str(key) + ": " + str(value)
        return string
