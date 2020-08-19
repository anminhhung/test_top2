from collections import deque
import numpy as np

class Track(object):

    def __init__(self, position, score, track_id, mm_steps):
        self.position = position
        self.init_position = position
        self.score = score
        self.id = track_id
        self.ims = deque()
        self.last_position = deque([], mm_steps+1)
        self.last_v = np.empty(0) # distance between two frame
        self.number_pred_det = {0: 0, 1: 0, 2:0, 3:0} 
        self.number_pred_track = {0: 0, 1: 0, 2:0, 3:0} 
        self.classify = False # If true => turn on classify module
        self.point_in = None
        self.point_out = None
        self.best_bbox = None
        self.class_id = None

    def has_positive_area(self):
        check = self.position[2] > self.position[0] and self.position[3] > self.position[1]
        return check

def get_center(position):
    x1 = position[0]
    y1 = position[1]
    x2 = position[2]
    y2 = position[3]

    return np.array([(x2 + x1) / 2, (y2 + y1) / 2], dtype=int)

def get_width(position):
    return position[2] - position[0]

def get_height(position):
    return position[3] - position[1]

def make_pos(cx, cy, width, height):
    return np.array([cx - width / 2,
                    cy - height / 2,
                    cx + width / 2,
                    cy + height / 2], dtype=int)