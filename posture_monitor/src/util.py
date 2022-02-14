from typing import Dict
import cv2
import mediapipe as mp
import logging
import sys
import time
from collections import OrderedDict
import numpy as np

class OrderedDefaultDict(OrderedDict):
    factory = list

    def __missing__(self, key):
        self[key] = value = self.factory()
        return value

def get_time():
        return int(time.time())

def calculate_landmark_line(landmark1:mp.framework.formats.landmark_pb2.NormalizedLandmark, 
                  landmark2:mp.framework.formats.landmark_pb2.NormalizedLandmark) -> tuple:
    """ Given two landmark points, calculate the slope of those two points. """
    slope = (landmark2.y - landmark1.y)/(landmark2.x - landmark1.x)
    y_intercept = landmark1.y - slope * landmark1.x
    return (slope, y_intercept)

def landmark_3d_dist(landmark1:mp.framework.formats.landmark_pb2.NormalizedLandmark, 
                  landmark2:mp.framework.formats.landmark_pb2.NormalizedLandmark) -> float:
    return np.sqrt((landmark1.y - landmark2.y)**2 + (landmark1.x - landmark2.x)**2 +(landmark1.z - landmark2.z)**2)

def make_raw_landmark_metric_func(landmark_index: int, axis: str):
    """
    higher order function to create raw landmark metric functions based on index
    """
    def landmark_metric_func(landmarks: list):
        if axis == 'x':
            return landmarks[landmark_index].x
        elif axis == 'y':
            return landmarks[landmark_index].y
        else:
            raise ValueError(f"axis can only be 'x' or 'y'")
    return landmark_metric_func

def create_basic_landmarks():# -> Dict[str, PostureMetricTs]:
    """ create metricTS for all metrics in the head and shoulder."""
    from posture_monitor.src.PostureMetricTs import PostureMetricTs
    landmark_indices = list(range(14 + 1))
    metricTs_dict = dict()
    for landmark_index in landmark_indices:
        landmark_x_metric = PostureMetricTs(f"landmark_{landmark_index}_x", metric_func=make_raw_landmark_metric_func(landmark_index, 'x'))
        landmark_y_metric = PostureMetricTs(f"landmark_{landmark_index}_y", metric_func=make_raw_landmark_metric_func(landmark_index, 'y'))
        metricTs_dict[landmark_x_metric.name] = landmark_x_metric
        metricTs_dict[landmark_y_metric.name] = landmark_y_metric
    return metricTs_dict
                