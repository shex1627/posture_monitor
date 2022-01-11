import cv2
import mediapipe as mp
import logging
import sys
import time
from collections import OrderedDict

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