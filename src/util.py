import cv2
import mediapipe as mp
import logging
import sys

def calculate_landmark_line(landmark1:mp.framework.formats.landmark_pb2.NormalizedLandmark, 
                  landmark2:mp.framework.formats.landmark_pb2.NormalizedLandmark) -> tuple:
    slope = (landmark2.y - landmark1.y)/(landmark2.x - landmark1.x)
    y_intercept = landmark1.y - slope * landmark1.x
    return (slope, y_intercept)