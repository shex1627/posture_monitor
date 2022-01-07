#import cv2
#import mediapipe as mp
import logging
import sys
from collections import OrderedDict
import time 
import os
from typing import Callable
import pickle
import json
import numpy as np
import threading
import winsound
import logging


logger = logging.getLogger("PostureState")

class OrderedDefaultDict(OrderedDict):
    factory = list

    def __missing__(self, key):
        self[key] = value = self.factory()
        return value


class PostureState:
    fps = 25
    EXPORT_MIN = 15
    SECOND_PER_MIN = 60
    FRAME_THRESHOLD = fps * SECOND_PER_MIN * EXPORT_MIN

    def __init__(self, name: str, score_func: Callable, if_bad_posture: Callable, dir=None):
        self.score_by_frame = OrderedDefaultDict()
        self.score_by_second = OrderedDefaultDict()
        self.score_lst = []
        self.score_func = score_func
        self.if_bad_posture = if_bad_posture
        self.dir = dir
        self.last_alert_time = 0
        self.name = name

    def update(self, landmarks: list):
        """ add postuer data per frame. """
        time_hash = self.get_time()
        score = self.score_func(landmarks)
        self.score_lst.append(score)
        self.score_by_frame[time_hash].append(score)
        if self.dir and len(self.score_lst) > PostureState.FRAME_THRESHOLD:
            # outfile_name = os.path.join(self.dir, f"data_{self.get_time()}.pkl")
            # logger.info(f"saving data to {outfile_name}")
            # with open(outfile_name, 'wb') as outfile:
            #     pickle.dump(self.score_lst, outfile)
            # exporting frame data
            outfile_name = os.path.join(self.dir, f"data_{self.name}_{self.get_time()}.json")
            logger.info(f"saving data to {outfile_name}")
            with open(outfile_name, 'w') as outfile:
                json.dump(self.score_by_frame, outfile, indent=4)

            self.score_lst = []
            self.score_by_frame = OrderedDefaultDict()
        if len(self.score_lst) > PostureState.fps:
            data_window = self.score_lst[-PostureState.fps::]
            data_window = list(map(lambda score: np.round(score, 2), data_window))
            mean_score = np.mean(list(map(self.if_bad_posture, data_window)))

    def get_time(self):
        return int(time.time())

    def bad_posture_frame(self):
        "return if last second has bad posture"
        if self.score_lst:
            return self.if_bad_posture(self.score_lst[-1])

    def bad_posture_second(self):
        "return if last second has bad posture"
        if len(self.score_lst) > PostureState.fps:
            data_window = self.score_lst[-PostureState.fps::]
            data_window = list(map(lambda score: np.round(score, 2), data_window))
            mean_score = np.mean(list(map(self.if_bad_posture, data_window)))
            return mean_score > 0.9
        return False

    def bad_posture(self):
        """ check if posture is bad in last X seconds"""
        if_in_bad_posture = False
        #logging.info(len(self.score_lst))
        seconds_threshold = 15
        frame_offset = seconds_threshold * PostureState.fps
        percent_threshold = 0.9
        if len(self.score_lst) >= frame_offset:
            #logging.info("check if posture is bad")
            data_window = self.score_lst[-frame_offset::]
            mean_score = np.mean(list(map(self.if_bad_posture, data_window)))
            logging.info(f"mean score {mean_score} in {seconds_threshold} seconds")
            if_in_bad_posture = mean_score > percent_threshold
            #logging.info(f"bad posture: {if_in_bad_posture}")
        return if_in_bad_posture
            
    
    def check_posture(self, alert_sound_file="./resources/alert.wav"):
        # relative import is bad
        # make sure alert don't spam
        in_bad_posture = self.bad_posture()
        logging.info(f"in bad posture: {in_bad_posture}")
        if in_bad_posture and self.bad_posture_second():
            logging.info("in bad posture, triggering alert")
            now = self.get_time()
            if now - self.last_alert_time > 0.5:
                self.last_alert_time = now 
                winsound.PlaySound(alert_sound_file, winsound.SND_ASYNC |  winsound.SND_FILENAME) #
            