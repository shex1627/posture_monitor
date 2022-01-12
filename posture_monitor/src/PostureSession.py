#import cv2
#import mediapipe as mp
import logging
import sys
from collections import OrderedDict
import time 
import os
from typing import Callable, List, Dict
import pickle
import json
import numpy as np
import threading
import winsound
import logging
import json
import threading

from posture_monitor.src.PostureMetricTs import PostureMetricTs, PostureSubMetricTs
from posture_monitor.src.PostureAlertRule import PostureKDeltaAlert, PostureAlertRule
from posture_monitor.src.PostureState import OrderedDefaultDict
from posture_monitor.src.util import get_time

logger = logging.getLogger("PostureSession")

class PostureSession:

        def __init__(self, metricTsDict: Dict[str, PostureMetricTs], alertRules: List[PostureAlertRule], 
        data_dir=None, data_export_min=15) -> None:
            self.metrics = metricTsDict
            self.alertRules = alertRules
            self._last_alert_time = 0
            self.start_time = get_time()
            self.last_data_export_time = self.start_time
            self.data_export_min = data_export_min
            # create session file dir if not exists
            self.data_dir = data_dir
            self.session_data_dir = os.path.join(data_dir, f"session_{self.start_time}")
            if data_dir:
                self.init_data_dir()

        def update_metrics(self, landmarks, export_data=True) -> None:
            """Update all the metrics. If a metric is a submetric, then feed in the metricDict"""
            for metric in self.metrics.values():
                logger.info(f"updateing {metric.name}")
                if isinstance(metric, PostureSubMetricTs):
                    metric.update(self.metrics)
                else:
                    metric.update(landmarks)
            now = get_time()
            if export_data and self.data_dir and (now - self.last_data_export_time) > 5:
                th = threading.Thread(target=self.export_data)
                th.start()
                self.last_data_export_time = now

        def trigger_alert_sound(self, alert_sound_file="./resources/alert.wav") -> None:
            """ Play the alert sound."""
            logging.info("in bad posture, triggering alert")
            now = get_time()
            if now - self._last_alert_time > 0.5:
                self._last_alert_time = now 
                winsound.PlaySound(alert_sound_file, winsound.SND_ASYNC |  winsound.SND_FILENAME) #

        def check_posture_alert(self, trigger_sound=True) -> List[str]:
            """ return any alerts that is triggered."""
            alerts_triggered = []
            for alert in self.alertRules:
                if alert.alert_trigger(self.metrics):
                    alerts_triggered.append(alert.name)
            if trigger_sound and len(alerts_triggered):
                self.trigger_alert_sound()
            return alerts_triggered    

        def init_data_dir(self) -> None:
            if not os.path.exists(self.data_dir):
            # Create a new directory because it does not exist 
                os.makedirs(self.data_dir)
            # create session root data folder
            session_data_dir = os.path.join(self.data_dir, f"session_{self.start_time}")
            os.makedirs(session_data_dir)
            #for metric_name in self.metrics:
            #    os.makedirs(os.path.join(session_data_dir, metric_name+".log"))

        
        def export_data(self):
            """export all metrics data if last export time is old enough."""
            logging.debug("writing to file")
            for metric_name in self.metrics:
                metric_filepath = os.path.join(self.session_data_dir, metric_name+".json")
                with open(metric_filepath, 'w') as outfile:
                    data = self.metrics[metric_name].second_to_avg_frame_scores
                    json.dump(data, outfile, indent=4)
                    

