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

from posture_monitor.src.PostureMetricTs import PostureMetricTs, PostureSubMetricTs
from posture_monitor.src.PostureAlertRule import PostureKDeltaAlert, PostureAlertRule
from posture_monitor.src.PostureState import OrderedDefaultDict
from posture_monitor.src.util import get_time

logger = logging.getLogger("PostureSession")

class PostureSession:
        def __init__(self, metricTsDict: Dict[str, PostureMetricTs], alertRules: List[PostureAlertRule]) -> None:
            self.metrics = metricTsDict
            self.alertRules = alertRules
            self._last_alert_time = 0

        def update_metrics(self, landmarks) -> None:
             for metric in self.metrics.values():
                logger.info(f"updateing {metric.name}")
                if isinstance(metric, PostureSubMetricTs):
                    metric.update(self.metrics)
                else:
                    metric.update(landmarks)

        def trigger_alert_sound(self, alert_sound_file="./resources/alert.wav") -> None:
            """ Play the alert sound."""
            logging.info("in bad posture, triggering alert")
            now = get_time()
            if now - self._last_alert_time > 0.5:
                self._last_alert_time = now 
                winsound.PlaySound(alert_sound_file, winsound.SND_ASYNC |  winsound.SND_FILENAME) #

        def check_posture_alert(self, trigger_sound=True) -> List[str]:
            alerts_triggered = []
            for alert in self.alertRules:
                if alert.alert_trigger(self.metrics):
                    alerts_triggered.append(alert.name)
            if trigger_sound and len(alerts_triggered):
                self.trigger_alert_sound()
            return alerts_triggered    
