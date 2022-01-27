import numpy as np
import os
from posture_monitor.src.util import get_time
from posture_monitor.src.PostureAlertRule import *
from posture_monitor.src.PostureMetricTs import *

from posture_monitor.src.util import calculate_landmark_line, get_time
from posture_monitor.src.PostureSession import PostureSession

PACKAGE_VERSION = (0, 0, 1)
CONFIG_VERSION = (0, 0, 1)

# to-do, make all those configurations templates, or initialize those somewhere else
headdown_metric = PostureMetricTs("headdown", metric_func=lambda landmarks: min(landmarks[6].y, landmarks[3].y))
left_shoulder_y_metric = PostureMetricTs("left_shoulder_y", metric_func=lambda landmarks: landmarks[11].y)
right_shoulder_y_metric =  PostureMetricTs("right_shoulder_y", metric_func=lambda landmarks: landmarks[12].y)
left_shoulder_x_metric = PostureMetricTs("left_shoulder_x", metric_func=lambda landmarks: landmarks[11].x)
right_shoulder_x_metric =  PostureMetricTs("right_shoulder_x", metric_func=lambda landmarks: landmarks[12].x)



headdown_10_seconds_metric_funct = if_metric_fail_avg_and_last_second("headdown", 
lambda head_level: head_level > 0.31, seconds=4, percent=1.00)
headdown_10_seconds = PostureSubMetricTs("headdown_seconds", headdown_10_seconds_metric_funct)
headdown_alert = PostureKDeltaAlert('headdown', headdown_10_seconds, 1)

rightshoulder_down_metric_funct = if_metric_fail_avg_and_last_second("right_shoulder_y", 
lambda right_shoulder_y: right_shoulder_y >= 0.57, seconds=2, percent=1.00)
rightshoulder_down_10_seconds = PostureSubMetricTs("rightshoulder_down_seconds", rightshoulder_down_metric_funct)    
rightshoulder_down_alert = PostureKDeltaAlert('rightshoulder_10s_down', rightshoulder_down_10_seconds, 1)

left_right_shoulder_y_diff = PostureMetricTs("left_right_shoulder_y_diff", metric_func=lambda landmarks: np.abs(landmarks[11].y - landmarks[12].y))
shoulder_tilt_metric_funct = if_metric_fail_avg_and_last_second("left_right_shoulder_y_diff", lambda level_diff: level_diff > 0.02, seconds=4, percent=1.00)
shoulder_tilt_seconds = PostureSubMetricTs("shoulder_tilt_seconds", shoulder_tilt_metric_funct)
shoulder_tilt_alert = PostureKDeltaAlert("shoulder_tilts", shoulder_tilt_seconds, 1)

metricTsDict = {    
    headdown_metric.name: headdown_metric,
    headdown_10_seconds.name: headdown_10_seconds,
    left_shoulder_y_metric.name: left_shoulder_y_metric,
    right_shoulder_y_metric.name: right_shoulder_y_metric,
    left_shoulder_x_metric.name: left_shoulder_x_metric,
    right_shoulder_x_metric.name: right_shoulder_x_metric,
    rightshoulder_down_10_seconds.name: rightshoulder_down_10_seconds,
    left_right_shoulder_y_diff.name: left_right_shoulder_y_diff,
    shoulder_tilt_seconds.name: shoulder_tilt_seconds
}

from pathlib import Path
config_file_path = os.path.abspath(__file__)

posture_session_args = {
    'metricTsDict': metricTsDict,
    'alertRules': [headdown_alert, shoulder_tilt_alert],
    'config_datapath':config_file_path
}