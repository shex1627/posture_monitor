import numpy as np
import os
from posture_monitor.src.util import get_time
from posture_monitor.src.PostureAlertRule import *
from posture_monitor.src.PostureMetricTs import *

from posture_monitor.src.util import calculate_landmark_line, get_time, landmark_3d_dist
from posture_monitor.src.PostureSession import PostureSession

PACKAGE_VERSION = (0, 0, 1)
CONFIG_VERSION = (0, 0, 1)

HEAD_LEVEL_THRESHOLD = 0.30#0.25
SHOULDER_TILT_THRESHOLD = 0.02
HEAD_SHIFT_THRESHOLD = 0.
SHOULDER_TILT_THRESHOLD
INFRONT_COMPUTER_MIN = 20

# to-do, make all those configurations templates, or initialize those somewhere else
headdown_metric = PostureMetricTs("headdown", metric_func=lambda landmarks: min(landmarks[6].y, landmarks[3].y))
left_shoulder_y_metric = PostureMetricTs("left_shoulder_y", metric_func=lambda landmarks: landmarks[11].y)
right_shoulder_y_metric =  PostureMetricTs("right_shoulder_y", metric_func=lambda landmarks: landmarks[12].y)
left_shoulder_x_metric = PostureMetricTs("left_shoulder_x", metric_func=lambda landmarks: landmarks[11].x)
right_shoulder_x_metric =  PostureMetricTs("right_shoulder_x", metric_func=lambda landmarks: landmarks[12].x)



headdown_10_seconds_metric_funct = if_metric_fail_avg_and_last_second("headdown", 
lambda head_level: head_level > HEAD_LEVEL_THRESHOLD, seconds=4, percent=1.00)
headdown_10_seconds = PostureSubMetricTs("headdown_seconds", headdown_10_seconds_metric_funct)
headdown_alert = PostureKDeltaAlert('headdown', headdown_10_seconds, 1)

rightshoulder_down_metric_funct = if_metric_fail_avg_and_last_second("right_shoulder_y", 
lambda right_shoulder_y: right_shoulder_y >= 0.57, seconds=2, percent=1.00)
rightshoulder_down_10_seconds = PostureSubMetricTs("rightshoulder_down_seconds", rightshoulder_down_metric_funct)    
rightshoulder_down_alert = PostureKDeltaAlert('rightshoulder_10s_down', rightshoulder_down_10_seconds, 1)

left_right_shoulder_y_diff = PostureMetricTs("left_right_shoulder_y_diff", metric_func=lambda landmarks: np.abs(landmarks[11].y - landmarks[12].y))
shoulder_tilt_metric_funct = if_metric_fail_avg_and_last_second("left_right_shoulder_y_diff", lambda level_diff: level_diff > SHOULDER_TILT_THRESHOLD, seconds=4, percent=1.00)
shoulder_tilt_seconds = PostureSubMetricTs("shoulder_tilt_seconds", shoulder_tilt_metric_funct)
shoulder_tilt_alert = PostureKDeltaAlert("shoulder_tilts", shoulder_tilt_seconds, 1)

good_posture = PostureSubMetricTs("good_posture", metric_func=lambda metricTsDict: int(not any([
    metricTsDict['left_right_shoulder_y_diff'].get_past_data(seconds=1)[0] > SHOULDER_TILT_THRESHOLD,
    metricTsDict['headdown'].get_past_data(seconds=1)[0] > HEAD_LEVEL_THRESHOLD,
])))


right_eye_shoulder_x_diff = PostureMetricTs("right_eye_shoulder_x_diff", metric_func=lambda landmarks: np.abs(landmarks[6].x - landmarks[12].x))#np.abs(landmarks[6].z - landmarks[12].z)landmark_3d_dist(landmarks[6], landmarks[12])
left_eye_shoulder_x_diff =  PostureMetricTs("left_eye_shoulder_x_diff", metric_func=lambda landmarks: np.abs(landmarks[3].x - landmarks[11].x)) #landmarks[3].x - landmarks[11].x)landmark_3d_dist(landmarks[3], landmarks[11])
right_shoulder_elbow_x_diff = PostureMetricTs("right_shoulder_elbow_x_diff", metric_func=lambda landmarks: np.abs(landmarks[12].x - landmarks[14].x))
left_shoulder_elbow_x_diff = PostureMetricTs("left_shoulder_elbow_x_diff", metric_func=lambda landmarks: np.abs(landmarks[11].x - landmarks[13].x))

head_shift_alert = if_metric_fail_avg_and_last_second("left_eye_shoulder_x_diff", lambda level_diff: level_diff > SHOULDER_TILT_THRESHOLD, seconds=3, percent=1.00)
body_shift_alert = if_metric_fail_avg_and_last_second("left_shoulder_elbow_x_diff", lambda level_diff: level_diff > SHOULDER_TILT_THRESHOLD, seconds=3, percent=1.00)

shift_metric_lst = [left_eye_shoulder_x_diff, right_eye_shoulder_x_diff, left_shoulder_elbow_x_diff, right_shoulder_elbow_x_diff]
shift_metric_dict = {metric.name: metric for metric in shift_metric_lst}

infront_computer = PostureMetricTs("infront_computer", metric_func=lambda landmarks: 
    (landmarks[1].y <= 2 and landmarks[1].y >=0.2) and 
    (landmarks[4].y <= 2 and landmarks[4].y >=0.2) and
    (landmarks[1].z < -0.5 and landmarks[4].z <-0.5),
    fillna=0
    )
infront_computer_for_30_func = if_metric_fail_avg("infront_computer", 
    lambda infront_computer: infront_computer > 0, 
    seconds=60*INFRONT_COMPUTER_MIN, percent=0.80, last_second=False)
infront_computer_for_30 = PostureSubMetricTs("infront_computer_30min", infront_computer_for_30_func)
infront_computer_for_30_alert = PostureKDeltaAlert("infront_computer_30min", infront_computer_for_30, 1)

metricTsDict = {    
    headdown_metric.name: headdown_metric,
    headdown_10_seconds.name: headdown_10_seconds,
    left_shoulder_y_metric.name: left_shoulder_y_metric,
    right_shoulder_y_metric.name: right_shoulder_y_metric,
    left_shoulder_x_metric.name: left_shoulder_x_metric,
    right_shoulder_x_metric.name: right_shoulder_x_metric,
    rightshoulder_down_10_seconds.name: rightshoulder_down_10_seconds,
    left_right_shoulder_y_diff.name: left_right_shoulder_y_diff,
    shoulder_tilt_seconds.name: shoulder_tilt_seconds,
    good_posture.name: good_posture,
    infront_computer.name: infront_computer,
    infront_computer_for_30.name: infront_computer_for_30
}

metricTsDict.update(shift_metric_dict)

from pathlib import Path
config_file_path = os.path.abspath(__file__)

posture_session_args = {
    'metricTsDict': metricTsDict,
    'alertRules': [headdown_alert, shoulder_tilt_alert, 
    infront_computer_for_30_alert
    ],
    'config_datapath':config_file_path
}