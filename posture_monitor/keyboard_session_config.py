import numpy as np
import os
from posture_monitor.src.util import get_time
from posture_monitor.src.PostureAlertRule import *
from posture_monitor.src.PostureMetricTs import *

from posture_monitor.src.util import calculate_landmark_line, get_time
from posture_monitor.src.PostureSession import PostureSession

PACKAGE_VERSION = (0, 0, 1)
CONFIG_VERSION = (0, 0, 1)

WRIST_DIFF_THRESHOLD = 0.06

# to-do, make all those configurations templates, or initialize those somewhere else
def left_right_wrist_not_symmetric(landmark_lst: list) -> float:
    # find midpoint btw two shoulders and reference, assuming shoulders even
    shoulder_mid_x =  landmark_lst[12].x + (landmark_lst[11].x - landmark_lst[12].x)/2.0
    # find the left-corresponding wrist points, 16 and 19
    right_wrist_reflection_x = shoulder_mid_x*2 - landmark_lst[16].x
    x_diff = np.round(np.abs(right_wrist_reflection_x-landmark_lst[15].x), 2)
    y_diff = np.round(np.abs(landmark_lst[16].y-landmark_lst[15].y), 2)
    return max([x_diff, y_diff]) > WRIST_DIFF_THRESHOLD

left_right_wrist_not_symmetric_metric = PostureMetricTs("left_right_wrist_not_symmetric", metric_func=left_right_wrist_not_symmetric)
wrist_not_symmetric_alertrule = PostureKDeltaAlert('wrist_not_symmetrict', left_right_wrist_not_symmetric_metric, 1)

metricTsDict = {    
    left_right_wrist_not_symmetric_metric.name: left_right_wrist_not_symmetric_metric
}

from pathlib import Path
config_file_path = os.path.abspath(__file__)

posture_session_args = {
    'metricTsDict': metricTsDict,
    'alertRules': [wrist_not_symmetric_alertrule],
    'config_datapath':config_file_path
}