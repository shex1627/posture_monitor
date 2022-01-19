from collections import defaultdict
import time
import os
import pytest

from posture_monitor.src.PostureMetricTs import PostureMetricTs, PostureSubMetricTs
from posture_monitor.src.PostureAlertRule import PostureKDeltaAlert, PostureAlertRule
from posture_monitor.src.util import get_time
from unittest.mock import patch


def test_alert_trigger():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])
    now = get_time()
    test_metricTs.second_to_avg_frame_scores.update({
        now -1: 1
    })

    test_alert = PostureKDeltaAlert("test_alert", test_metricTs, 1)
    assert test_alert.alert_trigger() == 1


def test_alert_not_trigger():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])
    now = get_time()
    test_metricTs.second_to_avg_frame_scores.update({
        now -1: 0.1
    })
    test_alert = PostureKDeltaAlert("test_alert", test_metricTs, 1)
    assert (test_alert.alert_trigger() == 0)