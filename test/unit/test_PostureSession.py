from collections import defaultdict
import time
import os
from posture_monitor.src.PostureAlertRule import PostureKDeltaAlert
import pytest
dir_path = os.path.dirname(os.path.realpath(__file__))
from posture_monitor.src.PostureMetricTs import PostureMetricTs, PostureSubMetricTs
from posture_monitor.src.PostureSession import PostureSession
from posture_monitor.src.util import get_time
from unittest.mock import patch

@pytest.fixture
def metrics_and_alerts():
    test_metricTs1 = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0], 
            data_dir="test_data")

    test_metricTs2 = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[1], 
            data_dir="test_data")

    test_alert1 = PostureKDeltaAlert("alert1", test_metricTs1, delta=1)
    test_alert2 = PostureKDeltaAlert("alert2", test_metricTs2, delta=2)
    return [test_metricTs1, test_metricTs2, test_alert1, test_alert2]

def test_update_metrics(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2]
            )

    test_landmarks = [1, 2]
    now = get_time()
    test_PostureSession.update_metrics(test_landmarks)
    assert test_metricTs1.second_to_frame_scores[now] == [1]
    assert test_metricTs2.second_to_frame_scores[now] == [2]
    assert test_metricTs1.second_to_avg_frame_scores[now] == 1
    assert test_metricTs2.second_to_avg_frame_scores[now] == 2


def test_check_posture_alert_no_alert(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2]
            )

    test_landmarks = [2, 2]
    now = get_time()
    test_PostureSession.update_metrics(test_landmarks)
    alert_trigger = test_PostureSession.check_posture_alert()
    assert alert_trigger == []

def test_check_posture_alert_one_alert(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2]
            )

    now = get_time()
    test_metricTs1.second_to_avg_frame_scores.update(
        {now-1:1}
    )
    alert_trigger = test_PostureSession.check_posture_alert()
    assert alert_trigger == ['alert1']


def test_check_posture_alert_two_alert(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2]
            )

    test_landmarks = [1, 2]
    now = get_time()
    test_metricTs1.second_to_avg_frame_scores.update(
        {now-1:1}
    )
    test_metricTs2.second_to_avg_frame_scores.update(
        {now-1:2}
    )
    alert_trigger = test_PostureSession.check_posture_alert()
    assert alert_trigger == ['alert1', 'alert2']
    