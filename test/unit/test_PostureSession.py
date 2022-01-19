from collections import defaultdict
import time
import os
import json
import pytest
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
from posture_monitor.src.PostureMetricTs import PostureMetricTs, PostureSubMetricTs
from posture_monitor.src.PostureSession import PostureSession
from posture_monitor.src.PostureAlertRule import PostureKDeltaAlert
from posture_monitor.src.util import get_time
from unittest.mock import patch

TEMP = "test/posture_session_dir"

def clean_up_temp():
    try:
        shutil.rmtree(TEMP)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

@pytest.fixture
def metrics_and_alerts():
    test_metricTs1 = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])

    test_metricTs2 = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[1])

    test_alert1 = PostureKDeltaAlert("alert1", test_metricTs1, delta=1)
    test_alert2 = PostureKDeltaAlert("alert2", test_metricTs2, delta=2)
    return [test_metricTs1, test_metricTs2, test_alert1, test_alert2]

def test_update_metrics(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2],
            data_dir=TEMP 
            )

    test_landmarks = [1, 2]
    now = get_time()
    test_PostureSession.update_metrics(test_landmarks)
    assert test_metricTs1.second_to_frame_scores[now] == [1]
    assert test_metricTs2.second_to_frame_scores[now] == [2]
    assert test_metricTs1.second_to_avg_frame_scores[now] == 1
    assert test_metricTs2.second_to_avg_frame_scores[now] == 2
    clean_up_temp()


def test_check_posture_alert_no_alert(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2],
            data_dir=TEMP 
            )

    test_landmarks = [2, 2]
    now = get_time()
    test_PostureSession.update_metrics(test_landmarks)
    alert_trigger = test_PostureSession.check_posture_alert()
    assert alert_trigger == []
    clean_up_temp()

def test_check_posture_alert_one_alert(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2],
            data_dir=TEMP 
            )

    now = get_time()
    test_metricTs1.second_to_avg_frame_scores.update(
        {now-1:1}
    )
    alert_trigger = test_PostureSession.check_posture_alert()
    assert alert_trigger == ['alert1']
    clean_up_temp()


def test_check_posture_alert_two_alert(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts

    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2],
            data_dir=TEMP 
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
    clean_up_temp()
    

def test_data_export(metrics_and_alerts):
    test_metricTs1, test_metricTs2, test_alert1, test_alert2 = metrics_and_alerts
    test_PostureSession = \
            PostureSession(dict(metric1=test_metricTs1, metric2=test_metricTs2),
            alertRules=[test_alert1, test_alert2],
            data_dir=TEMP 
            )

    historical_data = {1:1, 2:2}
    data_file = os.path.join(test_PostureSession.session_data_dir, test_metricTs1.name+".json")

    with open(data_file, 'w') as outfile:
        json.dump(historical_data, outfile, indent=4)

    new_data = {3:3, 4:4}
    test_metricTs1.second_to_avg_frame_scores.update(new_data)
    
    test_PostureSession.export_data()

    with open(data_file, 'r') as infile:
        update_data = json.load(infile)

    assert update_data == {
        str(i):i for i in range(1, 5)
    }
    assert len(test_metricTs1.second_to_avg_frame_scores) == 0 
    clean_up_temp()