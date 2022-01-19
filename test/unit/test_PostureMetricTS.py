from collections import defaultdict
import time
import os
from turtle import update
import pytest
from unittest.mock import patch
import json
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
from posture_monitor.src.PostureMetricTs import PostureMetricTs, PostureSubMetricTs
from posture_monitor.src.PostureMetricTs import if_metric_fail_avg_and_last_second
from posture_monitor.src.util import get_time


TEMP_DIR = 'test/temp_test_dir'
def create_temp_dir():
    if not os.path.exists(TEMP_DIR):
        # Create a new directory because it does not exist 
        print("creating temp dir")
        os.makedirs(TEMP_DIR)
    else:
        print('temp dir exists')

def cleanup_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


@pytest.fixture()
def test_metricTs():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])
    return test_metricTs

def test_update():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])

    test_landmarks_1 = [1]
    test_landmarks_2 = [4]
    test_landmarks_3 = [6]
    # t1, update 1
    t1 = get_time()
    test_metricTs.update(test_landmarks_1)
    #time.sleep(1)
    # t2, update 4, 6
    t2 = t1 + 1
    test_metricTs.update(test_landmarks_2, timestamp=t2)
    test_metricTs.update(test_landmarks_3, timestamp=t2)
    # check second to frame works
    expected_second_to_frame_scores = [[1], [4, 6]]
    expected_second_to_avg_frame_scores = [1, 5]
    assert(list(test_metricTs.second_to_frame_scores.values()) == \
            expected_second_to_frame_scores)
    # check second to avg frame score works
    assert(list(test_metricTs.second_to_avg_frame_scores.values()) == \
            expected_second_to_avg_frame_scores)


def test_get_past_data_full_data():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])

    
    now = get_time()
    #print(f"test case now: {now}")
    test_data ={
        now -1: 1,
        now -2: 2,
        now -3: 3
    }
    test_metricTs.second_to_avg_frame_scores.update(test_data)
    #print(test_metricTs.second_to_avg_frame_scores )
    past_data = test_metricTs.get_past_data(start_time=now, seconds=4)
    assert(past_data == [PostureMetricTs.DEFAULT_NULL_FILL, 
        3,2,1])



def test_get_past_data_skip_data():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])

    
    now = get_time()
    #print(f"test case now: {now}")
    test_data ={
        now -1: 1,
        now -3: 2,
        now -6: 3
    }
    test_metricTs.second_to_avg_frame_scores.update(test_data)
    #print(test_metricTs.second_to_avg_frame_scores )
    past_data = test_metricTs.get_past_data(start_time=now, seconds=6)
    assert(past_data == [3, 
        PostureMetricTs.DEFAULT_NULL_FILL,
        PostureMetricTs.DEFAULT_NULL_FILL,2,
        PostureMetricTs.DEFAULT_NULL_FILL,1])


def test_posture_checking_function():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])  
    test_metricTs_dict = {'test_metric': test_metricTs}
    sub_metric_func = if_metric_fail_avg_and_last_second('test_metric',
        threshold_rule = lambda metric: int(metric > 0.5),
        seconds = 2,
        percent = 0.5
    )
    # test_SubMetricTs = PostureSubMetricTs(
    #     "test_submetric",
    #     sub_metric_func)
    now = get_time()
    test_metricTs.second_to_avg_frame_scores.update({
        now-1: 1,
        now-2: 1,
        now-3: 0.1,
        now-4: 0.1
    })
    result = sub_metric_func(test_metricTs_dict)
    assert result == 1

def test_posture_submetrics_update():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])  
    test_metricTs_dict = {'test_metric': test_metricTs}
    sub_metric_func = if_metric_fail_avg_and_last_second('test_metric',
        threshold_rule = lambda metric: int(metric > 0.5),
        seconds = 2,
        percent = 0.5
    )
    test_SubMetricTs = PostureSubMetricTs(
        "test_submetric",
        sub_metric_func)
    now = get_time()
    # update original metric
    test_metricTs.second_to_avg_frame_scores.update({
        now-1: 0.1,
        now-2: 1,
        now-3: 1,
        now-4: 0.1
    })
    test_metricTs.update([0.1])
    # update submetric
    test_SubMetricTs.update(test_metricTs_dict)
    expected_result = [0]
    result = list(test_SubMetricTs.second_to_avg_frame_scores.values())
    assert result== expected_result

def test_posture_submetrics_get_past_data():
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])  
    test_metricTs_dict = {'test_metric': test_metricTs}
    sub_metric_func = if_metric_fail_avg_and_last_second('test_metric',
        threshold_rule = lambda metric: int(metric > 0.5),
        seconds = 2,
        percent = 0.5
    )
    test_SubMetricTs = PostureSubMetricTs(
        "test_submetric",
        sub_metric_func)
    now = get_time()
    test_SubMetricTs.second_to_avg_frame_scores.update({
        #now:0.1,
        now-1: 1,
        now-2: 1,
        now-3: 0.1,
        now-4: 0.1
    })
    #test_SubMetricTs.update(test_metricTs_dict)
    result = test_SubMetricTs.get_past_data(seconds=4)
    assert result==[0.1, 0.1, 1, 1]
    

def test_export_data():
    """
    create fake json, dump it as historical data
    create new data
    call expoert method
    
    load updated file
    
    assert if data is updated
    assert if data is reset

    clean up temp_dir
    """
    create_temp_dir()
    test_metricTs = \
            PostureMetricTs("test_metric", 
            metric_func=lambda landmarks: landmarks[0])

    historical_data = {1:1, 2:2}
    data_file = os.path.join(TEMP_DIR, test_metricTs.name+".json")

    with open(data_file, 'w') as outfile:
        json.dump(historical_data, outfile, indent=4)

    new_data = {3:3, 4:4}
    test_metricTs.second_to_avg_frame_scores.update(new_data)
    test_metricTs.export_data(TEMP_DIR)

    with open(data_file, 'r') as infile:
        update_data = json.load(infile)

    assert update_data == {
        str(i):i for i in range(1, 5)
    }
    assert len(test_metricTs.second_to_avg_frame_scores) == 0 
    cleanup_temp_dir()