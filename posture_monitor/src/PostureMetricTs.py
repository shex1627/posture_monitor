import time 
import os
from typing import Callable, OrderedDict, List, Dict
from posture_monitor.src.util import OrderedDefaultDict
from collections import defaultdict
import logging
import numpy as np
from posture_monitor.src.util import get_time

logger = logging.getLogger("PostureMetricTs")

class PostureMetricTs:
    DEFAULT_NULL_FILL = 0

    def __init__(self, name: str, metric_func: Callable, data_dir=None,
    fillna=DEFAULT_NULL_FILL):
        self.name = name
        self.get_metric = metric_func   
        self.second_to_frame_scores = OrderedDefaultDict()
        self.second_to_avg_frame_scores = defaultdict(lambda : fillna) # default to negative 
        self.data_dir = data_dir
        self.fillna = fillna
        
    
    def update(self, landmarks: list, timestamp: int=None):
        """ add postuer data per frame. """
        if not timestamp:
            timestamp = get_time()
        score = self.get_metric(landmarks)
        logger.debug(f"updating {self.name} with {score} at {timestamp}")
        self.second_to_frame_scores[timestamp].append(score) #data is partitioned by integer time
        self.second_to_avg_frame_scores[timestamp] = np.mean(self.second_to_frame_scores[timestamp])
        #### if there is data dir, and dataover  append stuff to data dir
    
    # to-do, add parameter to change the orders
    def get_past_data(self, start_time: int=None, seconds: int=1,
    transform: Callable[[float], int]=None) -> List[float]:
        """Get the last X seconds of data, not inclusive.
        The oldest data first
        example
        get_past_data(seconds=3, start_time=10) ->
            t9, t8, t7
        """
        #print(f"now {get_time()}")
        #print(f"past data: {self.second_to_avg_frame_scores}")
        if not start_time:
            start_time = get_time()
        past_data = []
        if seconds > 0:
            #print(f"seconds {seconds}")
            for i in reversed(range(seconds)):
                #print(f"index: {i}")
                past_time = start_time -1 - i
                data = self.second_to_avg_frame_scores[past_time]
                
                if transform and data != self.fillna:
                    past_data.append(transform(data))
                    #print(f"transform data: {transform(data)}")
                else:
                    past_data.append(data)
                    #print(f"data: {data}")
        return past_data

    def agg_past_data(agg_func: Callable, seconds: int, start_time: int=get_time()) -> float:
        """aggregate past data into a single number.
        """
        pass



class PostureSubMetricTs(PostureMetricTs):
    """
    derivative metrics from existing postureMetricTs,
    takes in a dictionary,

    .second_to_frame_scores
    .second_to_avg_frame_scores

    when it updates, supports get_past_data

    metric_func: Callable[[Dict[str, PostureMetricTs]], float]
    """
    
    # with or without pointer, store the dictionary, or it can change
    def update(self, metricTsDict: Dict[str, PostureMetricTs]):
        """ add postuer data per frame. """
        time_hash = get_time()
        score = self.get_metric(metricTsDict)
        logger.debug(f"updating {self.name} with {score} at {time_hash}")
        self.second_to_frame_scores[time_hash].append(score) #data is partitioned by integer time
        self.second_to_avg_frame_scores[time_hash] = np.mean(self.second_to_frame_scores[time_hash])


def if_metric_fail_avg_and_last_second(
    metric_name: str, 
    threshold_rule: Callable[[float], int], 
    seconds: int,
    percent: float) -> Callable:
    """
    create a higher order function which the avg
    """
    def metric_func(metricTsDict: Dict[str, PostureMetricTs]) -> float:
        #print(f"high order funct seconds {seconds}")
        #logger.debug(f"metric data: {metricTsDict[metric_name].get_past_data(seconds=3)}")
        last_second_match_threshold = \
                metricTsDict[metric_name].get_past_data(
                    seconds=1, 
                transform=threshold_rule)
        #logger.debug(f"last_second_match_threshold: {last_second_match_threshold}")
        if last_second_match_threshold and last_second_match_threshold[0] == 1:
            past_data_match_threshold = \
                    metricTsDict[metric_name].get_past_data(seconds=
                    seconds, transform=threshold_rule)
            #logger.debug(f"past_data_match_threshold: {past_data_match_threshold }")
            return last_second_match_threshold and np.mean(past_data_match_threshold) >= percent
        return 0
    return metric_func 