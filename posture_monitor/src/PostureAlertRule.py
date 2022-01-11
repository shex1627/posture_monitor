
from typing import Dict, Callable
from dataclasses import dataclass
import numpy as np
from posture_monitor.src.PostureMetricTs import PostureMetricTs, PostureSubMetricTs
from posture_monitor.src.util import get_time
import logging

logger = logging.getLogger("PostureAlertRule")

class PostureAlertRule:
    def __init__(self, name: str):
        self.name = name

    def alert_trigger(self, alert_dict:Dict[str, PostureMetricTs]=None) -> int:
        return NotImplementedError

@dataclass
class PostureKDeltaAlert():
    name: str
    #alert_rule: Callable[[Dict[str, PostureMetricTs]], int]
    metric: PostureMetricTs
    delta: float

    def alert_trigger(self, alert_dict:Dict[str, PostureMetricTs]=None) -> int:
        """
        return
        1 if alert trigger
        0 if alert not trigger
        -1 if any metric missing
        """
        try:
            #last_second_metric_data = self.metric.get_past_data(second=
            data = self.metric.get_past_data()
            return int(data[0] == self.delta)
        except Exception as e:
            logging.error(e, exc_info=True)
            return -1
