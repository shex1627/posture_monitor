import cv2
import mediapipe as mp
import logging
import sys
import time
import numpy as np
import os
from posture_monitor.src.util import get_time
from pynput.keyboard import Listener
from pynput.keyboard import Key
from pynput import keyboard
import argparse
from posture_monitor.src.PostureAlertRule import *
from posture_monitor.src.PostureMetricTs import *


from posture_monitor.src.util import calculate_landmark_line, get_time
from posture_monitor.src.PostureSession import PostureSession


file_handler = logging.FileHandler(filename=f'logs/posture_monitor/posture_monitor_{int(time.time())}.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    handlers=handlers
)
logger = logging.getLogger('posture_monitor')

# variable and functions for toggle feature
program_on = True
alert_toggle = True
KEY_ALERT_ON = Key.f7
KEY_ALERT_OFF = Key.f8
KEY_EXIT = Key.f9

def on_press_start(key, key_alert_on=KEY_ALERT_ON, exit=KEY_EXIT):
        global alert_toggle
        global program_on
        #print("Key pressed: {0}".format(key))
        if key == key_alert_on:
            logging.info("alert on")
            alert_toggle = True
            return False

        if key == exit:
            print('exiting...')
            sys.exit()


def on_press_loop(key, key_alert_off=KEY_ALERT_OFF, exit=KEY_EXIT):
    global alert_toggle
    global program_on
    if key == KEY_ALERT_ON:
        print('starting auto-cast')
        alert_toggle = True
        return True
    
    if key == key_alert_off:
        print('stopping auto-cast')
        alert_toggle = False
        return True

    if key == exit:
        print('exiting...')
        program_on = False
        sys.exit()

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose 

    font = cv2.FONT_HERSHEY_SIMPLEX
    # to-do, make all those configurations templates, or initialize those somewhere else
    headdown_metric = PostureMetricTs("headdown", metric_func=lambda landmarks: min(landmarks[6].y, landmarks[3].y))
    left_shoulder_y_metric = PostureMetricTs("left_shoulder_y", metric_func=lambda landmarks: landmark_lst[11].y)
    right_shoulder_y_metric =  PostureMetricTs("right_shoulder_y", metric_func=lambda landmarks: landmark_lst[12].y)
    left_shoulder_x_metric = PostureMetricTs("left_shoulder_x", metric_func=lambda landmarks: landmark_lst[11].x)
    right_shoulder_x_metric =  PostureMetricTs("right_shoulder_x", metric_func=lambda landmarks: landmark_lst[12].x)
    

    
    headdown_10_seconds_metric_funct = if_metric_fail_avg_and_last_second("headdown", 
    lambda head_level: head_level > 0.31, seconds=10, percent=1.00)
    headdown_10_seconds = PostureSubMetricTs("headdown_seconds", headdown_10_seconds_metric_funct)
    headdown_alert = PostureKDeltaAlert('headdown_10_seconds', headdown_10_seconds, 1)

    rightshoulder_down_metric_funct = if_metric_fail_avg_and_last_second("right_shoulder_y", 
    lambda right_shoulder_y: right_shoulder_y >= 0.57, seconds=2, percent=1.00)
    rightshoulder_down_10_seconds = PostureSubMetricTs("rightshoulder_down_seconds", rightshoulder_down_metric_funct)    
    rightshoulder_down_alert = PostureKDeltaAlert('rightshoulder_10s_down', rightshoulder_down_10_seconds, 1)

    left_right_shoulder_y_diff = PostureMetricTs("left_right_shoulder_y_diff", metric_func=lambda landmarks: np.abs(landmark_lst[11].y - landmark_lst[12].y))
    shoulder_tilt_metric_funct = if_metric_fail_avg_and_last_second("left_right_shoulder_y_diff", lambda level_diff: level_diff > 0.02, seconds=3, percent=1.00)
    shoulder_tilt_seconds = PostureSubMetricTs("shoulder_tilt_seconds", shoulder_tilt_metric_funct)
    shoulder_tilt_alert = PostureKDeltaAlert("shoulder_tilt_seconds", shoulder_tilt_seconds, 1)

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
    pSession = PostureSession(metricTsDict, [headdown_alert, shoulder_tilt_alert], data_dir="test_dir") #AlertRule(alert_rule=lambda metricDict: #headdown_alert, 

    if program_on:
        with Listener(on_press=on_press_start) as listener:
            listener.join() # wait for F11...

            with Listener(on_press=on_press_loop) as listener:
                # For webcam input:
                cap = cv2.VideoCapture(0)
                with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            print("Ignoring empty camera frame.")
                            # If loading a video, use 'break' instead of 'continue'.
                            continue

                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = pose.process(image)

                        # log landmarks
                        if results.pose_landmarks:
                            landmark_lst = results.pose_landmarks.ListFields()[0][1]
                            logging.info(f"alert_toggle: {alert_toggle}")
                            
                            if alert_toggle:
                                pSession.update_metrics(landmark_lst)
                                alerts_trigger = pSession.check_posture_alert()
                                logger.debug(f"alert trigger: {alerts_trigger}")

                        # Draw the pose annotation on the image.
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        
                        image_flip = cv2.flip(image, 1)
                        # Flip the image horizontally for a selfie-view display.
                        cv2.imshow('MediaPipe Pose', image_flip)
                        cv2.setWindowProperty('MediaPipe Pose', cv2.WND_PROP_TOPMOST, 1)
                        # leave app when click `esc` key
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
                cap.release()
                sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input_file", help="data file with filename field", default=False)
    args = parser.parse_args()
    main()

    # run main