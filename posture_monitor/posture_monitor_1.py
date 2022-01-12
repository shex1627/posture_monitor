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


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose 

    font = cv2.FONT_HERSHEY_SIMPLEX
    #headdown_metric = PostureState(score_func=lambda landmarks: np.abs(landmarks[10].y - landmarks[11].y),if_bad_posture=lambda score: int(score < 0.1), dir="posture_data")
    # to-do, make all those configurations templates, or initialize those somewhere else
    headdown_metric = PostureMetricTs("headdown", metric_func=lambda landmarks: min(landmarks[6].y, landmarks[3].y), data_dir="posture_data")
    leanleft_metric = PostureMetricTs("leanleft", metric_func=lambda landmarks: np.abs(landmark_lst[11].x - landmark_lst[9].x), data_dir="posture_data")
    leanright_metric = PostureMetricTs("leanright", metric_func=lambda landmarks: np.abs(landmark_lst[12].x - landmark_lst[10].x), data_dir="posture_data")

    

    
    headdown_3_seconds_metric_funct = if_metric_fail_avg_and_last_second("headdown", 
    lambda head_level: head_level > 0.31, seconds=10, percent=1.00)
    headdown_3_seconds = PostureSubMetricTs("headdown_seconds", headdown_3_seconds_metric_funct,
    "headdown_seconds")
    headdown_alert = PostureKDeltaAlert('headdown_3_seconds', headdown_3_seconds, 1)
    #int(metricDict['headdown'].get_past_data(10, transform=lambda metric: int(metric >))) > 0.31)
    #leanleft_alert = PostureAlertRule(alert_rule=lambda metricDict: int(metricDict['leanleft']) < 0.11)
    #leanright_alert = PostureAlertRule(alert_rule=lambda metricDict: int(metricDict['leanleft']) < 0.11)

    metricTsDict = {    
        headdown_metric.name: headdown_metric,
        headdown_3_seconds.name: headdown_3_seconds,
        #leanleft_metric.name: leanleft_metric,
        #leanright_metric.name: leanright_metric
    }
    pSession = PostureSession(metricTsDict, [headdown_alert], data_dir="test_dir") #AlertRule(alert_rule=lambda metricDict: 

    
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
                            slope, y_intercept = calculate_landmark_line(landmark_lst[12], landmark_lst[11])
                            
                            lip_shoulder_y_diff =np.abs(landmark_lst[10].y - landmark_lst[11].y)
                            eye_shoulder_y_diff =np.abs(min([landmark_lst[6].y, landmark_lst[3].y]) - max([landmark_lst[12].y, landmark_lst[11].y]))
                            right_lip_shoulder_x_diff =np.abs(landmark_lst[12].x - landmark_lst[10].x)
                            left_lip_shoulder_x_diff =np.abs(landmark_lst[11].x - landmark_lst[9].x)
                            
                            eye_pos = min(landmark_lst[6].y, landmark_lst[3].y)
                            logging.info(f"alert_toggle: {alert_toggle}")
                            
                            #in_bad_posture = headdown_metric.bad_posture()
                            if alert_toggle:
                                #headdown_metric.check_posture()
                                
                                pSession.update_metrics(landmark_lst)
                                #
                                headdown_data = pSession.metrics['headdown'].get_past_data(start_time=get_time(), seconds=5)
                                logger.info(f"headdown metric: {[np.round(data, 2) for data in headdown_data]}")
                                logger.info(f"headdown seconds metric: {headdown_3_seconds.get_past_data(seconds=1)}")
                                #headdown_metric.update(landmark_lst)
                                #leanleft_metric.update(landmark_lst)
                                #leanright_metric.update(landmark_lst)
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
                        # 
                        image_flip = cv2.flip(image, 1)
                        if results.pose_landmarks:
                            y_increment = 40
                            y_init = 300# - y_increment
                            font_scale = 0.8
                            if alerts_trigger:
                                red_background = np.full((image_flip.shape[0], image_flip.shape[1], 3), (0, 0, 255),dtype=np.uint8)
                                image_flip = cv2.addWeighted(image_flip, 0.5, red_background, 0.5, 0)
                            cv2.putText(image_flip, f"eye_pos: {np.round(eye_pos, 2)}", (10,y_init-y_increment*2), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image_flip, f"lip_shoulder_y_diff: {np.round(lip_shoulder_y_diff, 2)}", (10,y_init-y_increment*1), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(image_flip, f"eye_shoulder_y_diff: {np.round(eye_shoulder_y_diff, 2)}", (10,y_init), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(image_flip, f"left_lip_shoulder_x_diff: {np.round(left_lip_shoulder_x_diff, 2)}", (10,y_init+y_increment*1), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(image_flip, f"right_lip_shoulder_x_diff: {np.round(right_lip_shoulder_x_diff, 2)}", (10,y_init+y_increment*2), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
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