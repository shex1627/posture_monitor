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

from posture_monitor.session_config import posture_session_args

file_handler = logging.FileHandler(filename=f'logs/posture_monitor/posture_monitor_{int(time.time())}.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    handlers=handlers
)
logger = logging.getLogger('posture_monitor')

# variable and functions for toggle feature
camera_on = True
alert_on = True
track_data_on = True
program_on = True
# KEY_ALERT_ON = Key.f7
# KEY_ALERT_OFF = Key.f8
# KEY_EXIT = Key.f6
KEY_ALERT_TOGGLE = Key.f6
KEY_TRACK_DATA_TOGGLE = Key.f7
KEY_CAMERA_TOGGLE = Key.f8
KEY_EXIT = Key.f4


# def on_press_start(key, key_alert_on=KEY_ALERT_ON, exit=KEY_EXIT):
#         global alert_toggle
#         global camera_on
#         #print("Key pressed: {0}".format(key))
#         if key == key_alert_on:
#             logging.info("alert on")
#             alert_toggle = True
#             return False

#         if key == exit:
#             print('exiting...')
#             sys.exit()


def on_press_loop(key):
    global alert_on
    global track_data_on
    global camera_on
    global program_on

    if key == KEY_TRACK_DATA_TOGGLE:
        track_data_on = not track_data_on
        if not track_data_on:
            alert_on = False
        logging.info(f"track data toggle to {track_data_on}")
        return True

    if key == KEY_ALERT_TOGGLE:
        alert_on = not alert_on
        if alert_on:
            track_data_on = True
        logging.info(f"alert toggle to {alert_on}")
        return True
    
    if key == KEY_CAMERA_TOGGLE:
        camera_on = not camera_on
        logging.info(f"camera toggle to {camera_on}")
        return True

    if key == KEY_EXIT:
        camera_on = not camera_on
        program_on = False
        logging.info(f"exiting program")
        return True

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose 

    font = cv2.FONT_HERSHEY_SIMPLEX
    pSession = PostureSession(**posture_session_args, data_dir="test_dir")
    
    cap = cv2.VideoCapture(0)
    while program_on:
        with Listener(on_press=on_press_loop) as listener:
            #listener.join()
            # For webcam input:
            print("loop listener")
            if not camera_on:
                cv2.destroyAllWindows()
            else:
                cv2.namedWindow("Posture Monitor", cv2.WINDOW_AUTOSIZE)
            with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
                while camera_on and cap.isOpened():
                    
                    # initalize alert trigger is false
                    logging.info(f"alert_on: {alert_on}")
                    logging.info(f"track_data_on: {track_data_on}")

                    alerts_trigger = []
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
                        if track_data_on:
                            pSession.update_metrics(landmark_lst)

                        if alert_on:
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
                    #
                    if results.pose_landmarks and alerts_trigger:
                        y_increment = 120
                        y_init = 100# - y_increment
                        font_scale = 2.5
                        red_background = np.full((image_flip.shape[0], image_flip.shape[1], 3), (0, 0, 255),dtype=np.uint8)
                        image_flip = cv2.addWeighted(image_flip, 0.5, red_background, 0.5, 0)
                        for i in range(len(alerts_trigger)):
                            cv2.putText(img=image_flip, text=alerts_trigger[i], org=(100,y_init+i*y_increment), 
                            fontFace=font, 	fontScale=font_scale, color=(0, 255, 0), thickness=6, lineType=cv2.LINE_AA)
                    # Flip the image horizontally for a selfie-view display.
                    scale_percent = 50 # percent of original size
                    width = int(image_flip.shape[1] * scale_percent / 100)
                    height = int(image_flip.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    #cv2.imshow('MediaPipe Pose', cv2.resize(image_flip, dsize=dim, interpolation = cv2.INTER_AREA))
                    image_resized = cv2.resize(image_flip, dsize=dim, interpolation = cv2.INTER_AREA)
                    cv2.imshow('Posture Monitor', image_resized)
                    #cv2.setWindowProperty('Posture Monitor', cv2.WND_PROP_TOPMOST, 1)
                    # leave app when click `esc` key
                    if cv2.waitKey(5) & 0xFF == 27:
                        pSession.export_data()
                        break
            
    #cap.release()
    pSession.export_data()
    sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input_file", help="data file with filename field", default=False)
    args = parser.parse_args()
    parser.add_argument('--window_size',
                    default='small',
                    #const='all',
                    #nargs='?',
                    choices=['small', 'median', 'large'],
                    help="windows size.")
    parser.add_argument('--track_basic_metric',
                    help="flag for tracking basic head + shoulder landmarks", action="store_true")
    # this should default true
    parser.add_argument('--windows_on_top',
                    help="flag to determine if window is always on top of all other windows", action="store_true")
    main()

    # run main