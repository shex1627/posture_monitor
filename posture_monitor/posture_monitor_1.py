import cv2
import mediapipe as mp
import logging
import sys
import time
import numpy as np
import os
from pathlib import Path
from posture_monitor.src.util import get_time
from pynput.keyboard import Listener
from pynput.keyboard import Key
from pynput import keyboard
import argparse
from posture_monitor.src.PostureAlertRule import *
from posture_monitor.src.PostureMetricTs import *


from posture_monitor.src.util import calculate_landmark_line, get_time, create_basic_landmarks
from posture_monitor.src.PostureSession import PostureSession

from posture_monitor.session_config import posture_session_args, HEAD_LEVEL_THRESHOLD, SHOULDER_TILT_THRESHOLD

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
POSTURE_DATA_DIR = 'posture_data' 
#POSTURE_DATA_DIR = os.path.join(Path(__file__).resolve().parents[0], "posture_data")

WINDOW_NAME = "Posture Monitor"
SCALE_PERCENT = 100
WINDOW_ON_TOP = True
camera_on = True
sound_alert_on = True
track_data_on = True
program_on = True
KEY_ALERT_TOGGLE = Key.f6
KEY_TRACK_DATA_TOGGLE = Key.f7
KEY_CAMERA_TOGGLE = Key.f8
KEY_EXIT = Key.f4


def on_press_loop(key):
    global sound_alert_on
    global track_data_on
    global camera_on
    global program_on

    if key == KEY_TRACK_DATA_TOGGLE:
        """turn off alert if data tracking is off."""
        track_data_on = not track_data_on
        if not track_data_on:
            sound_alert_on = False
        logging.info(f"track data toggle to {track_data_on}")
        return True

    if key == KEY_ALERT_TOGGLE:
        """turn on data tracking if alert is on as well."""
        sound_alert_on = not sound_alert_on
        if sound_alert_on:
            track_data_on = True
        logging.info(f"alert toggle to {sound_alert_on}")
        return True
    
    if key == KEY_CAMERA_TOGGLE:
        camera_on = not camera_on
        logging.info(f"camera toggle to {camera_on}")
        return True

    if key == KEY_EXIT:
        """turn off camera before program exits"""
        camera_on = not camera_on
        program_on = False
        logging.info(f"exiting program")
        return True

def main():
    global SCALE_PERCENT
    global WINDOW_ON_TOP
    global TRACK_BASIC_LANDMARK
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose 

    font = cv2.FONT_HERSHEY_SIMPLEX
    if TRACK_BASIC_LANDMARK:
        basic_landmarks_metriTS_dict = create_basic_landmarks()
        posture_session_args['metricTsDict'].update(basic_landmarks_metriTS_dict)
    pSession = PostureSession(**posture_session_args, data_dir=POSTURE_DATA_DIR)
    
    cap = cv2.VideoCapture(0)
    while program_on:
        with Listener(on_press=on_press_loop) as listener:
            # For webcam input:
            print("loop listener")

            if not camera_on:
                cv2.destroyAllWindows()
                cap.release()
            else:
                cv2.namedWindow(WINDOW_NAME , cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(WINDOW_NAME , 20,20)
                if WINDOW_ON_TOP:
                    cv2.setWindowProperty(WINDOW_NAME , cv2.WND_PROP_TOPMOST, 1)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)

            with mp_pose.Pose(
                min_detection_confidence=0.8,
                min_tracking_confidence=0.5) as pose:
                while camera_on and cap.isOpened():
                    
                    # initalize alert trigger is false
                    logging.info(f"sound_alert_on: {sound_alert_on}")
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
                            
                            logger.info(f"headdown: {min(landmark_lst[6].y, landmark_lst[3].y)}")

                            alerts_trigger = pSession.check_posture_alert(trigger_sound=sound_alert_on) #
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
                    #logging.info(f"infront computer: {pSession.metrics['infront_computer'].get_past_data(seconds=1)[0]}")
                    if track_data_on and (pSession.metrics['infront_computer'].get_past_data(seconds=1)[0] <= 0): 
                        image_flip = cv2.cvtColor(image_flip, cv2.COLOR_BGR2GRAY)
                        y_increment = 120
                        y_init = 100# - y_increment
                        font_scale = 2
                        cv2.putText(img=image_flip, text="away from desk", org=(100,y_init), 
                            fontFace=font, 	fontScale=font_scale, color=(0, 0, 255), thickness=6, lineType=cv2.LINE_AA)

                    elif results.pose_landmarks and alerts_trigger:
                        y_increment = 120
                        y_init = 100# - y_increment
                        font_scale = 2.5
                        red_background = np.full((image_flip.shape[0], image_flip.shape[1], 3), (0, 0, 255),dtype=np.uint8)
                        image_flip = cv2.addWeighted(image_flip, 0.5, red_background, 0.5, 0)
                        for i in range(len(alerts_trigger)):
                            cv2.putText(img=image_flip, text=alerts_trigger[i], org=(100,y_init+i*y_increment), 
                            fontFace=font, 	fontScale=font_scale, color=(0, 255, 0), thickness=6, lineType=cv2.LINE_AA)
                    
                    # show on windows if tracking and alert are on
                    
                    y_init = 400
                    font_scale = 1
                    y_increment = 40 
                    
                    # if_good_posture = pSession.metrics['good_posture'].get_past_data(seconds=1)[0]
                    metricTsDict = pSession.metrics
                    # # shoulder_tilt_metric = metricTsDict['left_right_shoulder_y_diff'].get_past_data(seconds=1)[0]
                    # # headdown_metric = metricTsDict['headdown'].get_past_data(seconds=1)[0]
                    # # bad_posture = [metricTsDict['left_right_shoulder_y_diff'].get_past_data(seconds=1)[0] > SHOULDER_TILT_THRESHOLD,
                    # #         metricTsDict['headdown'].get_past_data(seconds=1)[0] > HEAD_LEVEL_THRESHOLD]
                    shift_metric_name_lst = ['left_eye_shoulder_x_diff', "right_eye_shoulder_x_diff", "left_shoulder_elbow_x_diff", "right_shoulder_elbow_x_diff"] #, "right_eye_shoulder_x_diff", "left_shoulder_elbow_x_diff", "right_shoulder_elbow_x_diff"
                    metric_to_display = { 
                        metric_name: np.round(metricTsDict[metric_name].get_past_data(seconds=1)[0], 2)
                        for metric_name in shift_metric_name_lst
                    }
                    i_counter = 1
                    for name, value in metric_to_display.items():
                        cv2.putText(img=image_flip, text=f"{name}: {value}", org=(0, y_init-i_counter*y_increment), 
                            fontFace=font, 	fontScale=font_scale, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
                        i_counter += 1

                    cv2.putText(img=image_flip, text=f"sound_alert_on: {sound_alert_on}", org=(0, y_init), 
                            fontFace=font, 	fontScale=font_scale, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
                    cv2.putText(img=image_flip, text=f"track_data_on: {track_data_on}", org=(0,y_init+y_increment), 
                            fontFace=font, 	fontScale=font_scale, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

                    # Flip the image horizontally for a selfie-view display.
                    width = int(image_flip.shape[1] * SCALE_PERCENT / 100)
                    height = int(image_flip.shape[0] * SCALE_PERCENT / 100)
                    dim = (width, height)
                    print(dim)
                    image_resized = cv2.resize(image_flip, dsize=dim, interpolation = cv2.INTER_AREA)
                    cv2.imshow(WINDOW_NAME , image_resized)

                    # idk why opencv doesn't work without it
                    if cv2.waitKey(5) & 0xFF == 27:
                        pSession.export_data()
                        break
            
    #cap.release()
    pSession.export_data()
    sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input_file", help="data file with filename field", default=False)
    parser.add_argument('--window_size',
                    default='small',
                    #const='all',
                    #nargs='?',
                    choices=['small', 'medium', 'large'],
                    help="windows size.")
    parser.add_argument('--track_basic_landmark',
                    help="flag for tracking basic head + shoulder landmarks", action="store_true")
    # this should default true
    parser.add_argument('--window_stay_top',
                    help="flag to determine if window is always on top of all other windows", action="store_true")
    parser.add_argument('--sound_alert_on',
                    help="flag to determine if alert is on", action="store_true")
    args = parser.parse_args()
    sound_alert_on = args.sound_alert_on
    WINDOW_ON_TOP = args.window_stay_top
    TRACK_BASIC_LANDMARK = args.track_basic_landmark
    window_size_mapping = {
        'small': 50,
        'medium':100,
        'large': 200
    }
    SCALE_PERCENT = window_size_mapping[args.window_size]
    main()