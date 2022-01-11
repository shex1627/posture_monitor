import cv2
import mediapipe as mp
import logging
import sys
import time
import numpy as np
from src.util import calculate_landmark_line
from src.PostureState import *

file_handler = logging.FileHandler(filename=f'logs/posture_landmark_{int(time.time())}.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    handlers=handlers
)
logger = logging.getLogger('posture_landmark')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose 

font = cv2.FONT_HERSHEY_SIMPLEX
headdown_posture = PostureState(score_func=lambda landmarks: np.abs(landmarks[10].y - landmarks[11].y),if_bad_posture=lambda score: int(score < 0.1), dir="posture_data")
# eye down
headdown_posture = PostureState(score_func=lambda landmarks: min(landmarks[6].y, landmarks[3].y),if_bad_posture=lambda score: int(score > 0.31), dir="posture_data")

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
            logging.info(f"eye position: {eye_pos}")
            headdown_posture.update(landmark_lst)
            #in_bad_posture = headdown_posture.bad_posture()
            headdown_posture.check_posture()

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
            if headdown_posture.bad_posture_frame():
                red_background = np.full((image_flip.shape[0], image_flip.shape[1], 3), (0, 0, 255),dtype=np.uint8)
                image_flip = cv2.addWeighted(image_flip, 0.5, red_background, 0.5, 0)
                cv2.putText(image_flip, f"eye_pos: {np.round(eye_pos, 2)}", (10,y_init-y_increment*2), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image_flip, f"lip_shoulder_y_diff: {np.round(lip_shoulder_y_diff, 2)}", (10,y_init-y_increment*1), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_flip, f"eye_shoulder_y_diff: {np.round(eye_shoulder_y_diff, 2)}", (10,y_init), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_flip, f"left_lip_shoulder_x_diff: {np.round(left_lip_shoulder_x_diff, 2)}", (10,y_init+y_increment*1), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_flip, f"right_lip_shoulder_x_diff: {np.round(right_lip_shoulder_x_diff, 2)}", (10,y_init+y_increment*2), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', image_flip)
        # leave app when click `esc` key
        if cv2.waitKey(5) & 0xFF == 27:
            break
            sys.exit()
cap.release()