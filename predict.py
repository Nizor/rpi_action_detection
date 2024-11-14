import mediapipe as mp
import numpy as np
import cv2
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

class SignLanguagePredictor:
    ACTION_NAMES = ['hello', 'thanks', 'I love you']

    def __init__(self, model):
        self.model = model
        self.sequence = deque(maxlen=30)  # Sliding window of 30 frames

    def predict(self, frame):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)
            
            if len(self.sequence) == 30:
                input_data = np.array(self.sequence).reshape(1, 30, -1)
                prediction = self.model.predict(input_data)
                sign_index = np.argmax(prediction)
                action = self.ACTION_NAMES[sign_index]
                return action
            return None
