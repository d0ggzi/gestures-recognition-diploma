import os

os.environ["GLOG_minloglevel"] = "2"  # чтоб не писал в логи много
import cv2
import mediapipe as mp


class MediapipeHands:
    def __init__(self, confidence: float):
        super().__init__()
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )

    def process(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image).multi_hand_landmarks
        if results:
            for hand_landmarks in results:
                keypoint_pos = []
                for landmark in hand_landmarks.landmark:
                    keypoint_pos.extend([landmark.x, landmark.y, landmark.z])
                return keypoint_pos
