import os
import cv2
import mediapipe as mp


def process_image(path: str):
    image = cv2.imread(path)
    with mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image).multi_hand_landmarks
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        if results:
            for hand_landmarks in results:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
        while True:
            cv2.imshow("mediapipe", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


palm = "/home/dggz/code/diploma/data/mine-gestures/augmentated/palm/palm_original_Photo from 2025-04-22 22-42-52.701801.jpeg_00d75157-61b1-432d-abc7-8c62ca8062ce.jpeg"
like = "/home/dggz/code/diploma/data/mine-gestures/augmentated/like/like_original_Photo from 2025-04-22 22-43-04.408775.jpeg_2dc45212-db34-469e-80a3-e6a4cbbb4c0c.jpeg"
like_original = "/home/dggz/code/diploma/data/mine-gestures/originals/like/Photo from 2025-04-22 22-43-04.408775.jpeg"
like_proccessed = "/home/dggz/Downloads/photo_2025-05-03_14-51-46.jpg"
process_image(path=like_proccessed)
