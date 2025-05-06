import time

import cv2
import mediapipe as mp


class Gestures:
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.cap = cv2.VideoCapture(0)

    def run(self):
        with self.mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7) as hands:
            while self.cap.isOpened() and self._run_flag:
                success, image = self.cap.read()
                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image).multi_hand_landmarks
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                h, w, _ = image.shape
                if results:
                    for hand_landmarks in results:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                cv2.imshow("mediapipe", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    gestures = Gestures()
    gestures.run()
