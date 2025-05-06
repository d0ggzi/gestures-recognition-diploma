import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from frameworks.mediapipe.process_data import MediapipeHands

IMAGE_SIZE = (128, 128)
PCA_COMPONENTS = 50
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
SMOOTHING_WINDOW = 5
KNN_NEIGHBORS = 25


def load_data(dataset_dir, confidence: float):
    X, y = [], []
    mediapipe_hands = MediapipeHands(confidence=confidence)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    for label in classes:
        class_dir = os.path.join(dataset_dir, label)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(class_dir, fname)
            image = cv2.imread(path)
            handmarks = mediapipe_hands.process(image)
            if handmarks is not None:
                X.append(handmarks)
                y.append(label)
    return np.array(X), np.array(y)


DATASET_DIR = "../../data/gestures"
for confidence in [0.25, 0.5, 0.75]:
    for c in [0.1, 1.0, 5.0]:
        X, y = load_data(DATASET_DIR, confidence=confidence)
        print(f"Загружено {len(y)} изображений из датасета с confidence={confidence}")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2
        )

        model = SVC(kernel='rbf', C=c, gamma='scale')
        # knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f"Точность на тесте с confidence={confidence}:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=le.classes_))

# def realtime_recognition():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Не удалось открыть камеру")
#         return
#     history = deque(maxlen=SMOOTHING_WINDOW)
#     print("Нажмите 'q' для выхода")
#
#     prev_time = time.time()
#     frame_counter = 0
#     fps = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         handmarks = mediapipe_hands.process(frame)
#
#         frame_counter += 1
#         if frame_counter >= 10:
#             curr_time = time.time()
#             fps = frame_counter / (curr_time - prev_time)
#             prev_time = curr_time
#             frame_counter = 0
#
#         if handmarks is not None:
#             pred = model.predict([handmarks])[0]
#             history.append(pred)
#             # сглаживание по окну
#             most_common = max(set(history), key=history.count)
#             gesture = le.inverse_transform([most_common])[0]
#             cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         else:
#             cv2.putText(
#                 frame, "Рука не найдена", (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
#             )
#         cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         cv2.imshow("Real-time", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     realtime_recognition()
