import time

import cv2
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D, PReLU, InputLayer

from frameworks.mediapipe.process_data import MediapipeHands

IMG_SIZE = (64, 64)
EPOCHS = 100
BATCH_SIZE = 32
SMOOTHING_WINDOW = 5
MODEL_PATH = "with_mediapipe.h5"
CONFIDENCE = 0.75


def build_advanced_cnn(input_dim, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    return model


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


MODEL_PATH = f"model_conf_{CONFIDENCE}.h5"
print(f"Загрузка датасета...")
X, y = load_data("../../data/gestures", confidence=CONFIDENCE)
print("Датасет загружен:", X.shape)

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat)

input_shape = X_train.shape[1]
num_classes = len(le.classes_)

if not os.path.exists(MODEL_PATH):
    print("Обучение модели...")
    model = build_advanced_cnn(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    model.save(MODEL_PATH)
    print("Модель сохранена")
else:
    print("Загрузка обученной модели...")
    model = load_model(MODEL_PATH)


def get_classification_report(model):
    _, X_test, _, y_test = train_test_split(X, y_enc, test_size=0.2)

    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    report = classification_report(y_test, predicted_classes, target_names=le.classes_)
    print(report)


def realtime_recognition():
    mediapipe_hands = MediapipeHands(confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка камеры")
        return
    history = deque(maxlen=SMOOTHING_WINDOW)
    print("Распознавание. Нажмите 'q' для выхода")

    prev_time = time.time()
    frame_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        handmarks = mediapipe_hands.process(frame)
        gesture, confidence = "Не распознано", 0

        if handmarks is not None:
            handmarks = np.array(handmarks, dtype=np.float32).reshape(1, -1)
            pred = model.predict(handmarks, verbose=0)[0]
            class_id = np.argmax(pred)
            confidence = pred[class_id]
            history.append(class_id)
            gesture = le.inverse_transform([max(set(history), key=history.count)])[0]

        if frame_counter >= 10:
            curr_time = time.time()
            fps = frame_counter / (curr_time - prev_time)
            prev_time = curr_time
            frame_counter = 0

        cv2.putText(frame, f"Gesture: {gesture} ({confidence * 100:.1f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Real time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # get_classification_report(model)
    realtime_recognition()
