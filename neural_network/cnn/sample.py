import time

import cv2
import numpy as np
import os
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

# === Параметры ===
IMG_SIZE = (64, 64)
EPOCHS = 50
BATCH_SIZE = 32
SMOOTHING_WINDOW = 5
MODEL_PATH = "gesture_cnn.h5"

def build_advanced_cnn(input_shape, num_classes):
    model = Sequential()

    # Блок 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Блок 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # Блок 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Глобальный пуллинг и полносвязные слои
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# === Загрузка данных ===
def load_data(dataset_dir):
    X, y = [], []
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(class_dir, file)
                img = cv2.imread(path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, IMG_SIZE)
                X.append(resized)
                y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1).astype('float32') / 255.0
    y = np.array(y)
    return X, y

print("Загрузка датасета...")
X, y = load_data("../../data/gestures")
print("Датасет загружен:", X.shape)

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat)

input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
num_classes = len(le.classes_)

# === Создание модели ===
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

# === Реалтайм-предсказание ===
def realtime_recognition():
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

        # === Обработка кадра ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        input_img = resized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1).astype('float32') / 255.0
        pred = model.predict(input_img, verbose=0)[0]
        class_id = np.argmax(pred)
        confidence = pred[class_id]
        history.append(class_id)
        gesture = le.inverse_transform([max(set(history), key=history.count)])[0]

        # === Подсчет FPS ===
        if frame_counter >= 10:
            curr_time = time.time()
            fps = frame_counter / (curr_time - prev_time)
            prev_time = curr_time
            frame_counter = 0

        # === Вывод ===
        cv2.putText(frame, f"Gesture: {gesture} ({confidence*100:.1f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Real time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_recognition()
