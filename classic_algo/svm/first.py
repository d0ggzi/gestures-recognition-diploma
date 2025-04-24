import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def extract_features(image, size=(64, 64)):
    image = cv2.resize(image, size)
    return image.flatten()


def load_dataset(dataset_path):
    X, y = [], []
    classes = os.listdir(dataset_path)
    for label in classes:
        class_path = os.path.join(dataset_path, label)
        for file in os.listdir(class_path):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(class_path, file), cv2.IMREAD_GRAYSCALE)
                features = extract_features(img)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)


dataset_dir = "../../data/gestures"
X, y = load_dataset(dataset_dir)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


def recognize_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = 200, 100, 200, 200
        roi = gray[y:y + h, x:x + w]

        features = extract_features(roi)
        prediction = svm_model.predict([features])
        gesture = le.inverse_transform(prediction)[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Жест: {gesture}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Алгоритм SVM', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


recognize_from_camera()
