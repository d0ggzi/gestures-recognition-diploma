import cv2
import numpy as np
import os
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog

IMAGE_SIZE = (128, 128)
PCA_COMPONENTS = 50
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
SMOOTHING_WINDOW = 5


def segment_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([100, 20, 70])
    upper_hsv = np.array([180, 100, 200])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = frame.shape[0] * frame.shape[1]

    if area < 1000 or area > 0.8 * img_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi = frame[y:y + h, x:x + w]
    return roi if roi.size > 0 else None


def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    features = hog(
        resized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys'
    )
    return features


def load_data(dataset_dir):
    X, y = [], []
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    for label in classes:
        class_dir = os.path.join(dataset_dir, label)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(class_dir, fname)
            image = cv2.imread(path)
            if image is None:
                continue
            roi = segment_hand(image)
            if roi is None:
                continue
            feat = extract_hog(roi)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)


DATASET_DIR = "../../data/gestures"
X, y = load_data(DATASET_DIR)
print(f"Загружено {len(y)} изображений из датасета")

le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=PCA_COMPONENTS)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Оценка
y_pred = svm.predict(X_test)
print("Точность на тесте:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))


def realtime_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return
    history = deque(maxlen=SMOOTHING_WINDOW)
    print("Нажмите 'q' для выхода")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = segment_hand(frame)
        if roi is not None:
            feat = extract_hog(roi)
            feat_norm = scaler.transform([feat])
            feat_pca = pca.transform(feat_norm)
            pred = svm.predict(feat_pca)[0]
            history.append(pred)
            most_common = max(set(history), key=history.count)
            gesture = le.inverse_transform([most_common])[0]
            cv2.putText(
                frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            disp = cv2.resize(roi, (128, 128))
            frame[10:138, 10:138] = disp
        else:
            cv2.putText(
                frame, "Рука не найдена", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
        cv2.imshow("Real-time HOG+SVM", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_recognition()
