import cv2
import pickle
import numpy as np
import os
import csv
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from data_manager import parse_label
from ui import draw_ui

BASE_DIR = os.path.dirname(__file__)

face_path = os.path.join(BASE_DIR, "data", "faces_data.pkl")
name_path = os.path.join(BASE_DIR, "data", "names.pkl")
cascade_path = os.path.join(BASE_DIR, "data", "haarcascade_frontalface_default.xml")

attendance_dir = os.path.join(BASE_DIR, "Attendance")
os.makedirs(attendance_dir, exist_ok=True)

# ---------------- LOAD ---------------- #
if not os.path.exists(face_path) or not os.path.exists(name_path):
    print("❌ No data found. Run add_faces.py")
    exit()

FACES = pickle.load(open(face_path, "rb"))
LABELS = pickle.load(open(name_path, "rb"))

if len(FACES) == 0:
    print("❌ Empty dataset")
    exit()

FACES = np.array(FACES) / 255.0

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(FACES, LABELS)

# ---------------- CAMERA ---------------- #
video = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cascade_path)

marked = set()
present = []

date = datetime.now().strftime("%d-%m-%Y")
file = os.path.join(attendance_dir, f"{date}.csv")

if not os.path.exists(file):
    with open(file, "w", newline='') as f:
        csv.writer(f).writerow(["ID", "NAME", "DATE", "TIME"])

# ---------------- LOOP ---------------- #
cv2.namedWindow("Face Attendance")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # ✅ FIX: Detect close button
    if cv2.getWindowProperty("Face Attendance", cv2.WND_PROP_VISIBLE) < 1:
        break

    frame = cv2.resize(frame, (1000, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        face = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1) / 255.0

        dist, _ = knn.kneighbors(face)
        pred = knn.predict(face)[0]

        d = np.mean(dist)
        name, uid = parse_label(pred)

        if d < 2000:
            text = f"{name} ({uid})"
            color = (0, 200, 0)

            if uid not in marked:
                marked.add(uid)
                present.append(text)

                with open(file, "a", newline='') as f:
                    csv.writer(f).writerow(
                        [uid, name, date, datetime.now().strftime("%H:%M:%S")]
                    )
        else:
            text = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    frame = draw_ui(frame, present)

    cv2.imshow("Face Attendance", frame)

    # ✅ Q key exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ---------------- #
video.release()
cv2.destroyAllWindows()