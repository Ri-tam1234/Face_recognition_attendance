import streamlit as st
try:
    import cv2
except:
    import streamlit as st
    st.error("OpenCV not supported in cloud. Switching mode...")
    st.stop()
import pickle
import numpy as np
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

from data_manager import parse_label

# ---------------- PATHS ---------------- #
BASE_DIR = os.path.dirname(__file__)

face_path = os.path.join(BASE_DIR, "data", "faces_data.pkl")
name_path = os.path.join(BASE_DIR, "data", "names.pkl")
cascade_path = os.path.join(BASE_DIR, "data", "haarcascade_frontalface_default.xml")

# ---------------- LOAD MODEL ---------------- #
if not os.path.exists(face_path):
    st.error("No data found. Please add faces first.")
    st.stop()

FACES = pickle.load(open(face_path, "rb"))
LABELS = pickle.load(open(name_path, "rb"))

if len(FACES) == 0:
    st.error("Empty dataset")
    st.stop()

FACES = np.array(FACES) / 255.0

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(FACES, LABELS)

face_detector = cv2.CascadeClassifier(cascade_path)

# ---------------- UI ---------------- #
st.title("📸 Smart Face Attendance System")

present = []

# ---------------- CAMERA ---------------- #
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        face = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1) / 255.0

        dist, _ = knn.kneighbors(face)
        pred = knn.predict(face)[0]

        d = np.mean(dist)
        name, uid = parse_label(pred)

        if d < 2000:
            label = f"{name} ({uid})"
            color = (0, 255, 0)

            if label not in present:
                present.append(label)

                # Save attendance
                date = datetime.now().strftime("%d-%m-%Y")
                time_now = datetime.now().strftime("%H:%M:%S")

                os.makedirs("Attendance", exist_ok=True)
                file = f"Attendance/{date}.csv"

                if not os.path.exists(file):
                    with open(file, "w") as f:
                        f.write("ID,NAME,DATE,TIME\n")

                with open(file, "a") as f:
                    f.write(f"{uid},{name},{date},{time_now}\n")

        else:
            label = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.image(frame, channels="BGR")

# ---------------- PRESENT LIST ---------------- #
st.subheader("✅ Present Students")

for p in present:
    st.success(p)
