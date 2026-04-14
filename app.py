import streamlit as st
import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from data_manager import parse_label

st.set_page_config(page_title="Live Face Attendance", layout="wide")

# ---------------- PATH ---------------- #
BASE_DIR = os.path.dirname(__file__)
face_path = os.path.join(BASE_DIR, "data", "faces_data.pkl")
name_path = os.path.join(BASE_DIR, "data", "names.pkl")
cascade_path = os.path.join(BASE_DIR, "data", "haarcascade_frontalface_default.xml")

# ---------------- LOAD MODEL ---------------- #
if not os.path.exists(face_path):
    st.error("No training data found!")
    st.stop()

FACES = pickle.load(open(face_path, "rb"))
LABELS = pickle.load(open(name_path, "rb"))

FACES = np.array(FACES) / 255.0

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(FACES, LABELS)

face_detector = cv2.CascadeClassifier(cascade_path)

present_set = set()

# ---------------- VIDEO PROCESSOR ---------------- #
class FaceRecognition(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop = img[y:y+h, x:x+w]
            face = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1) / 255.0

            dist, _ = knn.kneighbors(face)
            pred = knn.predict(face)[0]

            d = np.mean(dist)
            name, uid = parse_label(pred)

            if d < 2000:
                label = f"{name} ({uid})"
                color = (0, 255, 0)

                # Save attendance once
                if uid not in present_set:
                    present_set.add(uid)

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

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img


# ---------------- UI ---------------- #
st.title("🎥 Live Face Attendance System")
st.write("Click START to begin attendance")

webrtc_streamer(
    key="attendance",
    video_processor_factory=FaceRecognition
)

st.subheader("✅ Present Students (Session)")

for uid in present_set:
    st.success(uid)
