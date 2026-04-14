import streamlit as st
import cv2
import numpy as np
import pickle
import os
import csv
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from data_manager import parse_label
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# ---------------- UI CONFIG ---------------- #
st.set_page_config(page_title="Face Attendance System", layout="wide")

st.title("🎯 Smart Face Attendance System")

# ---------------- LOAD DATA ---------------- #
BASE_DIR = os.path.dirname(__file__)

face_path = os.path.join(BASE_DIR, "data", "faces_data.pkl")
name_path = os.path.join(BASE_DIR, "data", "names.pkl")

faces = np.array(pickle.load(open(face_path, "rb")))
labels = pickle.load(open(name_path, "rb"))

faces = faces / 255.0
faces = normalize(faces)

model = KNeighborsClassifier(n_neighbors=5, metric="cosine")
model.fit(faces, labels)

# ---------------- ATTENDANCE FILE ---------------- #
attendance_dir = os.path.join(BASE_DIR, "Attendance")
os.makedirs(attendance_dir, exist_ok=True)

date = datetime.now().strftime("%d-%m-%Y")
file_path = os.path.join(attendance_dir, f"{date}.csv")

if not os.path.exists(file_path):
    with open(file_path, "w", newline="") as f:
        csv.writer(f).writerow(["ID", "NAME", "DATE", "TIME"])

# ---------------- SESSION STATE ---------------- #
if "marked" not in st.session_state:
    st.session_state.marked = set()

if "present" not in st.session_state:
    st.session_state.present = []

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Controls")
run = st.sidebar.checkbox("Start Camera")

st.sidebar.markdown("### 📌 Instructions")
st.sidebar.info("Enable camera to start face recognition attendance")

# ---------------- VIDEO PROCESSING ---------------- #
class FaceTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces_detected:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)
            face = normalize(face / 255.0)

            distances, _ = model.kneighbors(face)
            pred = model.predict(face)[0]
            dist = distances[0][0]

            name, uid = parse_label(pred)

            if dist < 0.35:
                label = f"{name} ({uid})"
                color = (0, 255, 0)

                if uid not in st.session_state.marked:
                    st.session_state.marked.add(uid)
                    st.session_state.present.append(label)

                    with open(file_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            uid,
                            name,
                            date,
                            datetime.now().strftime("%H:%M:%S")
                        ])
            else:
                label = "UNKNOWN"
                color = (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera")
    if run:
        webrtc_streamer(key="face", video_transformer_factory=FaceTransformer)
    else:
        st.warning("Camera is OFF")

with col2:
    st.subheader("👥 Present Students")

    if st.session_state.present:
        for p in st.session_state.present:
            st.success(p)
    else:
        st.info("No attendance yet")

    st.subheader("📊 Attendance Log")

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            st.text(f.read())
