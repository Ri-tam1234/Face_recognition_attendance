import streamlit as st
import cv2
import numpy as np
import pickle
import os
from datetime import datetime

# ---------------- PATHS ---------------- #
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "faces_data.pkl")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "data", "attendance.pkl")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# ---------------- LOAD DATA ---------------- #
def load_faces():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as f:
            return pickle.load(f)
    return {"names": [], "faces": []}

def save_attendance(name):
    attendance = []

    if os.path.exists(ATTENDANCE_PATH):
        with open(ATTENDANCE_PATH, "rb") as f:
            attendance = pickle.load(f)

    if name not in [a["name"] for a in attendance]:
        attendance.append({
            "name": name,
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d")
        })

    with open(ATTENDANCE_PATH, "wb") as f:
        pickle.dump(attendance, f)

# ---------------- SESSION INIT ---------------- #
if "running" not in st.session_state:
    st.session_state.running = False

if "already_marked" not in st.session_state:
    st.session_state.already_marked = set()

# ---------------- UI ---------------- #
st.title("📸 Smart Face Attendance System")

start = st.button("▶ Start Camera")
stop = st.button("⛔ Stop Camera")
show = st.button("📋 Show Present Students")

if start:
    st.session_state.running = True

if stop:
    st.session_state.running = False

# ---------------- LOAD FACE DATA ONLY ONCE ---------------- #
faces_data = load_faces()

names = faces_data["names"]
faces = faces_data["faces"]

faces = np.array(faces) if len(faces) > 0 else np.empty((0, 10000))

# ---------------- FACE DETECTION ---------------- #
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- CAMERA ---------------- #
if st.session_state.running:

    cap = cv2.VideoCapture(0)

    FRAME_SKIP = 5   # 🔥 speed optimization
    frame_count = 0

    stframe = st.empty()

    while st.session_state.running:

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame_count += 1

        # skip frames for speed
        if frame_count % FRAME_SKIP != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (100, 100)).flatten()

            # ---- SIMPLE MATCHING ---- #
            if len(faces) != 0:
                distances = np.linalg.norm(faces - face_resized, axis=1)
                idx = np.argmin(distances)

                if distances[idx] < 3000:
                    name = names[idx]

                    # 🔥 Prevent duplicate marking
                    if name not in st.session_state.already_marked:
                        save_attendance(name)
                        st.session_state.already_marked.add(name)

                    cv2.putText(frame, name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()

# ---------------- SHOW ATTENDANCE ---------------- #
if show:
    if os.path.exists(ATTENDANCE_PATH):
        with open(ATTENDANCE_PATH, "rb") as f:
            data = pickle.load(f)
        st.dataframe(data)
    else:
        st.warning("No attendance data found")
