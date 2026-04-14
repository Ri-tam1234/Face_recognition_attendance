import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- PATHS ---------------- #
BASE_DIR = os.path.dirname(__file__)
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance.csv")

# ---------------- OPENCV FACE MODEL ---------------- #
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- SIMPLE MEMORY DATABASE ---------------- #
if "known_faces" not in st.session_state:
    st.session_state.known_faces = []   # (name, face_vector)

if "marked" not in st.session_state:
    st.session_state.marked = set()

# ---------------- ATTENDANCE ---------------- #
def mark_attendance(name):
    if os.path.exists(ATTENDANCE_PATH):
        df = pd.read_csv(ATTENDANCE_PATH)
    else:
        df = pd.DataFrame(columns=["Name", "Time", "Date"])

    today = datetime.now().strftime("%Y-%m-%d")

    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        df = pd.concat([df, pd.DataFrame([{
            "Name": name,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Date": today
        }])], ignore_index=True)

        df.to_csv(ATTENDANCE_PATH, index=False)

# ---------------- SIMPLE FACE MATCH ---------------- #
def match_face(face_img, known_faces):
    face = cv2.resize(face_img, (50, 50)).flatten()

    best_match = None
    min_dist = float("inf")

    for name, vec in known_faces:
        dist = np.linalg.norm(vec - face)
        if dist < min_dist:
            min_dist = dist
            best_match = name

    if min_dist < 3000:
        return best_match
    return "Unknown"

# ---------------- STREAM PROCESSOR ---------------- #
class FaceProcessor(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]

            name = match_face(face_roi, st.session_state.known_faces)

            if name != "Unknown" and name not in st.session_state.marked:
                mark_attendance(name)
                st.session_state.marked.add(name)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        return img

# ---------------- UI ---------------- #
st.title("⚡ FAST Face Attendance System (Stable Version)")

menu = st.sidebar.selectbox("Menu", ["Live", "Add Face", "Attendance"])

# ---------------- LIVE ---------------- #
if menu == "Live":
    webrtc_streamer(
        key="fast",
        video_transformer_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# ---------------- ADD FACE ---------------- #
elif menu == "Add Face":

    st.subheader("Add Face (Take Snapshot)")

    name = st.text_input("Enter Name")

    img_file = st.camera_input("Capture Face")

    if img_file is not None and name:

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.error("No face detected")
        else:
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            face_vec = cv2.resize(face_roi, (50, 50)).flatten()

            st.session_state.known_faces.append((name, face_vec))

            st.success(f"Face added for {name}")

# ---------------- ATTENDANCE ---------------- #
elif menu == "Attendance":

    if os.path.exists(ATTENDANCE_PATH):
        df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(df)
    else:
        st.warning("No attendance yet")
