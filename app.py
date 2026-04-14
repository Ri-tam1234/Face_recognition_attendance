import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- FILE PATH ---------------- #
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(DATASET_PATH, exist_ok=True)

# ---------------- FACE MODEL ---------------- #
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ---------------- TRAIN MODEL ---------------- #
def train_model():
    faces = []
    labels = []
    label_map = {}

    label_id = 0

    for file in os.listdir(DATASET_PATH):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            data = np.load(os.path.join(DATASET_PATH, file))

            label_map[label_id] = name

            for face in data:
                faces.append(face)
                labels.append(label_id)

            label_id += 1

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        return label_map
    return {}

label_map = train_model()

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

# ---------------- SESSION STATE ---------------- #
if "marked" not in st.session_state:
    st.session_state.marked = set()

# ---------------- FACE DETECTOR ---------------- #
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- UI ---------------- #
st.title("🔥 FAST Face Attendance System (NO LAG VERSION)")

menu = st.sidebar.selectbox("Menu", ["Live", "Add Face", "Attendance"])

# ---------------- LIVE CAMERA ---------------- #
class FaceProcessor(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)

            name = "Unknown"

            if confidence < 70:
                name = label_map.get(label, "Unknown")

                if name not in st.session_state.marked:
                    mark_attendance(name)
                    st.session_state.marked.add(name)

            # Draw box + text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{name}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

        return img

# ---------------- LIVE PAGE ---------------- #
if menu == "Live":

    webrtc_streamer(
        key="fast-face",
        video_transformer_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# ---------------- ADD FACE ---------------- #
elif menu == "Add Face":

    st.subheader("➕ Add Face (Fast Dataset Builder)")

    name = st.text_input("Enter Name + ID")

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
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            file_path = os.path.join(DATASET_PATH, f"{name}.npy")

            if os.path.exists(file_path):
                data = np.load(file_path)
                data = np.append(data, [face], axis=0)
            else:
                data = np.array([face])

            np.save(file_path, data)

            st.success(f"Face saved for {name}")
            st.info("Restart app to retrain model")

# ---------------- ATTENDANCE PAGE ---------------- #
elif menu == "Attendance":

    if os.path.exists(ATTENDANCE_PATH):
        df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(df)
    else:
        st.warning("No attendance recorded yet")
