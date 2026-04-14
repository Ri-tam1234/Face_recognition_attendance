import streamlit as st
import cv2
import numpy as np
import face_recognition
import pickle
import os
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- PATHS ---------------- #
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "faces.pkl")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "data", "attendance.csv")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# ---------------- LOAD FACES ---------------- #
def load_faces():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as f:
            return pickle.load(f)
    return {"names": [], "encodings": []}

def save_faces(data):
    with open(DATA_PATH, "wb") as f:
        pickle.dump(data, f)

faces_data = load_faces()
known_encodings = faces_data["encodings"]
known_names = faces_data["names"]

# ---------------- ATTENDANCE ---------------- #
def mark_attendance(name):
    if os.path.exists(ATTENDANCE_PATH):
        df = pd.read_csv(ATTENDANCE_PATH)
    else:
        df = pd.DataFrame(columns=["Name", "Time", "Date"])

    today = datetime.now().strftime("%Y-%m-%d")

    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        new_row = {
            "Name": name,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Date": today
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_PATH, index=False)

# ---------------- SESSION STATE ---------------- #
if "marked" not in st.session_state:
    st.session_state.marked = set()

if "new_face_encoding" not in st.session_state:
    st.session_state.new_face_encoding = None

if "new_face_name" not in st.session_state:
    st.session_state.new_face_name = ""

# ---------------- UI ---------------- #
st.title("🔥 PRO Face Attendance System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Live Attendance", "Add New Face", "View Attendance"]
)

# ---------------- LIVE CAMERA ---------------- #
class FaceProcessor(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, encoding)

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)

                if matches[best_match]:
                    name = known_names[best_match]

                    # prevent duplicates
                    if name not in st.session_state.marked:
                        mark_attendance(name)
                        st.session_state.marked.add(name)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        return img

# ---------------- PAGE 1: LIVE ---------------- #
if menu == "Live Attendance":

    st.subheader("📡 Live Face Recognition")

    webrtc_streamer(
        key="pro-face",
        video_transformer_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# ---------------- PAGE 2: ADD FACE ---------------- #
elif menu == "Add New Face":

    st.subheader("➕ Add New Face")

    name = st.text_input("Enter Name")

    webrtc_ctx = webrtc_streamer(
        key="add-face",
        media_stream_constraints={"video": True, "audio": False}
    )

    if st.button("Capture Face"):
        if webrtc_ctx.video_transformer:
            frame = webrtc_ctx.video_transformer.frame

            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)

                if len(encodings) > 0:
                    encoding = encodings[0]

                    known_encodings.append(encoding)
                    known_names.append(name)

                    save_faces({
                        "names": known_names,
                        "encodings": known_encodings
                    })

                    st.success(f"Face added for {name}")
                else:
                    st.error("No face detected")

# ---------------- PAGE 3: VIEW ATTENDANCE ---------------- #
elif menu == "View Attendance":

    st.subheader("📋 Attendance Records")

    if os.path.exists(ATTENDANCE_PATH):
        df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(df)
    else:
        st.warning("No attendance yet")
