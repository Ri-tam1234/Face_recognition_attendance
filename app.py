import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

from data_manager import parse_label

st.set_page_config(page_title="Face Attendance", layout="wide")

# ---------------- PATH ---------------- #
BASE_DIR = os.path.dirname(__file__)
face_path = os.path.join(BASE_DIR, "data", "faces_data.pkl")
name_path = os.path.join(BASE_DIR, "data", "names.pkl")

# ---------------- LOAD ---------------- #
if not os.path.exists(face_path):
    st.error("No training data found!")
    st.stop()

FACES = pickle.load(open(face_path, "rb"))
LABELS = pickle.load(open(name_path, "rb"))

if len(FACES) == 0:
    st.error("Empty dataset")
    st.stop()

FACES = np.array(FACES) / 255.0

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(FACES, LABELS)

# ---------------- UI ---------------- #
st.title("📸 Smart Face Attendance System")
st.write("Upload or capture image to mark attendance")

present = []

# ---------------- IMAGE INPUT ---------------- #
img_file = st.camera_input("Take a picture")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    # Resize for consistency
    face = np.resize(frame, (50, 50, 3)).flatten().reshape(1, -1) / 255.0

    dist, _ = knn.kneighbors(face)
    pred = knn.predict(face)[0]

    d = np.mean(dist)
    name, uid = parse_label(pred)

    if d < 2000:
        st.success(f"✅ {name} (ID: {uid}) marked present")

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
        st.error("❌ Face not recognized")

    st.image(frame)
