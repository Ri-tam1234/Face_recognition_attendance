import cv2
import numpy as np
from data_manager import load_data, save_data, parse_label

# ---------------- LOAD MODEL ---------------- #
face_detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

# ---------------- INPUT ---------------- #
name = input("Enter Name: ").strip().title()
user_id = input("Enter ID: ").strip()

if not user_id:
    print("❌ ID cannot be empty!")
    exit()

# ---------------- LOAD EXISTING ---------------- #
faces, names = load_data()
faces = list(faces)
names = list(names)

# ---------------- CHECK DUPLICATE ID ---------------- #
for label in names:
    _, uid = parse_label(label)
    if uid == user_id:
        print(f"❌ ID '{user_id}' already exists! Use different ID.")
        exit()

label = f"{name}_{user_id}"

# ---------------- CAPTURE ---------------- #
faces_data = []
count = 0

print("📸 Collecting faces... Look at camera")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected:
        crop = frame[y:y+h, x:x+w]
        resized = cv2.resize(crop, (50, 50))

        faces_data.append(resized.flatten())
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Samples: {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Collect Faces", frame)

    if cv2.waitKey(1) == ord('q') or count >= 100:
        break

video.release()
cv2.destroyAllWindows()

# ---------------- SAVE ---------------- #
if count < 20:
    print("❌ Not enough samples collected!")
    exit()

for f in faces_data:
    faces.append(f)
    names.append(label)

save_data(faces, names)

print(f"✅ User saved: {label}")