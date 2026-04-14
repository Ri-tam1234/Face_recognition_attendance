import os
import pickle
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)

FACE_PATH = os.path.join(BASE_DIR, "data", "faces_data.pkl")
NAME_PATH = os.path.join(BASE_DIR, "data", "names.pkl")


# ---------------- BASIC UTILS ---------------- #
def normalize(text):
    return str(text).strip().lower().replace(" ", "")


def parse_label(label):
    label = str(label).strip().replace("\n", "").replace("\r", "")

    if "_" in label:
        name, uid = label.split("_", 1)
    else:
        name, uid = label, "N/A"

    return name.strip(), uid.strip()


# ---------------- LOAD / SAVE ---------------- #
def load_data():
    if not os.path.exists(FACE_PATH) or not os.path.exists(NAME_PATH):
        return [], []

    with open(FACE_PATH, "rb") as f:
        faces = pickle.load(f)

    with open(NAME_PATH, "rb") as f:
        names = pickle.load(f)

    return list(faces), list(names)


def save_data(faces, names):
    with open(FACE_PATH, "wb") as f:
        pickle.dump(list(faces), f)

    with open(NAME_PATH, "wb") as f:
        pickle.dump(list(names), f)


# ---------------- SHOW USERS ---------------- #
def show_users():
    _, names = load_data()

    if not names:
        print("❌ No users found!")
        return

    users = defaultdict(set)

    for label in names:
        name, uid = parse_label(label)
        users[name].add(uid)

    print("\n📋 Registered Users:\n")

    for name, ids in users.items():
        for uid in ids:
            print(f"{name} (ID: {uid})")

    print(f"\nTotal Users: {len(users)}")


# ---------------- DELETE USER ---------------- #
def delete_user(target):
    faces, names = load_data()

    if not names:
        print("❌ No data found!")
        return

    target = normalize(target)

    new_faces = []
    new_names = []
    removed = 0

    for face, label in zip(faces, names):
        name, uid = parse_label(label)

        if (
            target == normalize(name) or
            target == normalize(uid) or
            target == normalize(label)
        ):
            removed += 1
        else:
            new_faces.append(face)
            new_names.append(label)

    if removed == 0:
        print("❌ User not found!")
        return

    save_data(new_faces, new_names)

    if len(new_names) == 0:
        print("⚠️ All users deleted!")
    else:
        print(f"✅ Deleted {removed} samples")


# ---------------- REMOVE CORRUPTED DATA ---------------- #
def remove_corrupted():
    faces, names = load_data()

    new_faces = []
    new_names = []
    removed = 0

    for face, label in zip(faces, names):
        label_str = str(label).lower()

        # detect garbage entries
        if (
            "python" in label_str or
            ".py" in label_str or
            "\\" in label_str or
            "/" in label_str
        ):
            removed += 1
            continue

        new_faces.append(face)
        new_names.append(label)

    save_data(new_faces, new_names)

    print(f"🧹 Removed {removed} corrupted entries")


# ---------------- CLEAN FORMAT ---------------- #
def clean_data():
    faces, names = load_data()

    cleaned = []

    for label in names:
        name, uid = parse_label(label)
        name = name.title().strip()
        uid = uid.strip()

        if uid == "N/A":
            cleaned.append(name)
        else:
            cleaned.append(f"{name}_{uid}")

    save_data(faces, cleaned)

    print("✅ Data cleaned!")


# ---------------- MENU ---------------- #
def main():
    while True:
        print("\n====== FACE DATA MANAGER ======")
        print("1. Show Users")
        print("2. Delete User")
        print("3. Clean Data")
        print("4. Remove Corrupted Data")
        print("5. Exit")

        ch = input("Choice: ").strip()

        if ch == "1":
            show_users()
        elif ch == "2":
            delete_user(input("Enter Name or ID: "))
        elif ch == "3":
            clean_data()
        elif ch == "4":
            remove_corrupted()
        elif ch == "5":
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()