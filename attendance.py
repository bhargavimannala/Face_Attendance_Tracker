import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# ---------- PATH ----------
path = "images"
images = []
classNames = []

files = os.listdir(path)
print("Files found in images folder:", files)

# ---------- LOAD & FORCE-CONVERT IMAGES ----------
for file in files:
    file_path = os.path.join(path, file)

    # Read image using OpenCV
    img = cv2.imread(file_path)

    if img is None:
        print(f"❌ Cannot read image file: {file}")
        continue

    # FORCE conversion to 8-bit RGB
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        print(
            f"✅ Loaded {file} | shape={img.shape} | dtype={img.dtype}"
        )

        images.append(img)
        classNames.append(os.path.splitext(file)[0])

    except Exception as e:
        print(f"❌ Conversion failed for {file}: {e}")

# ---------- ENCODE ----------
def findEncodings(images):
    encodeList = []
    for img in images:
        try:
            enc = face_recognition.face_encodings(img)
            if len(enc) == 0:
                print("❌ No face detected in image")
            else:
                encodeList.append(enc[0])
        except Exception as e:
            print("❌ Encoding error:", e)
    return encodeList


encodeListKnown = findEncodings(images)

if len(encodeListKnown) == 0:
    print("❌ No valid face encodings. STOPPING.")
    exit()

print("✅ Encoding Complete")

# ---------- ATTENDANCE ----------
def markAttendance(name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Name,Time")

    with open("attendance.csv", "r+") as f:
        lines = f.readlines()
        names = [line.split(",")[0] for line in lines]

        if name not in names:
            time = datetime.now().strftime("%H:%M:%S")
            f.write(f"\n{name},{time}")

# ---------- WEBCAM ----------
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Webcam not working")
        break

    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb_small)
    encodes = face_recognition.face_encodings(rgb_small, faces)

    for encode, loc in zip(encodes, faces):
        matches = face_recognition.compare_faces(encodeListKnown, encode)
        dist = face_recognition.face_distance(encodeListKnown, encode)
        idx = np.argmin(dist)

        if matches[idx]:
            name = classNames[idx].upper()
            y1, x2, y2, x1 = [v * 4 for v in loc]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, name, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            markAttendance(name)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) == 13:  # ENTER key
        break

cap.release()
cv2.destroyAllWindows()