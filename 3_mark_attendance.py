"""
==================================================================
 STEP 3: MARK ATTENDANCE
 Recognizes faces against the trained model and logs attendance.
 Two modes are offered:
   1) Live webcam  - continuously recognizes faces and marks attendance
   2) Uploaded image - checks a single photo you provide

 HOW TO RUN (in VS Code terminal):
     python 3_mark_attendance.py
==================================================================
"""

import cv2
import os
import json
import csv
from datetime import datetime
import config


def load_recognizer_and_labels():
    if not os.path.exists(config.TRAINER_FILE):
        raise FileNotFoundError("trainer.yml not found. Run 2_train_model.py first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(config.TRAINER_FILE)

    with open(config.LABELS_FILE, "r") as f:
        raw = json.load(f)
    id_to_name = {int(k): v for k, v in raw.items()}
    return recognizer, id_to_name


def get_already_marked_today():
    """Returns the set of names already marked present today, so we don't duplicate entries."""
    marked = set()
    if os.path.isfile(config.ATTENDANCE_FILE):
        today = datetime.now().strftime("%Y-%m-%d")
        with open(config.ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2 and row[1] == today:
                    marked.add(row[0])
    return marked


def mark_attendance(name, marked_today):
    if name in marked_today:
        return False

    os.makedirs(config.ATTENDANCE_DIR, exist_ok=True)
    file_exists = os.path.isfile(config.ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    with open(config.ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, today, now_time])

    marked_today.add(name)
    print(f"Attendance marked: {name} at {now_time}")
    return True


def recognize_and_annotate(frame, gray, face_detector, recognizer, id_to_name, marked_today):
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)
        label_id, confidence = recognizer.predict(face_roi)

        # In LBPH, a LOWER confidence value means a BETTER match
        if confidence < config.CONFIDENCE_THRESHOLD:
            name = id_to_name.get(label_id, "Unknown")
            mark_attendance(name, marked_today)
            match_score = max(0, round(100 - confidence))
            display_text = f"{name} ({match_score}%)"
            color = (0, 255, 0)
        else:
            display_text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, display_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def run_webcam_mode():
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer, id_to_name = load_recognizer_and_labels()
    marked_today = get_already_marked_today()

    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        print("Could not access the camera. Check CAMERA_INDEX in config.py.")
        return

    print("Live attendance started. Press 'q' to quit.\n")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = recognize_and_annotate(frame, gray, face_detector, recognizer, id_to_name, marked_today)
        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def run_image_mode(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer, id_to_name = load_recognizer_and_labels()
    marked_today = get_already_marked_today()

    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not read that image file. Check the path and file format.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = recognize_and_annotate(frame, gray, face_detector, recognizer, id_to_name, marked_today)

    os.makedirs(config.ATTENDANCE_DIR, exist_ok=True)
    output_path = os.path.join(config.ATTENDANCE_DIR, "last_result.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Annotated result image saved to {output_path}")

    cv2.imshow("Attendance Result - press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Choose a mode:")
    print("  1 - Live webcam attendance")
    print("  2 - Recognize from an uploaded image")
    choice = input("Enter 1(To open webcam) or 2(To upload image): ").strip()

    if choice == "1":
        run_webcam_mode()
    elif choice == "2":
        path = input("Enter the full path to the image file: ").strip()
        run_image_mode(path)
    else:
        print("Invalid choice.")
