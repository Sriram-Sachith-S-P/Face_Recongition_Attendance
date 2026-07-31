"""
==================================================================
 STEP 1: CAPTURE FACES
 Run this once for EACH person you want the system to recognize.
 It opens your webcam, detects the face, and saves sample images
 into the 'dataset' folder so the model can learn that person.

 HOW TO RUN (in VS Code terminal):
     python 1_capture_faces.py
==================================================================
"""

import cv2
import os
import config


def capture_faces():
    name = input("Enter the person's name (use underscores instead of spaces, e.g. Rahul_Sharma): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    person_dir = os.path.join(config.DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cam.isOpened():
        print("Could not access the camera. Check CAMERA_INDEX in config.py.")
        return

    print(f"\nLook straight at the camera. Slowly turn your head slightly left/right/up/down for variety.")
    print(f"Capturing {config.NUMBER_OF_SAMPLES} images for '{name}'. Press 'q' to stop early.\n")

    count = 0
    while count < config.NUMBER_OF_SAMPLES:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_img = cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/{config.NUMBER_OF_SAMPLES}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # only save one face per frame, in case someone else appears in the background

        cv2.imshow("Capturing Faces - press q to stop", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= config.NUMBER_OF_SAMPLES:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Saved {count} images to '{person_dir}'.")
    print("Repeat this script for every other person, then run 2_train_model.py")


if __name__ == "__main__":
    capture_faces()
