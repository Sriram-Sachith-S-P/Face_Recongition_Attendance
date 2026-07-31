"""
==================================================================
 STEP 2: TRAIN MODEL
 Run this AFTER you have captured face images for everyone
 (1_capture_faces.py). It trains a recognizer on everything
 inside the 'dataset' folder and saves the trained model so
 3_mark_attendance.py can use it.

 HOW TO RUN (in VS Code terminal):
     python 2_train_model.py
==================================================================
"""

import cv2
import os
import json
import numpy as np
import config


def train_model():
    if not os.path.isdir(config.DATASET_DIR):
        print(f"Dataset folder '{config.DATASET_DIR}' not found. Run 1_capture_faces.py first.")
        return

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    people = sorted([p for p in os.listdir(config.DATASET_DIR)
                      if os.path.isdir(os.path.join(config.DATASET_DIR, p))])

    if not people:
        print(f"No person folders found inside '{config.DATASET_DIR}'. Run 1_capture_faces.py first.")
        return

    for person_name in people:
        person_path = os.path.join(config.DATASET_DIR, person_name)

        if person_name not in label_ids:
            label_ids[person_name] = current_id
            current_id += 1
        person_id = label_ids[person_name]

        image_count = 0
        for image_name in os.listdir(person_path):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(person_path, image_name)
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue

            detected = face_detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
            if len(detected) > 0:
                (x, y, w, h) = detected[0]
                face_crop = gray_img[y:y + h, x:x + w]
            else:
                # Image is likely already a tightly-cropped face (e.g. from 1_capture_faces.py)
                face_crop = gray_img

            face_crop = cv2.resize(face_crop, config.FACE_SIZE)
            faces.append(face_crop)
            labels.append(person_id)
            image_count += 1

        print(f"  {person_name}: {image_count} images processed")

    if not faces:
        print("No usable face images found in the dataset folder.")
        return

    recognizer.train(faces, np.array(labels))

    os.makedirs(config.TRAINER_DIR, exist_ok=True)
    recognizer.save(config.TRAINER_FILE)

    id_to_name = {v: k for k, v in label_ids.items()}
    with open(config.LABELS_FILE, "w") as f:
        json.dump(id_to_name, f, indent=2)

    print(f"\nTraining complete: {len(faces)} images across {len(label_ids)} people.")
    print(f"Model saved to '{config.TRAINER_FILE}'")
    print("You can now run 3_mark_attendance.py")


if __name__ == "__main__":
    print("Training model on dataset...\n")
    train_model()
