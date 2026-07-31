"""
==================================================================
 CONFIGURATION FILE
 This is the ONLY file you should normally need to edit.
 Change the values below to match your setup, then run the
 scripts in order (1 -> 2 -> 3 -> 4).
==================================================================
"""

import os

# ------------------------------------------------------------------
# FOLDER PATHS
# By default everything is stored inside this project folder using
# relative paths, so it works no matter where you place the folder.
# Change these ONLY if you want the data saved somewhere else,
# e.g. DATASET_DIR = r"C:\Users\YourName\Desktop\face_dataset"
# ------------------------------------------------------------------

# Where each person's captured face images are stored
DATASET_DIR = "dataset"

# Where the trained recognition model is saved
TRAINER_DIR = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.json")

# Where attendance records are saved
ATTENDANCE_DIR = "attendance"
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "Attendance.csv")

# ------------------------------------------------------------------
# CAMERA SETTINGS
# ------------------------------------------------------------------

# Which webcam to use. 0 = default/built-in camera.
# If the wrong camera opens (e.g. on a laptop with multiple cameras),
# try changing this to 1, 2, etc.
CAMERA_INDEX = 0

# How many face photos to capture per person during registration.
# More images = generally better accuracy but takes longer. 30-50 is a good range.
NUMBER_OF_SAMPLES = 40

# ------------------------------------------------------------------
# RECOGNITION SETTINGS
# ------------------------------------------------------------------

# All face crops are resized to this size before training/recognition.
# You normally don't need to change this.
FACE_SIZE = (200, 200)

# Recognition strictness for the LBPH algorithm.
# LOWER confidence number = stricter / better match required.
#   - If a KNOWN person keeps being shown as "Unknown"  -> INCREASE this (try 80-90)
#   - If two different people get confused with each other -> DECREASE this (try 50-60)
CONFIDENCE_THRESHOLD = 70
