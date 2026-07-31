# Face Recognition Attendance System

A Python project that marks attendance by recognizing faces — either from a
**live webcam feed** or from an **uploaded photo** — and compares them against
a stored set of known people, logging the result with name, date, and time.

Built with OpenCV (Haar Cascade for face detection + LBPH for recognition),
so there's no complicated dlib/CMake installation required.

---

## How it works

1. **Register people** — capture face photos for each person (webcam) and store them.
2. **Train** — build a recognition model from those stored photos.
3. **Recognize** — compare a live camera feed or an uploaded photo against the trained model.
4. **Log** — when a known face is matched, write their name + timestamp to an attendance CSV (once per person per day).

## Folder structure

```
face_recognition_attendance/
├── config.py              ← ALL settings you might want to change live here
├── 1_capture_faces.py     ← Step 1: register a person via webcam
├── 2_train_model.py       ← Step 2: train the recognizer on everyone registered
├── 3_mark_attendance.py   ← Step 3: recognize + mark attendance (webcam or uploaded photo)
├── 4_view_attendance.py   ← Bonus: print today's attendance
├── requirements.txt
├── .gitignore
├── dataset/                (created automatically) captured face photos, per person
├── trainer/                (created automatically) the trained model
└── attendance/             (created automatically) Attendance.csv + last recognized photo
```

---

## Setup (do this once)

1. Install **Python 3.8+** and open this folder in VS Code.
2. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

---

## Usage — run these in order

### Step 1 — Register faces
```
python 1_capture_faces.py
```
Enter a name when prompted (e.g. `Rahul_Sharma`). It opens your webcam and
automatically saves ~40 cropped face photos into `dataset/Rahul_Sharma/`.
**Run this once per person** you want the system to recognize (yourself,
classmates, colleagues — anyone who agrees to be registered).

### Step 2 — Train the model
```
python 2_train_model.py
```
Reads everything in `dataset/` and produces `trainer/trainer.yml`. Re-run this
any time you add or remove someone from `dataset/`.

### Step 3 — Mark attendance
```
python 3_mark_attendance.py
```
Choose:
- **1** = live webcam — recognizes continuously, press `q` to stop
- **2** = uploaded image — enter a file path (e.g. `C:\Users\You\Pictures\photo.jpg`) and it checks that one photo

Recognized people get a green box + their name and are logged to
`attendance/Attendance.csv`. Unrecognized faces are boxed in red as `Unknown`
and are **not** logged.

### Step 4 (bonus) — View today's attendance
```
python 4_view_attendance.py
```

---

## About the "5 sample face images"

I didn't bundle sample photos of real people in this project, and did that
deliberately rather than as an oversight: a face-recognition attendance system
only becomes actually useful when it's trained on the specific people you want
it to recognize — random stock photos wouldn't be able to recognize *you* or
your classmates anyway, and downloading/reusing photos of real, named people
without their knowledge isn't something I'll do, even as placeholder data.

The good news: `1_capture_faces.py` makes building your own dataset genuinely
take under a minute per person. To get a working demo:
1. Run `1_capture_faces.py` on yourself.
2. Run it again for 2-4 friends/classmates/family members **who agree to it**.
3. You now have a real, working, multi-person dataset — far more convincing
   for a demo or interview than generic stock photos would have been.

If you'd rather not use a webcam right now, you can also just drop a few
existing photos (clear, front-facing, one person per photo) into
`dataset/<PersonName>/` folders manually — `2_train_model.py` will detect the
face in each photo automatically.

---

## What you'll most likely need to modify

Everything lives in **`config.py`** — you shouldn't need to touch the other files:

| Setting | What it does | When to change it |
|---|---|---|
| `DATASET_DIR`, `TRAINER_DIR`, `ATTENDANCE_DIR` | Where files are saved | Only if you want data saved outside this folder, e.g. `r"D:\attendance_data"` |
| `CAMERA_INDEX` | Which webcam to use | If your webcam doesn't open, try `1` or `2` instead of `0` |
| `NUMBER_OF_SAMPLES` | Photos captured per person | Increase for better accuracy (e.g. `60`); decrease for faster registration |
| `CONFIDENCE_THRESHOLD` | How strict matching is | A known person shows as "Unknown" → **increase** it (try `85`). Two people get confused → **decrease** it (try `55`) |
| `FACE_SIZE` | Internal resize dimensions | Rarely needs changing |

---

## Ideas to extend it further (good for standing out)

These aren't built in, but are natural next steps if you want to go further
for a portfolio or resume project:

- **Web/GUI interface** — wrap it in Streamlit or Flask so it's a clickable app instead of a terminal script.
- **SQLite database** instead of CSV, so you can query attendance history properly.
- **Email/SMS alert** on each check-in using `smtplib` or a service like Twilio.
- **Excel report export** with `openpyxl`/`pandas` — daily/monthly summaries with charts.
- **Liveness detection** (e.g. blink detection) — worth mentioning in interviews, since as-is, this project (like most tutorial-level face recognition systems) could in principle be fooled by a photo held up to the camera. Addressing that is a great "what would you improve" talking point.
- **Higher-accuracy recognition** — swap LBPH for a deep-learning embedding model (e.g. the `face_recognition` library or ArcFace) if you need better accuracy across varied lighting/angles.

---

## Troubleshooting

**`AttributeError: module 'cv2' has no attribute 'face'`**
You have `opencv-python` installed instead of (or alongside) `opencv-contrib-python`. Fix:
```
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python
```

**Webcam doesn't open / black window**
Try changing `CAMERA_INDEX` in `config.py` to `1` or `2`. Also make sure no other app (Zoom, Teams) is using the camera.

**A known person is recognized as "Unknown"**
Increase `CONFIDENCE_THRESHOLD` in `config.py`, or capture more/better-lit samples for that person and retrain.

**Two people keep getting mixed up**
Decrease `CONFIDENCE_THRESHOLD`, and make sure each person's photos are well-lit and clearly show their face from slightly different angles.

**Running inside WSL, SSH, or a remote container**
`cv2.imshow` needs a real display. This project is meant to run on a normal local machine with VS Code open directly (not a headless remote session).

---

## Privacy note

`dataset/` and `attendance/` contain real people's photos and attendance
records. The included `.gitignore` keeps both out of version control — worth
keeping that way if you push this project to a public GitHub repo for your
portfolio.
