"""
==================================================================
 STEP 4 (BONUS): VIEW ATTENDANCE
 Quick helper that prints today's attendance to the terminal.

 HOW TO RUN (in VS Code terminal):
     python 4_view_attendance.py
==================================================================
"""

import csv
import os
from datetime import datetime
import config


def view_today_attendance():
    if not os.path.isfile(config.ATTENDANCE_FILE):
        print("No attendance recorded yet.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\nAttendance for {today}")
    print("-" * 32)

    found = False
    with open(config.ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[1] == today:
                print(f"{row[0]:<20} {row[2]}")
                found = True

    if not found:
        print("No one has been marked present today yet.")
    print()


if __name__ == "__main__":
    view_today_attendance()
