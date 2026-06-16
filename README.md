# Face Recognition Attendance System — Flask Web App

## Project Structure
```
attendance_app/
├── app.py                  # Flask backend (all routes & logic)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Full web UI
├── haarcascade_frontalface_default.xml   ← YOU MUST ADD THIS
├── StudentDetails/         # Auto-created
├── TrainingImage/          # Auto-created
├── TrainingImageLabel/     # Auto-created
└── Attendance/             # Auto-created
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Haar Cascade file
Download `haarcascade_frontalface_default.xml` from:
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

Place it in the **same folder as `app.py`**.

### 3. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Features (all original features preserved)

| Original (Tkinter)              | Web (Flask)                        |
|---------------------------------|------------------------------------|
| Take Images (camera capture)    | ✅ `/api/take_images`              |
| Save Profile (train model)      | ✅ `/api/train` with password auth |
| Take Attendance (face recog.)   | ✅ `/api/track_attendance`         |
| Attendance table view           | ✅ Live table in browser           |
| Send email with CSV             | ✅ `/api/send_email`               |
| Change password                 | ✅ Password tab in UI              |
| Delete registration CSV         | ✅ Danger zone buttons             |
| Delete attendance CSV           | ✅ Danger zone buttons             |
| Delete registered images        | ✅ Danger zone buttons             |
| Live clock & date               | ✅ JS clock in header              |
| Registration count              | ✅ Auto-refreshed badge            |

## Email Setup (Gmail)
1. Enable 2-Factor Authentication on your Gmail account
2. Generate an **App Password**: Google Account → Security → App Passwords
3. Enter your Gmail and App Password in the "Send Attendance" section

## Notes
- Camera windows (OpenCV) still open as desktop windows — press **Q** to close them
- All data files are created automatically in the project folder
- Password is stored in `TrainingImageLabel/psd.txt`
