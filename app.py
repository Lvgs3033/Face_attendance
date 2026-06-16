from flask import Flask, render_template, request, jsonify, send_file
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)

# ─── Helpers ────────────────────────────────────────────────────────────────

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_haarcascade():
    return os.path.isfile("haarcascade_frontalface_default.xml")

def get_password():
    assure_path_exists("TrainingImageLabel")
    path = os.path.join("TrainingImageLabel", "psd.txt")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None

def save_password(pw):
    assure_path_exists("TrainingImageLabel")
    with open(os.path.join("TrainingImageLabel", "psd.txt"), "w") as f:
        f.write(pw)

def count_registrations():
    path = os.path.join("StudentDetails", "StudentDetails.csv")
    if not os.path.isfile(path):
        return 0
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    return max(0, (len(rows) // 2) - 1)

def getImagesAndLabels(path):
    """Walk all subdirectories under path to collect training images."""
    faces, ids = [], []
    for root, dirs, files in os.walk(path):
        for fname in files:
            if not fname.lower().endswith(".jpg"):
                continue
            imgPath = os.path.join(root, fname)
            try:
                img    = Image.open(imgPath).convert('L')
                img_np = np.array(img, 'uint8')
                id_    = int(os.path.split(imgPath)[-1].split(".")[1])
                faces.append(img_np)
                ids.append(id_)
            except (IndexError, ValueError):
                pass
    return faces, ids

def get_today_att_file():
    date = datetime.datetime.now().strftime('%d-%m-%Y')
    return os.path.join("Attendance", f"Attendance_{date}.csv")

def read_att_records(att_file):
    records = []
    if not os.path.isfile(att_file):
        return records
    with open(att_file, 'r') as f:
        rows = list(csv.reader(f))
    for i, row in enumerate(rows):
        if i > 0 and len(row) >= 4:
            records.append({"id": row[0], "name": row[1], "date": row[2], "time": row[3]})
    return records

def rewrite_csv(att_file, rows_to_keep, header):
    """Rewrite attendance CSV keeping only rows_to_keep (list of lists)."""
    with open(att_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_to_keep)

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    ts = time.time()
    date_str = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    registrations = count_registrations()
    return render_template("index.html", date=date_str, registrations=registrations)

@app.route("/api/status")
def status():
    return jsonify({
        "time": time.strftime('%I:%M:%S %p'),
        "registrations": count_registrations()
    })

# ─── Take Images ─────────────────────────────────────────────────────────────
# Saves captured face images into TrainingImage/{studentID}_{studentName}/
# Each student gets their own dedicated subfolder.

@app.route("/api/take_images", methods=["POST"])
def take_images():
    if not check_haarcascade():
        return jsonify({"success": False,
                        "message": "haarcascade_frontalface_default.xml not found."})

    data       = request.json
    student_id = data.get("id",   "").strip()
    name       = data.get("name", "").strip()

    if not student_id:
        return jsonify({"success": False, "message": "Please enter a student ID."})
    if not name or not (name.replace(" ", "").isalpha()):
        return jsonify({"success": False, "message": "Please enter a valid name (letters only)."})

    assure_path_exists("StudentDetails")

    # ── Create a dedicated subfolder for this student ─────────────────────
    # Folder name: {studentID}_{studentName}  (spaces in name → underscores)
    folder_name   = f"{student_id}_{name.replace(' ', '_')}"
    student_folder = os.path.join("TrainingImage", folder_name)
    assure_path_exists(student_folder)

    columns  = ['SERIAL NO.', '', 'ID', '', 'NAME']
    csv_path = os.path.join("StudentDetails", "StudentDetails.csv")

    if os.path.isfile(csv_path):
        with open(csv_path, 'r') as f:
            serial = sum(1 for _ in csv.reader(f)) // 2
    else:
        with open(csv_path, 'a+') as f:
            csv.writer(f).writerow(columns)
        serial = 1

    # ── Open camera ──────────────────────────────────────────────────────────
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        return jsonify({"success": False, "message": "Cannot open webcam. Check camera connection."})

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector   = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sample_num = 0
    MAX_IMAGES = 10

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            sample_num += 1
            # Save inside the student's own subfolder
            save_path = os.path.join(
                student_folder,
                f"{name}.{serial}.{student_id}.{sample_num}.jpg"
            )
            cv2.imwrite(save_path, gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"{name}  [{sample_num}/{MAX_IMAGES}]",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Progress bar at bottom
        progress = int((sample_num / MAX_IMAGES) * img.shape[1])
        cv2.rectangle(img, (0, img.shape[0]-8), (progress, img.shape[0]), (0, 255, 0), -1)
        cv2.putText(img, "Press Q to stop early", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Taking Images — Press Q to stop early", img)

        if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= MAX_IMAGES:
            break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if sample_num == 0:
        return jsonify({"success": False,
                        "message": "No face detected. Ensure good lighting and face the camera directly."})

    # Save student to registration CSV
    with open(csv_path, 'a+') as f:
        csv.writer(f).writerow([serial, '', student_id, '', name])

    return jsonify({"success": True,
                    "message": f"✓ {sample_num} images captured for {name} (ID: {student_id}) → folder: {folder_name}"})

# ─── Train / Save Profile ────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    data     = request.json
    password = data.get("password", "")
    stored   = get_password()

    if stored is None:
        return jsonify({"success": False,
                        "message": "No password set. Please set a password first.",
                        "no_password": True})
    if password != stored:
        return jsonify({"success": False, "message": "Wrong password. Please try again."})
    if not check_haarcascade():
        return jsonify({"success": False, "message": "haarcascade_frontalface_default.xml missing."})

    assure_path_exists("TrainingImageLabel")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = getImagesAndLabels("TrainingImage")

    if not faces:
        return jsonify({"success": False,
                        "message": "No registered faces found. Please take images first."})

    recognizer.train(faces, np.array(ids))
    recognizer.save(os.path.join("TrainingImageLabel", "Trainner.yml"))
    return jsonify({"success": True,
                    "message": f"✓ Profile saved! Total registrations: {count_registrations()}"})

# ─── Set / Change Password ───────────────────────────────────────────────────

@app.route("/api/set_password", methods=["POST"])
def set_password():
    new_pw = request.json.get("new_password", "").strip()
    if not new_pw:
        return jsonify({"success": False, "message": "Password cannot be empty."})
    save_password(new_pw)
    return jsonify({"success": True, "message": "Password registered successfully!"})

@app.route("/api/change_password", methods=["POST"])
def change_password():
    data       = request.json
    old_pw     = data.get("old_password", "")
    new_pw     = data.get("new_password", "")
    confirm_pw = data.get("confirm_password", "")
    stored     = get_password()

    if stored is None:
        return jsonify({"success": False, "message": "No password found. Please set one first."})
    if old_pw != stored:
        return jsonify({"success": False, "message": "Old password is incorrect."})
    if new_pw != confirm_pw:
        return jsonify({"success": False, "message": "New passwords do not match."})
    if not new_pw:
        return jsonify({"success": False, "message": "New password cannot be empty."})

    save_password(new_pw)
    return jsonify({"success": True, "message": "Password changed successfully!"})

# ─── Track Attendance ────────────────────────────────────────────────────────

@app.route("/api/track_attendance", methods=["POST"])
def track_attendance():
    if not check_haarcascade():
        return jsonify({"success": False, "message": "haarcascade_frontalface_default.xml missing."})

    yml_path = os.path.join("TrainingImageLabel", "Trainner.yml")
    if not os.path.isfile(yml_path):
        return jsonify({"success": False, "message": "No trained model found. Please save profile first."})

    csv_path = os.path.join("StudentDetails", "StudentDetails.csv")
    if not os.path.isfile(csv_path):
        return jsonify({"success": False, "message": "Student details missing. Please register students first."})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(yml_path)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.read_csv(csv_path)

    assure_path_exists("Attendance")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        return jsonify({"success": False, "message": "Cannot open webcam."})

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    attendance = None
    recognized_name = "Unknown"

    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 90:
                ts        = time.time()
                date      = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa        = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID_vals   = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID        = str(ID_vals[0]) if len(ID_vals) > 0 else 'Unknown'
                bb        = str(aa[0])      if len(aa)      > 0 else 'Unknown'
                recognized_name = bb
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]
                # Show only the name — no confidence percentage
                cv2.putText(im, bb, (x, y+h+25), font, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(im, "Unknown", (x, y+h+25), font, 0.9, (0, 0, 255), 2)

        cv2.putText(im, "Press Q to stop", (10, 30), font, 0.6, (200, 200, 200), 1)
        cv2.imshow('Taking Attendance — Press Q to stop', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if attendance is None:
        return jsonify({"success": False,
                        "message": "No face recognized. Ensure you are registered and face the camera clearly."})

    date      = datetime.datetime.now().strftime('%d-%m-%Y')
    att_file  = os.path.join("Attendance", f"Attendance_{date}.csv")
    col_names = ['Id', 'Name', 'Date', 'Time']
    clean_row = [attendance[0], attendance[2], attendance[4], attendance[6]]

    if os.path.isfile(att_file):
        with open(att_file, 'a+', newline='') as f:
            csv.writer(f).writerow(clean_row)
    else:
        with open(att_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(col_names)
            w.writerow(clean_row)

    records = read_att_records(att_file)
    return jsonify({"success": True,
                    "message": f"✓ Attendance marked for {recognized_name}",
                    "records": records})

# ─── Delete Single Attendance Record from CSV ────────────────────────────────

@app.route("/api/delete_attendance_record", methods=["POST"])
def delete_attendance_record():
    data     = request.json
    rec_id   = str(data.get("id",   "")).strip()
    rec_name = str(data.get("name", "")).strip()
    rec_date = str(data.get("date", "")).strip()
    rec_time = str(data.get("time", "")).strip()

    if not rec_date:
        return jsonify({"success": False, "message": "Missing date field."})

    att_file = os.path.join("Attendance", f"Attendance_{rec_date}.csv")
    if not os.path.isfile(att_file):
        return jsonify({"success": True, "message": "Record not found in CSV (already gone)."})

    with open(att_file, 'r', newline='') as f:
        rows = list(csv.reader(f))

    if not rows:
        return jsonify({"success": True, "message": "CSV is empty."})

    header    = rows[0]
    data_rows = rows[1:]
    new_rows  = []
    deleted   = False

    for row in data_rows:
        if (not deleted and len(row) >= 4 and
                str(row[0]).strip() == rec_id   and
                str(row[1]).strip() == rec_name and
                str(row[2]).strip() == rec_date and
                str(row[3]).strip() == rec_time):
            deleted = True
        else:
            new_rows.append(row)

    if not deleted:
        return jsonify({"success": False, "message": "Matching record not found in CSV."})

    rewrite_csv(att_file, new_rows, header)
    return jsonify({"success": True, "message": "Record deleted from CSV."})

# ─── Permanently delete multiple records at once ─────────────────────────────

@app.route("/api/delete_attendance_records_bulk", methods=["POST"])
def delete_attendance_records_bulk():
    records_to_delete = request.json.get("records", [])
    if not records_to_delete:
        return jsonify({"success": True, "message": "Nothing to delete."})

    from collections import defaultdict
    by_date = defaultdict(list)
    for rec in records_to_delete:
        by_date[str(rec.get("date", "")).strip()].append(rec)

    for rec_date, recs in by_date.items():
        att_file = os.path.join("Attendance", f"Attendance_{rec_date}.csv")
        if not os.path.isfile(att_file):
            continue

        with open(att_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            continue

        header    = rows[0]
        data_rows = rows[1:]

        delete_set = set()
        for rec in recs:
            delete_set.add((
                str(rec.get("id",   "")).strip(),
                str(rec.get("name", "")).strip(),
                str(rec.get("date", "")).strip(),
                str(rec.get("time", "")).strip(),
            ))

        new_rows = []
        for row in data_rows:
            if len(row) >= 4:
                key = (str(row[0]).strip(), str(row[1]).strip(),
                       str(row[2]).strip(), str(row[3]).strip())
                if key in delete_set:
                    delete_set.discard(key)
                    continue
            new_rows.append(row)

        rewrite_csv(att_file, new_rows, header)

    return jsonify({"success": True, "message": f"Deleted {len(records_to_delete)} record(s) from CSV."})

# ─── Restore Single Attendance Record to CSV ─────────────────────────────────

@app.route("/api/restore_attendance_record", methods=["POST"])
def restore_attendance_record():
    data     = request.json
    rec_id   = str(data.get("id",   "")).strip()
    rec_name = str(data.get("name", "")).strip()
    rec_date = str(data.get("date", "")).strip()
    rec_time = str(data.get("time", "")).strip()

    if not rec_date:
        return jsonify({"success": False, "message": "Missing date field."})

    assure_path_exists("Attendance")
    att_file  = os.path.join("Attendance", f"Attendance_{rec_date}.csv")
    col_names = ['Id', 'Name', 'Date', 'Time']

    if os.path.isfile(att_file):
        existing = read_att_records(att_file)
        for r in existing:
            if (str(r['id']).strip()   == rec_id   and
                str(r['name']).strip() == rec_name and
                str(r['date']).strip() == rec_date and
                str(r['time']).strip() == rec_time):
                return jsonify({"success": True, "message": "Record already exists in CSV."})
        with open(att_file, 'a+', newline='') as f:
            csv.writer(f).writerow([rec_id, rec_name, rec_date, rec_time])
    else:
        with open(att_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(col_names)
            w.writerow([rec_id, rec_name, rec_date, rec_time])

    return jsonify({"success": True, "message": "Record restored to CSV."})

# ─── Load Attendance Table ───────────────────────────────────────────────────

@app.route("/api/attendance_records")
def attendance_records():
    att_file = get_today_att_file()
    records  = read_att_records(att_file)
    return jsonify({"records": records})

# ─── Attendance Stats for Dashboard ─────────────────────────────────────────

@app.route("/api/attendance_stats")
def attendance_stats():
    attendance_dir = "Attendance"
    all_records    = []

    if os.path.exists(attendance_dir):
        for fname in os.listdir(attendance_dir):
            if fname.startswith("Attendance_") and fname.endswith(".csv"):
                fpath = os.path.join(attendance_dir, fname)
                try:
                    all_records.extend(read_att_records(fpath))
                except Exception:
                    pass

    today         = datetime.datetime.now().strftime('%d-%m-%Y')
    today_records = [r for r in all_records if r['date'] == today]

    from collections import defaultdict, Counter
    daily_counts = defaultdict(int)
    for r in all_records:
        daily_counts[r['date']] += 1

    sorted_dates = sorted(
        daily_counts.keys(),
        key=lambda d: datetime.datetime.strptime(d, '%d-%m-%Y'),
        reverse=True
    )[:7]
    daily_trend   = [{"date": d, "count": daily_counts[d]} for d in reversed(sorted_dates)]
    name_counts   = Counter(r['name'] for r in all_records)
    top_attendees = [{"name": k, "count": v} for k, v in name_counts.most_common(5)]

    hourly = defaultdict(int)
    for r in today_records:
        try:
            hour = int(r['time'].split(':')[0])
            hourly[hour] += 1
        except Exception:
            pass
    hourly_dist = [{"hour": f"{h:02d}:00", "count": hourly[h]} for h in sorted(hourly.keys())]

    return jsonify({
        "total_records":  len(all_records),
        "today_count":    len(today_records),
        "unique_today":   len(set(r['name'] for r in today_records)),
        "total_students": count_registrations(),
        "daily_trend":    daily_trend,
        "top_attendees":  top_attendees,
        "hourly_dist":    hourly_dist,
        "today_records":  today_records
    })

# ─── Send Email ──────────────────────────────────────────────────────────────

@app.route("/api/send_email", methods=["POST"])
def send_email_route():
    data      = request.json
    recipient = data.get("email",          "").strip()
    from_email= data.get("from_email",     "").strip()
    password  = data.get("email_password", "").strip()

    if not recipient or not from_email or not password:
        return jsonify({"success": False, "message": "Please fill in all email fields."})

    date     = datetime.datetime.now().strftime('%d-%m-%Y')
    filename = os.path.join("Attendance", f"Attendance_{date}.csv")
    if not os.path.isfile(filename):
        return jsonify({"success": False, "message": f"No attendance file found for {date}."})

    try:
        msg = MIMEMultipart()
        msg['From']    = from_email
        msg['To']      = recipient
        msg['Subject'] = f"Today's Attendance Report — {date} {time.strftime('%I:%M:%S %p')}"
        msg.attach(MIMEText("Please find attached the attendance report.", 'plain'))

        with open(filename, "rb") as att:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(att.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        f'attachment; filename="{os.path.basename(filename)}"')
        msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, recipient, msg.as_string())
        server.quit()
        return jsonify({"success": True, "message": "✓ Attendance report sent successfully!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to send email: {str(e)}"})

# ─── Danger Zone Deletes ─────────────────────────────────────────────────────

@app.route("/api/delete_registration_csv", methods=["POST"])
def delete_registration_csv():
    path = os.path.join("StudentDetails", "StudentDetails.csv")
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True, "message": "Registration CSV deleted successfully."})
    return jsonify({"success": False, "message": "Registration CSV not found."})

@app.route("/api/delete_attendance_csv", methods=["POST"])
def delete_attendance_csv():
    date = datetime.datetime.now().strftime('%d-%m-%Y')
    path = os.path.join("Attendance", f"Attendance_{date}.csv")
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True, "message": f"Attendance CSV for {date} deleted."})
    return jsonify({"success": False, "message": f"Attendance CSV for {date} not found."})

@app.route("/api/delete_registered_images", methods=["POST"])
def delete_registered_images():
    import shutil
    folder = "TrainingImage"
    if not os.path.exists(folder):
        return jsonify({"success": False, "message": "TrainingImage folder not found."})
    errors = []
    for entry in os.listdir(folder):
        entry_path = os.path.join(folder, entry)
        try:
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)   # remove student subfolders
            else:
                os.remove(entry_path)        # remove any loose files
        except Exception as e:
            errors.append(str(e))
    if errors:
        return jsonify({"success": False, "message": f"Some files could not be deleted: {errors}"})
    return jsonify({"success": True, "message": "✓ All registered images deleted."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)