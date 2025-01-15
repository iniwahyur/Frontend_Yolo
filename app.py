from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, session, abort
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import winsound
import pygame
import threading
import uuid
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Konfigurasi Database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi pygame mixer
pygame.mixer.init()

# Inisialisasi database
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Model untuk log deteksi
class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    label = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    duration = db.Column(db.Float, nullable=False)
    completed = db.Column(db.Boolean, default=False)
    image_path = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return f"<DetectionLog {self.label}, {self.gender}, {self.confidence}, {self.duration}, {self.image_path}>"

# Model untuk pengguna
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default='mahasiswa')  # Default role for new users

# Load model YOLOv8
model = YOLO('best.pt')

# Warna untuk kelas yang berbeda
colors = {
    0: (0, 0, 255),  # Merah untuk "jefri"
    1: (0, 255, 0),  # Hijau untuk "manca"
    2: (255, 0, 0)   # Biru untuk "minuman"
}

class_names = {
    0: "angga",
    1: "manca",
    2: "minuman"
}

# Load detektor wajah (Haarcascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def classify_gender(cls):
    if cls == 0 or cls == 1:  # Manca dan Jefri
        return "Male"
    elif cls == 2:  # Minuman
        return "Unknown"
    return "Unknown"

def detect_behavior(frame):
    results = model(frame)
    return results

cap1 = cv2.VideoCapture(1)  # Kamera pertama
cap2 = cv2.VideoCapture(2)  # Kamera kedua

# rtsp_url = 'rtsp://admin:admin@10.3.1.210:8554/Streaming/Channels/102'
# cap = cv2.VideoCapture(rtsp_url)

last_detection_time = time.time()
detection_interval = 1
box_buffer = []

def log_detection(label, gender, confidence, duration, image_path=None):
    try:
        with app.app_context():
            log = DetectionLog.query.filter_by(label=label, completed=False).one_or_none()
            if log:
                log.duration += duration
                log.timestamp = datetime.now()
                if image_path:
                    log.image_path = image_path
                db.session.commit()
            else:
                new_log = DetectionLog(
                    label=label,
                    confidence=confidence,
                    gender=gender,
                    duration=duration,
                    timestamp=datetime.now(),
                    completed=False,
                    image_path=image_path
                )
                db.session.add(new_log)
                db.session.commit()
    except Exception as e:
        print(f"Error logging detection: {e}")

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y2_1)

    iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou

def is_box_in_buffer(new_box, buffer, iou_threshold=0.5):
    for i, (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in enumerate(buffer):
        existing_box = (x1, y1, x2, y2)
        iou = calculate_iou(new_box, existing_box)
        if iou > iou_threshold:
            return i
    return -1

box_expiry_duration = 2

def play_custom_sound():
    pygame.mixer.music.load('static/sounds/tes.mp3')  # Ganti dengan path ke file suara Anda
    pygame.mixer.music.play()  # Memutar suara

def generate_frames(camera_id):
    global last_detection_time, box_buffer
    minuman_alert = False

    # Pilih kamera berdasarkan camera_id
    cap = cap1 if camera_id == 1 else cap2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if current_time - last_detection_time >= detection_interval:
            results = detect_behavior(frame)
            last_detection_time = current_time

            detected_minuman = False

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]

                    # Hanya mendeteksi jika confidence >= 70%
                    if conf >= 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        new_box = (x1, y1, x2, y2)

                        # Tetapkan gender berdasarkan kelas
                        gender = classify_gender(cls)

                        # Tambahkan logika untuk memfilter berdasarkan wajah yang sudah ada
                        if cls == 0 or cls == 1:  # Jefri atau Manca
                            # Gunakan Haarcascade untuk mendeteksi wajah
                            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

                            # Jika wajah tidak terdeteksi, lewati
                            if len(faces) == 0:
                                continue

                        box_index = is_box_in_buffer(new_box, box_buffer)

                        label = f"{class_names[cls]} {conf:.2f}"
                        if box_index == -1:
                            box_buffer.append((x1, y1, x2, y2, label, gender, current_time, current_time))
                            log_detection(label, gender, conf, 0)

                            if cls == 2:
                                detected_minuman = True
                                unique_id = uuid.uuid4()
                                image_filename = f"images/minuman_{unique_id}.jpg"
                                static_image_filename = f"static/images/minuman_{unique_id}.jpg"

                                # Gambar bounding box pada frame sebelum menyimpan
                                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)

                                # Simpan gambar dengan bounding box
                                cv2.imwrite(image_filename, frame)
                                cv2.imwrite(static_image_filename, frame)

                                log_detection(label, gender, conf, 0, image_path=image_filename)
                        else:
                            buffer_item = list(box_buffer[box_index])
                            buffer_item[0], buffer_item[1], buffer_item[2], buffer_item[3] = x1, y1, x2, y2
                            buffer_item[7] = current_time
                            box_buffer[box_index] = tuple(buffer_item)

                        # Selalu gambar bounding box untuk semua kelas yang terdeteksi
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)

                        if cls == 2:
                            detected_minuman = True

            if detected_minuman:
                cv2.putText(frame, "Peringatan: Minuman terdeteksi!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not minuman_alert:
                    threading.Thread(target=play_custom_sound).start()
                    minuman_alert = True
            else:
                minuman_alert = False

        expired_boxes = [(x1, y1, x2, y2, label, gender, init_time, last_seen_time) for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer
                         if current_time - last_seen_time > box_expiry_duration]

        for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in expired_boxes:
            detection_duration = current_time - init_time
            log_detection(label, gender, 0, detection_duration)

        box_buffer = [(x1, y1, x2, y2, label, gender, init_time, last_seen_time) for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer
                      if current_time - last_seen_time <= box_expiry_duration]

        for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer:
            time_elapsed = current_time - init_time
            timer_label = f"{label}, {gender} | {int(time_elapsed)}s"

            class_label = label.split()[0]
            class_index = [k for k, v in class_names.items() if v == class_label]

            if class_index:
                color = colors[class_index[0]]
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, timer_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))  # Arahkan ke login jika tidak ada session

    user = User.query.filter_by(username=session['username']).first()  # Ambil pengguna dari session
    current_time = datetime.now()  # Dapatkan waktu saat ini

    total_logs = DetectionLog.query.count() 

    return render_template('dashboard.html', user=user, current_time=current_time, total_logs=total_logs)

@app.route('/laporan')
def laporan():
    logs = DetectionLog.query.filter(~DetectionLog.label.like('%minuman%')).all()
    return render_template('laporan.html', logs=logs) 

@app.route('/ulaporan')
def ulaporann():
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated

    user = User.query.filter_by(username=session['username']).first()  # Ambil data pengguna yang login

    if not user:
        return redirect(url_for('login'))  # Jika pengguna tidak ditemukan, arahkan kembali ke login

    try:
        # Ambil logs yang sesuai dengan pengguna yang login, kecuali label 'minuman'
        logs = DetectionLog.query.filter(
            ~DetectionLog.label.like('%minuman%'),
            DetectionLog.label.like(f"%{user.username}%")  # Filter berdasarkan username
        ).all()
    except Exception as e:
        print("Error fetching logs:", e)  # Log error jika terjadi masalah
        logs = []

    # Ambil waktu saat ini
    current_time = datetime.now()

    return render_template('ulaporan.html', user=user, logs=logs, current_time=current_time)

@app.route('/pelanggaran')
def pelanggaran():
    pelanggarans = DetectionLog.query.filter(DetectionLog.label.like('%minuman%')).all()
    return render_template('pelanggaran.html', pelanggarans=pelanggarans)

@app.route('/edit_log/<int:id>', methods=['GET', 'POST'])
def edit_log(id):
    log = DetectionLog.query.get_or_404(id)
    if request.method == 'POST':
        log.label = request.form['label']
        log.confidence = float(request.form['confidence'])
        log.gender = request.form['gender']
        log.duration = float(request.form['duration'])
        db.session.commit()
        return redirect(url_for('laporan'))
    return render_template('edit_log.html', log=log)

@app.route('/delete_log/<int:id>', methods=['POST'])
def delete_log(id):
    log = DetectionLog.query.get_or_404(id)
    # Hapus gambar dari folder 'images'
    if log.image_path:  # Periksa apakah ada jalur gambar
        try:
            # Hapus file gambar dari disk
            if os.path.exists(log.image_path):
                os.remove(log.image_path)
        except Exception as e:
            print(f"Error deleting image file: {e}")
    
    # Hapus gambar dari folder 'static/images' jika ada
    static_image_path = os.path.join('static/images', os.path.basename(log.image_path)) if log.image_path else None
    if static_image_path and os.path.exists(static_image_path):
        try:
            os.remove(static_image_path)
        except Exception as e:
            print(f"Error deleting static image file: {e}")

    db.session.delete(log)
    db.session.commit()
    return redirect(url_for('laporan'))

@app.route('/delete_logs', methods=['POST'])
def delete_logs():
    ids_to_delete = request.json.get('ids', [])
    for id in ids_to_delete:
        log = DetectionLog.query.get_or_404(id)
        # Hapus gambar dari folder 'images'
        if log.image_path:  # Periksa apakah ada jalur gambar
            try:
                # Hapus file gambar dari disk
                if os.path.exists(log.image_path):
                    os.remove(log.image_path)
            except Exception as e:
                print(f"Error deleting image file: {e}")

        # Hapus gambar dari folder 'static/images' jika ada
        static_image_path = os.path.join('static/images', os.path.basename(log.image_path)) if log.image_path else None
        if static_image_path and os.path.exists(static_image_path):
            try:
                os.remove(static_image_path)
            except Exception as e:
                print(f"Error deleting static image file: {e}")

    DetectionLog.query.filter(DetectionLog.id.in_(ids_to_delete)).delete(synchronize_session=False)
    db.session.commit()
    return jsonify({"success": True, "message": "Logs deleted successfully."}), 200

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        pass
    return render_template('dataset.html')

# User authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Peran default untuk pengguna baru adalah 'mahasiswa'
        new_user = User(username=username, password=hashed_password, role='mahasiswa')
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))  # Redirect ke halaman login setelah pendaftaran
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['username'] = user.username  # Simpan username ke session
            session['role'] = user.role  # Simpan role ke session
            
            # Arahkan berdasarkan role
            if user.role == 'mahasiswa':
                return redirect(url_for('ulaporann'))  # Arahkan mahasiswa ke halaman ulaporann
            else:
                return redirect(url_for('dashboard'))  # Arahkan pengguna lain ke halaman dashboard
            
        else:
            return "Invalid credentials."
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    if not os.path.exists('images'):
        os.makedirs('images')

    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)

# Cleanup
cap.release()
cv2.destroyAllWindows()