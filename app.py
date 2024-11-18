from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import random
import winsound  # For sound notifications
import threading  # For threading

app = Flask(__name__)

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Model for detection logs
class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    label = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    duration = db.Column(db.Float, nullable=False)  # Duration in seconds
    completed = db.Column(db.Boolean, default=False)  # Mark if detection is completed

    def __repr__(self):
        return f"<DetectionLog {self.label}, {self.gender}, {self.confidence}, {self.duration}>"

# Load YOLOv8 model
model = YOLO('best.pt')  # Ensure you have 'best.pt' downloaded

# Colors for different classes
colors = {
    0: (0, 0, 255),  # Red for "jefri"
    1: (0, 255, 0),  # Green for "manca"
    2: (255, 0, 0)   # Blue for "minuman"
}

class_names = {
    0: "jefri",
    1: "manca",
    2: "minuman"
}

# Load pre-trained face detector (Haarcascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Example gender classifier function (dummy, returns Male/Female based on random chance)
def classify_gender(face):
    return "Male" if random.random() > 0.5 else "Female"

# Function to detect behavior (people only)
def detect_behavior(frame):
    results = model(frame)
    return results

# Access the camera
cap = cv2.VideoCapture(0)  # Change to the correct index for your camera

# Time control variables
last_detection_time = time.time()  # Initialize the time of the last detection
detection_interval = 1  # Interval in seconds (1 second)

# Buffer to store detected boxes and their timestamps
box_buffer = []  # List to store (x1, y1, x2, y2, label, gender, init_time, last_seen_time)

# Function to log detection
def log_detection(label, gender, confidence, duration):
    try:
        with app.app_context():
            log = DetectionLog.query.filter_by(label=label, completed=False).one_or_none()
            if log:
                log.duration += duration  # Increment duration
                log.timestamp = datetime.now()  # Update timestamp
                db.session.commit()
            else:
                new_log = DetectionLog(
                    label=label,
                    confidence=confidence,
                    gender=gender,
                    duration=duration,
                    timestamp=datetime.now(),
                    completed=False
                )
                db.session.add(new_log)
                db.session.commit()
    except Exception as e:
        print(f"Error logging detection: {e}")

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Calculate the area of intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the areas of the input boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate IoU
    iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou

# Function to check if a box is already in the buffer using IoU
def is_box_in_buffer(new_box, buffer, iou_threshold=0.5):
    for i, (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in enumerate(buffer):
        existing_box = (x1, y1, x2, y2)
        iou = calculate_iou(new_box, existing_box)
        if iou > iou_threshold:
            return i  # Return index of the matched box in the buffer
    return -1  # If not found, return -1

# Duration to keep boxes if they are not seen (in seconds)
box_expiry_duration = 2

# Function to play strobing sound for "minuman" in a separate thread
def play_strobe_sound():
    for _ in range(3):  # Repeat sound 3 times for a strobing effect
        winsound.Beep(1000, 200)  # 1000 Hz beep for 200 ms
        time.sleep(0.5)  # Delay between beeps

# Generate frames for streaming
def generate_frames():
    global last_detection_time, box_buffer
    minuman_alert = False  # Flag to control sound for "minuman"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get current time
        current_time = time.time()

        # Perform detection only if the detection interval has passed
        if current_time - last_detection_time >= detection_interval:
            results = detect_behavior(frame)
            last_detection_time = current_time  # Update the time of the last detection

            detected_minuman = False  # Flag for detecting "minuman"

            # Add new detections to the buffer with current timestamp
            for result in results:
                boxes = result.boxes  # Get the detected boxes
                for box in boxes:
                    cls = int(box.cls[0])  # Class index
                    if cls in class_names:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates of the bounding box
                        conf = box.conf[0]  # Confidence score
                        new_box = (x1, y1, x2, y2)

                        # Detect face inside the bounding box
                        face_region = frame[y1:y2, x1:x2]
                        gender = "Unknown"
                        if face_region.shape[0] > 0 and face_region.shape[1] > 0:
                            faces = face_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5)
                            if len(faces) > 0:
                                for (fx, fy, fw, fh) in faces:
                                    face = face_region[fy:fy + fh, fx:fx + fw]
                                    gender = classify_gender(face)

                        # Check if this box is already in the buffer using IoU
                        box_index = is_box_in_buffer(new_box, box_buffer)

                        if box_index == -1:
                            # If the box is new, add to buffer with initial timestamp
                            label = f"{class_names[cls]} {conf:.2f}"  # Use class name for labeling
                            box_buffer.append((x1, y1, x2, y2, label, gender, current_time, current_time))

                            # Log the detection
                            log_detection(label, gender, conf, 0)  # Start duration at 0
                        else:
                            # If the box is already in the buffer, update the last seen time
                            buffer_item = list(box_buffer[box_index])
                            buffer_item[0], buffer_item[1], buffer_item[2], buffer_item[3] = x1, y1, x2, y2  # Update the box coordinates
                            buffer_item[7] = current_time  # Update last seen time
                            box_buffer[box_index] = tuple(buffer_item)

                        # Check if "minuman" is detected
                        if cls == 2:  # "minuman"
                            detected_minuman = True

            # Show persistent warning and play strobe sound if "minuman" is detected
            if detected_minuman:
                cv2.putText(frame, "Peringatan: Minuman terdeteksi!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not minuman_alert:
                    threading.Thread(target=play_strobe_sound).start()  # Play strobe sound for "minuman" in a new thread
                    minuman_alert = True
            else:
                minuman_alert = False  # Reset flag if "minuman" is not detected

        # Remove boxes that have not been seen for a certain amount of time
        expired_boxes = [(x1, y1, x2, y2, label, gender, init_time, last_seen_time) for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer
                         if current_time - last_seen_time > box_expiry_duration]

        # Write the log for boxes that are no longer detected
        for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in expired_boxes:
            detection_duration = current_time - init_time
            log_detection(label, gender, 0, detection_duration)  # Log completion

        # Update buffer by removing expired boxes
        box_buffer = [(x1, y1, x2, y2, label, gender, init_time, last_seen_time) for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer
                      if current_time - last_seen_time <= box_expiry_duration]

        # Draw all boxes on the current frame
        for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer:
            time_elapsed = current_time - init_time
            timer_label = f"{label}, {gender} | {int(time_elapsed)}s"

            # Extract the class index from the label
            class_label = label.split()[0]  # Get the class name (e.g., 'minuman')
            class_index = [k for k, v in class_names.items() if v == class_label]  # Find the index
            
            if class_index:
                color = colors[class_index[0]]  # Use the color associated with the class index
            else:
                color = (255, 255, 255)  # Default to white if class not found

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw the bounding box
            cv2.putText(frame, timer_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Draw the label

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def dashboard():
    total_logs = DetectionLog.query.count()  # Count the total number of logs
    return render_template('dashboard.html', total_logs=total_logs)  # Pass count to the template

@app.route('/laporan')
def laporan():
    with app.app_context():
        logs = DetectionLog.query.all()
    return render_template('laporan.html', logs=logs)

@app.route('/edit_log/<int:id>', methods=['GET', 'POST'])
def edit_log(id):
    with app.app_context():
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
    with app.app_context():
        log = DetectionLog.query.get_or_404(id)
        db.session.delete(log)
        db.session.commit()
    return redirect(url_for('laporan'))

@app.route('/delete_logs', methods=['POST'])
def delete_logs():
    ids_to_delete = request.json.get('ids', [])
    with app.app_context():
        # Menghapus semua log berdasarkan ID yang diterima
        DetectionLog.query.filter(DetectionLog.id.in_(ids_to_delete)).delete(synchronize_session=False)
        db.session.commit()
    return jsonify({"success": True, "message": "Logs deleted successfully."}), 200

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    # Logic to handle dataset addition or viewing
    if request.method == 'POST':
        # Handle dataset submission logic here
        pass
    return render_template('dataset.html')  # Return the dataset template

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(host='0.0.0.0', port=5000, debug=True)

# Cleanup
cap.release()
cv2.destroyAllWindows()