import cv2
import time
from ultralytics import YOLO
import random
import winsound  # For Windows sound notifications
import threading  # For threading

# Load YOLOv8 model
model = YOLO('best.pt')  # Make sure you have the model file, 'best.pt', downloaded

# Define colors for each class
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
rtsp_url = 'rtsp://admin:admin@10.3.1.165:8554/Streaming/Channels/102'
cap = cv2.VideoCapture(rtsp_url)  # Change to the correct index for your camera

# Open log file to record detections
log_file = open('log.txt', 'a')  # Open file in append mode

# Time control variables
last_detection_time = time.time()
detection_interval = 1  # Interval in seconds

# Buffer to store detected boxes and their timestamps
box_buffer = []
minuman_alert = False  # Flag to control sound for "minuman"

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Function to check if a box is already in the buffer using IoU
def is_box_in_buffer(new_box, buffer, iou_threshold=0.5):
    for i, (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in enumerate(buffer):
        existing_box = (x1, y1, x2, y2)
        iou = calculate_iou(new_box, existing_box)
        if iou > iou_threshold:
            return i
    return -1

# Duration to keep boxes if they are not seen
box_expiry_duration = 2

# Function to play strobing sound for "minuman" in a separate thread
def play_strobe_sound():
    for _ in range(3):  # Repeat sound 3 times for a strobing effect
        winsound.Beep(1000, 200)  # 1000 Hz beep for 200 ms
        time.sleep(0.5)  # Delay between beeps

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
                if cls in class_names:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    new_box = (x1, y1, x2, y2)

                    face_region = frame[y1:y2, x1:x2]
                    
                    # Detect face inside the bounding box
                    gender = "Unknown"
                    if face_region.shape[0] > 0 and face_region.shape[1] > 0:
                        faces = face_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5)
                        if len(faces) > 0:
                            for (fx, fy, fw, fh) in faces:
                                face = face_region[fy:fy + fh, fx:fx + fw]
                                if face.size > 0:
                                    gender = classify_gender(face)

                    box_index = is_box_in_buffer(new_box, box_buffer)

                    if box_index == -1:
                        label = f"{class_names[cls]} {conf:.2f}"
                        box_buffer.append((x1, y1, x2, y2, label, gender, current_time, current_time))
                        log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Detected: {label}, Gender: {gender} (Confidence: {conf:.2f})\n"
                        log_file.write(log_entry)
                    else:
                        buffer_item = list(box_buffer[box_index])
                        buffer_item[0], buffer_item[1], buffer_item[2], buffer_item[3] = x1, y1, x2, y2
                        buffer_item[7] = current_time
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

        expired_boxes = [(x1, y1, x2, y2, label, gender, init_time, last_seen_time) for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer
                         if current_time - last_seen_time > box_expiry_duration]

        for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in expired_boxes:
            detection_duration = current_time - init_time
            log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {label}, Gender: {gender} left. Detected for {detection_duration:.2f} seconds.\n"
            log_file.write(log_entry)

        box_buffer = [(x1, y1, x2, y2, label, gender, init_time, last_seen_time) for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer
                      if current_time - last_seen_time <= box_expiry_duration]

    for (x1, y1, x2, y2, label, gender, init_time, last_seen_time) in box_buffer:
        time_elapsed = current_time - init_time
        timer_label = f"{label}, {gender} | {int(time_elapsed)}s"

        cls = [key for key, value in class_names.items() if value in label.split()[0]][0]
        color = colors[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, timer_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Deteksi Perilaku Mahasiswa', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()
