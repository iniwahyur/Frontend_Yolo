import cv2
import time
from ultralytics import YOLO

# Memuat model YOLO yang telah dilatih
model = YOLO('best.pt')  # Path ke model YOLO

# Definisikan warna khusus untuk setiap kelas dalam model Anda
colors = {
    0: (0, 0, 255),      # Merah untuk "Risyad Maulana"
    1: (0, 255, 0)      # Hijau untuk "Adhitya Hendri"
    # 2: (255, 0, 0),      # Biru untuk "Fadeta Ilham Gandi"
    # 3: (255, 255, 0),    # Cyan untuk "Aldhi Ramadhan"
    # 4: (255, 0, 255),    # Magenta untuk "Alif Andika"
    # 5: (0, 255, 255),    # Kuning untuk "Allam Rosyad Akbar"
    # 6: (128, 0, 128),    # Ungu untuk "Annisa Rizkyta"
    # 7: (128, 128, 0),    # Olive untuk "Ardiansyah Al Faiz"
    # 8: (0, 128, 128),    # Teal untuk "Arif"
    # 9: (0, 128, 0),      # Hijau gelap untuk "Ayunda Lintang"
    # 10: (128, 0, 0),     # Merah gelap untuk "Azwar Syifa"
    # 11: (0, 0, 128),     # Navy untuk "Daffa"
    # 12: (128, 128, 128), # Abu-abu untuk "Desfianto"
    # 13: (64, 64, 64),    # Abu-abu gelap untuk "Dhery Akbar Ramadhan"
    # 14: (192, 192, 192), # Abu-abu terang untuk "Iqbal Farissi"
    # 15: (64, 0, 64),     # Ungu gelap untuk "Merthisia Roszi"
}

# Definisikan nama kelas sesuai dengan model Anda
class_names = {
    0: "jefri",
    1: "manca"
    # 2: "Alif Andika",
    # 3: "Allam Rosyad Akbar",
    # 4: "Annisa Rizkyta",
    # 5: "Ardiansyah Al Faiz",
    # 6: "Arif",
    # 7: "Ayunda Lintang",
    # 8: "Azwar Syifa",
    # 9: "Daffa",
    # 10: "Desfianto",
    # 11: "Dhery Akbar Ramadhan",
    # 12: "Fadeta Ilham Gandi",
    # 13: "Iqbal Farissi",
    # 14: "Merthisia Roszi",
    # 15: "Risyad Maulana",
}

# Fungsi untuk menggambar bounding box dengan label dan warna masing-masing
def custom_plot(results):
    img = results[0].orig_img  # Gambar asli
    for box in results[0].boxes:
        cls = int(box.cls[0])  # Dapatkan indeks kelas
        color = colors.get(cls, (255, 255, 255))  # Dapatkan warna, default putih jika tidak ditemukan
        label = f"{class_names.get(cls, 'Tidak Dikenal')} {box.conf[0]:.2f}"  # Label dengan skor kepercayaan
        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
        
        # Gambar kotak pembatas
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Tampilkan label di atas kotak pembatas
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Dapatkan tanggal dan waktu saat ini
        current_time = time.strftime("%d-%m-%Y %H:%M:%S")
        
        # Tampilkan tanggal dan waktu di bawah kotak pembatas
        cv2.putText(img, current_time, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img

# Fungsi untuk mendeteksi nama dari stream kamera
def detect_person_names_from_camera(confidence_threshold=0.7, iou_threshold=0.5):
    kamera = cv2.VideoCapture(0)  # Gunakan kamera default
    frame_count = 0
    frame_skip = 5  # Lewati setiap 5 frame untuk efisiensi

    while True:
        berhasil, frame = kamera.read()  # Tangkap frame dari kamera
        if not berhasil:
            print("Gagal menangkap gambar")
            break

        frame_count += 1
        
        if frame_count % frame_skip == 0:
            # Lakukan deteksi menggunakan model YOLO
            results = model(frame, conf=confidence_threshold, iou=iou_threshold)
            
            # Gambar kotak pembatas dengan label dan tanggal/waktu
            frame_with_boxes = custom_plot(results)

            # Cetak nama objek yang terdeteksi
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_idx = int(box.cls[0])
                    person_name = class_names.get(class_idx, "Tidak Dikenal")
                    print(f"Terdeteksi: {person_name}")
            
            # Tampilkan frame dengan kotak pembatas dan waktu
            cv2.imshow('Absensi Kehadiran', frame_with_boxes)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup semua jendela
    kamera.release()
    cv2.destroyAllWindows()

# Mulai mendeteksi nama dari webcam
if __name__ == "__main__":
    detect_person_names_from_camera()