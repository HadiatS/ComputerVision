import cv2
import os
from ultralytics import YOLO

# Menginisialisasi model YOLOv5
model = YOLO('yolov5s.pt')  # Gunakan model yang sesuai, bisa juga yang lebih besar 'yolov5m.pt'

# Inisialisasi kamera USB (biasanya kamera pertama adalah index 0)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Kamera tidak dapat diakses")
    exit()

# Folder untuk menyimpan gambar (diabaikan dalam kasus ini)
save_folder = r'C:\Users\User\Pictures'

# Pastikan folder ada (diabaikan dalam kasus ini)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Variabel untuk menghitung orang yang lewat
person_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame")
        break

    # Deteksi objek (termasuk orang)
    results = model(frame)

    # Ambil hasil deteksi (bounding boxes, confidence, class IDs)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in format [xmin, ymin, xmax, ymax]
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence score
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

    # Variabel untuk mendeteksi apakah ada orang
    person_detected = False

    # Iterasi untuk setiap deteksi dan filter untuk orang (class_id = 0)
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box

        # Hanya deteksi orang (kelas 0)
        if class_ids[i] == 0:
            person_detected = True  # Menandakan ada orang yang terdeteksi

            # Gambar bounding box untuk orang yang terdeteksi
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, 'Hati-hati, ada orang!', (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Jika ada orang yang terdeteksi, tampilkan peringatan
    if person_detected:
        print("Hati-hati, ada orang!")

    # Tampilkan frame dengan bounding box orang yang terdeteksi
    cv2.imshow("Person Detection", frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup kamera
cap.release()
cv2.destroyAllWindows()
