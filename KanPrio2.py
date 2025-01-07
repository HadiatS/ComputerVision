import cv2
import numpy as np

# Buat video capture
video = cv2.VideoCapture(2)  # Gunakan kamera default

if not video.isOpened():
    print("Error: Tidak dapat mengakses kamera.")
    exit()

# Pengaturan ukuran frame
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Penulis video untuk menyimpan hasil (opsional)
output = cv2.VideoWriter(
    'thermal_video.avi', 
    cv2.VideoWriter_fourcc(*'XVID'), 
    30, 
    (frame_width, frame_height)
)

while True:
    rval, frame = video.read()
    if not rval:
        print("Error: Tidak dapat membaca frame.")
        break

    # Konversi ke grayscale sebagai simulasi data suhu
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalisasi ke rentang [0, 255] untuk visualisasi
    normalized_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

    # Terapkan peta warna termal (COLORMAP_JET adalah standar untuk visualisasi termal)
    thermal_frame = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)

    # Tampilkan video termal
    cv2.imshow("Thermal Video", thermal_frame)

    # Simpan frame termal ke video output
    output.write(thermal_frame)

    # Keluar jika menekan ESC
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# Bersihkan semua sumber daya
video.release()
output.release()
cv2.destroyAllWindows()
