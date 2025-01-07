import cv2
import numpy as np

# Konfigurasi rentang suhu simulasi
min_temp = 20  # Suhu minimum (20°C)
max_temp = 80  # Suhu maksimum (80°C)

# Fungsi callback untuk mouse event
pointer_position = (0, 0)  # Inisialisasi posisi pointer

def mouse_callback(event, x, y, flags, param):
    global pointer_position
    if event == cv2.EVENT_MOUSEMOVE:  # Update posisi pointer saat mouse bergerak
        pointer_position = (x, y)

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
    'thermal_video_with_pointer.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    30,
    (frame_width, frame_height)
)

# Daftarkan callback mouse
cv2.namedWindow("Thermal Video with Pointer")
cv2.setMouseCallback("Thermal Video with Pointer", mouse_callback)

while True:
    rval, frame = video.read()
    if not rval:
        print("Error: Tidak dapat membaca frame.")
        break

    # Konversi ke grayscale sebagai simulasi data suhu
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalisasi ke rentang [0, 255]
    normalized_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

    # Terapkan peta warna termal
    thermal_frame = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)

    # Hitung nilai suhu dari grayscale
    temp_frame = normalized_frame.astype(np.float32)  # Konversi ke float untuk operasi matematika
    temp_frame = min_temp + (temp_frame / 255.0) * (max_temp - min_temp)  # Peta ke suhu [20°C, 80°C]

    # Dapatkan nilai suhu di posisi pointer
    px, py = pointer_position
    if 0 <= px < frame_width and 0 <= py < frame_height:
        temp_value = temp_frame[py, px]
        text = f"{temp_value:.1f}C"

        # Gambarkan pointer dan nilai suhu
        cv2.circle(thermal_frame, (px, py), 10, (255, 255, 255), 2)  # Pointer lingkaran
        cv2.putText(
            thermal_frame,
            text,
            (px + 15, py - 10),  # Posisi teks di dekat pointer
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Ukuran font
            (255, 255, 255),  # Warna teks (putih)
            2,  # Ketebalan font
            cv2.LINE_AA
        )

    # Tampilkan video termal
    cv2.imshow("Thermal Video with Pointer", thermal_frame)

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
