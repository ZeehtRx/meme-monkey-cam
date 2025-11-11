import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 🔧 Ganti ke 1 jika pakai kamera eksternal
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Tidak bisa membuka kamera. Coba ganti angka di VideoCapture(0/1/2).")
    exit()

print("✅ Kamera aktif. Tekan ESC untuk keluar.")

# --- 👇 Load gambar status ---
img_near = cv2.imread("near.png")  # Gambar saat tangan DEKAT wajah
img_far = cv2.imread("far.png")    # Gambar saat tangan JAUH dari wajah

# Jika gambar tidak ditemukan, buat gambar otomatis sebagai fallback
if img_near is None:
    print("⚠️ near.png tidak ditemukan. Membuat gambar otomatis...")
    img_near = np.zeros((400, 400, 3), dtype=np.uint8)
    img_near[:] = (0, 150, 0)  # Hijau tua
    cv2.putText(img_near, "✅ DEKAT", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

if img_far is None:
    print("⚠️ far.png tidak ditemukan. Membuat gambar otomatis...")
    img_far = np.ones((400, 400, 3), dtype=np.uint8) * 255
    img_far[:] = (0, 0, 255)  # Merah
    cv2.putText(img_far, "❌ JAUH", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ Gagal membaca frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    face_results = face_detection.process(rgb_frame)
    face_center = None
    if face_results.detections:
        detection = face_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = int(bbox.xmin * w + bbox.width * w / 2)
        y = int(bbox.ymin * h + bbox.height * h / 2)
        face_center = np.array([x, y])
        cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)

    # Deteksi tangan
    hand_results = hands.process(rgb_frame)
    hand_near_face = False

    if hand_results.multi_hand_landmarks and face_center is not None:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            idx_finger = hand_landmarks.landmark[8]  # ujung telunjuk
            hand_point = np.array([int(idx_finger.x * w), int(idx_finger.y * h)])
            cv2.circle(frame, tuple(hand_point), 8, (0, 255, 255), -1)

            distance = np.linalg.norm(face_center - hand_point)
            if distance < 120:  # Threshold jarak (pixel)
                hand_near_face = True

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan gambar sesuai kondisi
    display_img = img_near if hand_near_face else img_far
    cv2.imshow("Status Tangan", display_img)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()