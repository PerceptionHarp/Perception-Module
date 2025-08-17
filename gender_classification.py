# save_live_gender_vlc.py
import time
import cv2
from ultralytics import YOLO
import os

# ---------- SETTINGS ----------
MODEL_PATH = "/home/talha/runs/classify/train2/weights/best.pt"
OUTPUT_PATH = "/home/talha/gender_output.mp4"  # VLC-friendly file
CODEC = "mp4v"   # 'mp4v' for .mp4, or 'XVID' for .avi
DEFAULT_FPS = 20.0
# ------------------------------

# Check model path
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load YOLO classification model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

# Get frame size & FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
print(f"Camera opened: {width}x{height} @ {fps:.1f} FPS")

# Video writer (VLC-compatible)
fourcc = cv2.VideoWriter_fourcc(*CODEC)
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Check OpenCV installation.")

prev_time = time.time()
fps_smoothed = fps
smoothing = 0.9

print("Press 'q' to stop and save the video.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not read; exiting.")
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64,64))

    # Classify each detected face
    for (x, y, w, h) in faces:
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Run YOLO classifier
        res = model.predict(source=face_crop, imgsz=224, verbose=False, show=False)

        label = "unknown"
        conf = 0.0
        try:
            r = res[0]
            top_idx = int(r.probs.top1)
            label = r.names[top_idx]
            conf = float(r.probs.top1conf)
        except Exception:
            pass

        # Choose color for label
        color = (0, 255, 0) if "man" in label.lower() or "male" in label.lower() else (255, 0, 255)

        # Draw bounding box & label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Smooth FPS display
    now = time.time()
    instant_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
    prev_time = now
    fps_smoothed = smoothing * fps_smoothed + (1 - smoothing) * instant_fps
    cv2.putText(frame, f"FPS: {fps_smoothed:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # Show and save frame (always same resolution)
    cv2.imshow("Live Gender (press q to quit)", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopping by user request.")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Saved output to: {OUTPUT_PATH}")
print("💡 To open in VLC:  vlc " + OUTPUT_PATH)

