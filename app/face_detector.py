import cv2
import numpy as np
import os
import time
from datetime import datetime
from flask import flash

# Load model once globally
modelFile = "face_model.caffemodel"
configFile = "deploy.prototxt"

# Check if model and config files exist
if not os.path.isfile(modelFile) or not os.path.isfile(configFile):
    print("[503: ERROR] Model or config file missing.")
    net = None
else:
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def capture_face(user_name):
    user_name = user_name.strip().lower().replace(" ", "_")
    if not user_name:
        flash("Invalid username! Please enter a valid name.", "error")
        print("[404: ERROR] Invalid username!")
        return

    if net is None:
        flash("Face detection model not loaded. Check model files.", "error")
        print("[503: ERROR] Model not initialized.")
        return

    save_dir = os.path.join("static", "saved-faces")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash("Cannot access webcam. Please check your camera settings.", "error")
        print("[404: ERROR] Cannot access webcam.")
        return

    prev_time = time.time()
    saved_count = 0
    max_faces_to_save = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            flash("Failed to grab frame. Please check your webcam.", "error")
            print("[503: ERROR] Failed to grab frame.")
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     1.0, (300, 300),
                                     (104.0, 177.0, 123.0),
                                     swapRB=False, crop=False)

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"{user_name}_{timestamp}.jpg")
            cv2.imwrite(filename, face_crop)

            flash(f"🎉 Success! {user_name.title()}, your face has been captured and saved securely.", "success")
            print(f"[200: INFO] Saved: {filename}")
            saved_count += 1

            if saved_count >= max_faces_to_save:
                break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Capturing Face", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or saved_count >= max_faces_to_save:
            break

    cap.release()
    cv2.destroyAllWindows()
