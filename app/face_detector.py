import cv2
import os
import numpy as np
from flask import flash
from .camera import Camera

# Paths
modelFile = "face_model.caffemodel"
configFile = "deploy.prototxt"
label_map_file = os.path.join("face_data", "labels.txt")
model_path = os.path.join("face_data", "trained_model.yml")

# Load DNN model
if os.path.exists(modelFile) and os.path.exists(configFile):
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    net = None
    print("[ERROR] DNN model or config not found!")


# Ensure the face_data directory exists
def load_face_cascade():
    # First try loading from local 'face_data' folder
    local_path = os.path.join("face_data", "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        cascade = cv2.CascadeClassifier(local_path)
        if not cascade.empty():
            print(f"[100: INFO] Loaded Haar Cascade from {local_path}")
            return cascade

    # Fallback to OpenCV's built-in path
    default_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(default_path)
    if cascade.empty():
        raise IOError(f"[503: ERROR] Failed to load Haar Cascade from both local and OpenCV default path.")
    print(f"[100: INFO] Loaded Haar Cascade from OpenCV default path: {default_path}")
    return cascade

# Load face cascade & the camera globally
face_cascade = load_face_cascade()
camera = None


# To ensure the face_data directory exists 
def get_camera():
    global camera
    if camera is None:
        try:
            camera = Camera()
        except RuntimeError as e:
            print(f"[404: ERROR] {e}")
            camera = None
    return camera


# Capture face using LBPH (Local Binary Patterns Histograms)
def capture_face_lbph(user_name):
    cam = get_camera()
    frame = cam.get_frame()
    if frame is None:
        print("[404: ERROR] No frame captured.")
        return

    user_name = user_name.strip().lower().replace(" ", "_")
    if not user_name:
        flash("Invalid username!", "error")
        return

    dataset_dir = os.path.join("face_data", "images")
    os.makedirs(dataset_dir, exist_ok=True)

    label_map = {}
    user_id = None

    if os.path.exists(label_map_file):
        with open(label_map_file, "r") as f:
            for line in f:
                id_, name = line.strip().split(",")
                label_map[name] = int(id_)

    if user_name not in label_map:
        user_id = len(label_map) + 1
        label_map[user_name] = user_id
        with open(label_map_file, "a") as f:
            f.write(f"{user_id},{user_name}\n")
    else:
        user_id = label_map[user_name]

    win_name = "Capturing Face - Press Q to stop"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 480)

    count = 0
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            path = os.path.join(dataset_dir, f"{user_name}.{user_id}.{count+1}.jpg")
            cv2.imwrite(path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 1:
            break

    cv2.destroyAllWindows()
    flash(f"🎉 {user_name.title()} captured!", "success")
    print(f"[200: INFO] Captured {count} samples for {user_name}")


# Train the recognizer using captured faces models
def train_recognizer():
    cam = get_camera()
    frame = cam.get_frame()
    if frame is None:
        print("[404: ERROR] No frame captured.")
        return False

    dataset_dir = os.path.join("face_data", "images")
    if not os.path.exists(label_map_file):
        print("[503: ERROR] No label map. Capture face first.")
        return False

    label_map = {}
    with open(label_map_file, "r") as f:
        for line in f:
            id_, name = line.strip().split(",")
            label_map[int(id_)] = name

    faces, labels = [], []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            path = os.path.join(dataset_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                try:
                    user_id = int(filename.split(".")[1])
                    faces.append(img)
                    labels.append(user_id)
                except ValueError:
                    print(f"[WARNING] Skipped: {filename}")

    if not faces:
        print("[404: ERROR] No training data found.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)

    print("[200: INFO] Model trained and saved.")
    return True


# Recognize face using live camera feed
def recognize_face_live():
    cam = get_camera()
    frame = cam.get_frame()
    if frame is None:
        return None, "Unknown"

    if not os.path.exists(model_path) or not os.path.exists(label_map_file):
        print("[503: ERROR] Model/labels missing.")
        return None, "Unknown"

    label_map = {}
    with open(label_map_file, "r") as f:
        for line in f:
            id_, name = line.strip().split(",")
            label_map[int(id_)] = name

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            try:
                label, conf = recognizer.predict(gray)
                name = label_map.get(label, "Unknown")
                return frame, name
            except:
                continue

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return None, "Unknown"


# Generate frames for streaming
def generate_frames(mode="detect"):
    cam = get_camera()
    if not cam or not cam.cap.isOpened():
        print("[404: ERROR] Camera unavailable.")
        return

    recognizer = None
    label_map = {}
    if mode == "recognize" and os.path.exists(model_path) and os.path.exists(label_map_file):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
        with open(label_map_file, "r") as f:
            for line in f:
                id_, name = line.strip().split(",")
                label_map[int(id_)] = name

    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            label = "Face"
            color = (0, 255, 0)

            if mode == "recognize" and recognizer:
                try:
                    id_, conf = recognizer.predict(roi)
                    label = f"{label_map.get(id_, 'Unknown')} ({int(conf)})"
                    color = (0, 255, 255)
                except:
                    pass

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# Generate frames for detection mode
def generate_frames_detect():
    return generate_frames(mode="detect")


# Generate frames for recognization mode
def generate_frames_recognize_then_detect():
    recognized = False
    while not recognized:
        frame, label = recognize_face_live()
        if frame is None:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        if label not in ["Unknown", ""]:
            print(f"[INFO] Recognized: {label}")
            recognized = True
            break

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # Then fallback to normal detect
    for frame in generate_frames_detect():
        yield frame
