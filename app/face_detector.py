import cv2
import os
import numpy as np
from flask import flash

# Load the DNN model globally
modelFile = "face_model.caffemodel"
configFile = "deploy.prototxt"

if os.path.exists(modelFile) and os.path.exists(configFile):
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    net = None
    print("[ERROR] DNN model or config not found!")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# camera = cv2.VideoCapture(0)

# Function to capture face images using LBPH {Local Binary Patterns Histograms}
def capture_face_lbph(user_name):
    user_name = user_name.strip().lower().replace(" ", "_")
    if not user_name:
        flash("Invalid username! Please enter a valid name.", "error")
        print("[404: ERROR] Invalid username!")
        return

    dataset_dir = os.path.join("face_data", "images")
    label_map_file = os.path.join("face_data", "labels.txt")
    os.makedirs(dataset_dir, exist_ok=True)

    user_id = None
    label_map = {}

    # Read existing labels
    if os.path.exists(label_map_file):
        with open(label_map_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                id_, name = line.strip().split(",")
                label_map[name] = int(id_)

    # Assign new ID if user not found
    if user_name not in label_map:
        user_id = len(label_map) + 1
        label_map[user_name] = user_id
        with open(label_map_file, "a") as f:
            f.write(f"{user_id},{user_name}\n")
    else:
        user_id = label_map[user_name]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        flash("Cannot access webcam.", "error")
        return

    # Setup the display window
    win_name = "Capturing Face - Press Q to stop"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 480)
    cv2.moveWindow(win_name, 300, 150)  # Adjust position as needed

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            flash("Failed to grab frame.", "error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(dataset_dir, f"{user_name}.{user_id}.{count+1}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow(win_name, frame)

        # Break if 'q' is pressed or enough samples are collected
        if cv2.waitKey(1) & 0xFF == ord("q") or count >= 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    flash(f"🎉 Success! {user_name.title()}, face captured.", "success")
    print(f"[200: INFO] Captured {count} samples for {user_name}")


# Function to train the LBPH face recognizer
def train_recognizer():
    dataset_dir = os.path.join("face_data", "images")
    label_map_file = os.path.join("face_data", "labels.txt")
    model_path = os.path.join("face_data", "trained_model.yml")

    if not os.path.exists(label_map_file):
        print("[503: ERROR] No label map found. Please register some faces first.")
        return

    # Load label mappings
    label_map = {}
    with open(label_map_file, "r") as f:
        for line in f:
            id_, name = line.strip().split(",")
            label_map[int(id_)] = name

    # Prepare training data
    faces = []
    labels = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            path = os.path.join(dataset_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            user_id = int(filename.split(".")[1])  # user.{id}.{count}.jpg
            faces.append(img)
            labels.append(user_id)

    if not faces:
        print("[404: ERROR] No training data found.")
        return

    # Create LBPH recognizer and train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)

    print("[200: INFO] Training completed and model saved at:", model_path)


# Function to recognize faces in live video feed
def recognize_face_live():
    model_path = os.path.join("face_data", "trained_model.yml")
    label_map_file = os.path.join("face_data", "labels.txt")

    if not os.path.exists(model_path) or not os.path.exists(label_map_file):
        print("[503: ERROR] Model or label map not found. Please train the model first.")
        return

    # Load label map
    label_map = {}
    with open(label_map_file, "r") as f:
        for line in f:
            id_, name = line.strip().split(",")
            label_map[int(id_)] = name

    # Load LBPH model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[404: ERROR] Cannot access webcam.")
        return

    print("[200: INFO] Starting live recognition... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[404: ERROR] Failed to grab frame.")
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
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            label, confidence = recognizer.predict(gray)

            name = label_map.get(label, "Unknown")
            text = f"{name} ({confidence:.2f})"
            color = (0, 255, 0) if confidence < 70 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def generate_frames():
    model_path = os.path.join("face_data", "trained_model.yml")
    label_map_file = os.path.join("face_data", "labels.txt")

    if not os.path.exists(model_path) or not os.path.exists(label_map_file):
        print("[503: ERROR] Model or label map not found.")
        return

    # Load label map
    label_map = {}
    with open(label_map_file, "r") as f:
        for line in f:
            id_, name = line.strip().split(",")
            label_map[int(id_)] = name

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
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

            # Skip if face_crop is empty or dimensions are too small
            if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Resize gray image to ensure it is large enough for prediction
            try:
                label, conf = recognizer.predict(gray)
                name = label_map.get(label, "Unknown")
                color = (0, 255, 0) if conf < 70 else (0, 0, 255)
                text = f"{name} ({conf:.1f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except:
                pass  # Skip prediction if face too small or model error

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


"""
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the output frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
"""