import cv2
import numpy as np
import os
import time
from datetime import datetime

def main():
    modelFile = "face_model.caffemodel"
    configFile = "deploy.prototxt"
    
    user_name = input("\nEnter your name: ").strip().lower().replace(" ", "_")  # Enter the username
    if not user_name:
        print("Invalid username! Exiting...")
        return

    save_dir = "saved_faces"
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isfile(modelFile) or not os.path.isfile(configFile):
        print("Model or config files not found. Please check the file paths.")
        return

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prev_time = time.time() 
    saved_count = 0
    max_faces_to_save = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
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

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if saved_count < max_faces_to_save:
                    face_crop = frame[y1:y2, x1:x2]
                    # Check face size {skip too small faces}
                    if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                        continue
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{save_dir}/{user_name}_{timestamp}.jpg"
                    cv2.imwrite(filename, face_crop)
                    print(f"[INFO] Saved: {filename}")
                    saved_count += 1

        # Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Deep Face Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or saved_count >= max_faces_to_save:
            print(f"[INFO] Finished saving {saved_count} face model(s).")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
