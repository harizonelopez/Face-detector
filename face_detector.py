import cv2
import numpy as np
import os
import time

def main():
    # Load model paths
    modelFile = "face_model.caffemodel"
    configFile = "deploy.prototxt"

    # Check if model files exist
    if not os.path.isfile(modelFile) or not os.path.isfile(configFile):
        print("Model or config files not found. Please check the file paths.")
        return

    # Load the Caffe model
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prev_time = time.time()

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

        face_count = 0  # Count number of faces detected

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                face_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save frame when face is detected
                filename = f"face_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)

        # Display face count
        cv2.putText(frame, f"Faces: {face_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Deep Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
