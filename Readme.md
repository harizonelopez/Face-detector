# Real-Time Face Detection & Recognition System

A real-time face detection and recognition system built with **Python, Flask, and OpenCV**. The application uses OpenCV's DNN module for face detection and an **LBPH (Local Binary Patterns Histograms) face recognizer** for identifying trained faces.

It captures video from your webcam, detects faces in real time, displays bounding boxes and confidence information, and allows users to capture and train faces for live recognition.

---

## 📸 Demo

> **Example only. Actual detection and recognition use the live webcam.**

---

## ⚙️ Requirements

Make sure you have the following installed:

* Python 3.12
* Flask 3.1+
* OpenCV Contrib
* NumPy
* dlib
* face-recognition
* Pillow
* setuptools

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Virtual Environment

It is recommended to run the project inside a virtual environment.

### Windows (PowerShell)

```bash
py -3.12 -m venv .env
```

Activate it:

```bash
.env\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .env
```

Activate it:

```bash
source .env/bin/activate
```

---

## 🧠 Detection Model

The face detection system uses OpenCV's **DNN (Deep Neural Network)** module with a pre-trained **ResNet10 SSD (Single Shot Multibox Detector)** model.

### Required model files

The project requires:

```text
face_model.caffemodel
deploy.prototxt
```

The original model files are:

```text
res10_300x300_ssd_iter_140000.caffemodel
deploy.prototxt
```

Place them in the project root directory and make sure the Caffe model is named:

```text
face_model.caffemodel
```

---

## 👤 Face Recognition

The project also supports face recognition using OpenCV's **LBPHFaceRecognizer**.

### Recognition workflow

1. Capture a face using the application.
2. The captured face is stored in the `face_data/images` directory.
3. Train the LBPH recognizer.
4. The trained model is saved to:

```text
face_data/trained_model.yml
```

5. The application can then recognize trained faces through the live webcam.

Multiple people can be added to the training dataset.

---

## 🚀 How to Run

Activate your virtual environment first:

```bash
.env\Scripts\activate
```

Then start the Flask application:

```bash
python run.py
```

Open the application in your browser using the local address shown by Flask.

---

## 🎯 Features

* ✅ Real-time face detection
* ✅ Live webcam streaming
* ✅ Face bounding boxes
* ✅ Confidence score display
* ✅ ResNet10 SSD face detection
* ✅ LBPH face recognition
* ✅ Capture faces for training
* ✅ Train custom face recognition models
* ✅ Recognize trained faces through the webcam
* ✅ Input validation for model files and webcam
* ✅ Frame-bound protection
* ✅ Flask-based web interface

---

## 📌 Notes

* A working webcam is required.
* Detection and recognition work best in well-lit environments.
* Only faces that have been captured and included in the training dataset can be recognized.
* The detection confidence threshold can be adjusted in `face_detector.py`.
* Keep the `face_data` directory available because it contains the captured training data, labels, and trained recognition model.

---

## 👨‍💻 Author

**Harizonelopez**

* Email: [harizonelopez23@gmail.com](mailto:harizonelopez23@gmail.com)

---

## 📜 License

This project is licensed under the **MIT License**.
