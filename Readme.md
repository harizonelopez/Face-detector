# Real-Time Face Detection System

This project demonstrates real-time face detection using OpenCV's DNN (Deep Neural Network) module and a pre-trained Caffe model. It captures video from your webcam, detects faces in each frame, and displays bounding boxes and confidence scores.

---

## 📸 Demo

*Example only. Actual live webcam will be used.*

---

## 📂 Project Structure

face_detection_project/
├── face_detection.py
├── face_model.caffemodel         # (You need to download this)
├── deploy.prototxt               # (You need to download this)
├── requirements.txt
└── README.md



## ⚙️ Requirements

Make sure you have the following installed:

- Python 3.6+
- OpenCV with DNN module
- NumPy

You can install the dependencies with:

```bash
    pip install -r requirements.txt
```


## 🧠 Model Details

This project uses a deep learning model based on a ResNet10 SSD (Single Shot Multibox Detector) architecture.


### Download the required files:
```bash
    deploy.prototxt
    res10_300x300_ssd_iter_140000.caffemodel
```

Rename or place them in the same folder as:

```bash
    face_model.caffemodel
    deploy.prototxt
```


## 🚀 How to Run

```bash
    python face_detection.py
```

Press `q` to quit the webcam window.


## 🎯 Features

- ✅ Real-time face detection from webcam

- ✅ Confidence score display

- ✅ Bounding boxes around detected faces

- ✅ Frames-per-second (FPS) calculation

- ✅ Input validation (model file and webcam check)

- ✅ Auto-clamping to prevent frame-bound errors



## 📌 Notes

- Works best in well-lit environments.

- Accuracy threshold can be modified from `confidence > 0.5` in the `face_detector.py`.



## 🔧 Future Improvements
- Add face recognition

- Integrate emotion or age/gender detection

- Save detected faces

- Deploy via Flask or Streamlit


## 👨‍💻 Author

- GitHub: @harizonelopez
- Email: @harizonelopez23@gmail.com



## 📜 License

    MIT License


### 📄 `requirements.txt`

- opencv-python
- numpy
