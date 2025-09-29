from flask import Blueprint, render_template, request, flash, url_for, redirect, Response, session
from .face_detector import capture_face_lbph, train_recognizer, generate_frames_recognize_then_detect, generate_frames_detect
# from . import camera

views = Blueprint('views', __name__)

@views.route('/')
def home():
    name = session.pop('recognized_name', None)
    if name:
        flash(f"✅ Face recognized: {name}", "success")
    mode = request.args.get('mode', 'detect')  #  Gets mode from query string
    return render_template("index.html", mode=mode)


# Route to handle the face capture
@views.route('/capture', methods=["POST"])
def capture():
    username = request.form.get("username").strip()
    if username:
        capture_face_lbph(username)
    else:
        flash("Please enter a valid name.", "error")
    return redirect(url_for('views.home'))


# Route to handle the training of the recognizer
@views.route('/train')
def train():
    if train_recognizer():
        print("Training started...") # Debug log
        print("Training completed...") # Debug log
        flash("🎉 Model trained successfully!", "success")
    else:
        flash("⚠️ Training failed. Make sure face data exists.", "warning")
    return redirect(url_for('views.home'))


# Route to handle the face recognition
@views.route('/recognize')
def recognize():
    flash("🔍 Starting face recognition...", "info")
    return redirect(url_for('views.home', mode="recognize"))


# Route to handle the video feed
@views.route('/video_feed')
def video_feed():
    mode = request.args.get("mode", "detect") # Default to `detection` mode
    if mode == "recognize":
        return Response(generate_frames_recognize_then_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_frames_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')
    