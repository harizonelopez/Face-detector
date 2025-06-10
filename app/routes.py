from flask import Blueprint, render_template, request, flash, url_for, redirect, Response
from .face_detector import capture_face_lbph, train_recognizer, recognize_face_live, generate_frames

views = Blueprint('views', __name__)


@views.route('/')
def home():
    return render_template("index.html")


@views.route('/capture', methods=["POST"])
def capture():
    username = request.form.get("username")
    if username:
        capture_face_lbph(username)
    else:
        flash("Please enter a valid name.", "error")
    return redirect(url_for('views.home'))


@views.route('/train')
def train():
    train_recognizer()
    flash("Model trained successfully!", "success")
    return redirect(url_for('views.home'))


@views.route('/recognize')
def recognize():
    recognize_face_live()
    flash("Recognition session ended.", "info")
    return redirect(url_for('views.home'))


@views.route('/video_feed')
def video_feed():
   return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

