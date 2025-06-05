from flask import Blueprint, render_template, request, flash, url_for, redirect
from .face_detector import capture_face

views = Blueprint('views', __name__)


@views.route('/')
def home():
    return render_template("index.html")


@views.route('/capture', methods=["POST"])
def capture():
    username = request.form.get("username")
    if username:
        capture_face(username)
    else:
        flash("Please enter a valid name.", "error")
    return redirect(url_for('views.home'))  # Redirect to home after capture
