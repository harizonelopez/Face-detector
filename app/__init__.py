from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'J@rvis_007'
    app.config['UPLOAD_FOLDER'] = 'static/saved-faces'

    from .routes import views
    app.register_blueprint(views)

    return app

