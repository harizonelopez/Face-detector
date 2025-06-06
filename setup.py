from setuptools import setup, find_packages

setup(
    name="face_recognition_app",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask>=2.0.0",
        "opencv-python",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "face-recognition-run = run:main",
        ],
    },
    author="Your Name",
    description="A Flask app for face recognition using LBPH",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: Flask",
    ],
)
