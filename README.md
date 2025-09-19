# Drowsiness-Detection-System
A real-time Drowsiness Detection System built using Python, Keras, OpenCV, and Streamlit. This system detects driver drowsiness via live webcam or IP camera to prevent accidents.

# Features

Real-time drowsiness detection using face and eye detection.

Supports local cameras and IP cameras.

Visual alerts on Streamlit dashboard when drowsiness is detected.

Audio alert for immediate notification.

Saves images of drowsy instances for record-keeping.

Simple, interactive Streamlit UI with Home, IP Camera, and Camera options.

# Requirements

Python 3.8+

Streamlit

OpenCV (opencv-python)

Keras / TensorFlow

NumPy

# Models and XML files:

eye.h5 → Pre-trained Keras eye state classification model.

face.xml → Haar cascade classifier for face detection.

# Usage

Run the Streamlit app:

streamlit run DDT.py

Select the menu option from the sidebar:

HOME → About page.

IP CAMERA → Enter your IP camera URL.

CAMERA → Select your local camera (0 for primary, 1 for secondary).

Click Start Detection to start monitoring.

Click Stop Detection to stop the camera stream.
