import cv2
import dlib
import os
import numpy as np
import requests
import time
from config import RTSP_URL

# Base directory of the script
base_directory = os.path.dirname(os.path.abspath(__file__))

# Paths to the models (ensure these paths are correct)
shape_predictor_path = os.path.join(base_directory, "shape_predictor_5_face_landmarks.dat")
face_rec_model_path = os.path.join(base_directory, "dlib_face_recognition_resnet_model_v1.dat")

# Initialize Dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Initialize Dlib's face recognition model (ResNet-based)
sp = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Setup
photo_directory = os.path.join(base_directory, 'Students')
student_encodings = []
student_names = []

# Load each student's facial data from images
for filename in os.listdir(photo_directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(photo_directory, filename)
        student_image = dlib.load_rgb_image(image_path)
        detections = detector(student_image, 1)
        if detections:
            shape = sp(student_image, detections[0])
            student_face_encoding = np.array(face_rec_model.compute_face_descriptor(student_image, shape))
            student_encodings.append(student_face_encoding)
            student_names.append(filename[:-4])

# RTSP Stream using the imported URL
video_capture = cv2.VideoCapture(RTSP_URL)

if not video_capture.isOpened():
    print("Error opening video stream")
    exit()

# Create a named window that can be resized
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 2560, 1600)  # Set the window size

last_detection_time = 0
alert_interval = 30  # seconds
frame_skip = 5  # Process every 5th frame
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip this frame

    # Resize frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    detections = detector(small_frame, 1)

    for detection in detections:
        shape = sp(small_frame, detection)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(small_frame, shape))

        matches = np.linalg.norm(student_encodings - face_encoding, axis=1) < 0.6  # Threshold for matching
        name = "Unknown"

        if np.any(matches):
            best_match_index = np.argmin(np.linalg.norm(student_encodings - face_encoding, axis=1))
            name = student_names[best_match_index]
            current_time = time.time()
            if (current_time - last_detection_time) > alert_interval:
                last_detection_time = current_time
                # Send a POST request to the server
                payload = {'rollNumber': name, 'cameraName': 'Abdullah\'s Room'}
                response = requests.post('http://localhost:5001/report/incident', json=payload)
                print(f"Alert sent to server: {response.text}")

        left, top, right, bottom = (int(detection.left() * 4), int(detection.top() * 4), int(detection.right() * 4), int(detection.bottom() * 4))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()