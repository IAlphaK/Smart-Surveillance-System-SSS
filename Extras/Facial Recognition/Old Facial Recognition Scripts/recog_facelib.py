import face_recognition
import cv2
import os
import numpy as np
import requests
import time
from config import RTSP_URL

# Base directory of the script
base_directory = os.path.dirname(os.path.abspath(__file__))

# Setup
photo_directory = os.path.join(base_directory, 'Students')
student_encodings = []
student_names = []

# Load each student's facial data from images
for filename in os.listdir(photo_directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")): 
        image_path = os.path.join(photo_directory, filename)
        student_image = face_recognition.load_image_file(image_path)
        try:
            student_face_encoding = face_recognition.face_encodings(student_image)[0]
            student_encodings.append(student_face_encoding)
            student_names.append(filename[:-4])
        except IndexError:
            continue

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

    # Resize frame to 1/4 size for speed up face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    frame_face_locations = face_recognition.face_locations(small_frame)
    frame_face_encodings = face_recognition.face_encodings(small_frame, frame_face_locations)

    for (top, right, bottom, left), face_encoding in zip(frame_face_locations, frame_face_encodings):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        matches = face_recognition.compare_faces(student_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            face_distances = face_recognition.face_distance(student_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = student_names[best_match_index]
                current_time = time.time()
                if (current_time - last_detection_time) > alert_interval:
                    last_detection_time = current_time
                    # Send a POST request to the server
                    payload = {'rollNumber': name, 'cameraName': 'Abdullah\'s Room'}
                    response = requests.post('http://localhost:5001/report/incident', json=payload)
                    print(f"Alert sent to server: {response.text}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
