import face_recognition
import cv2
import os
import numpy as np
import requests
import time

# Base directory of the script
base_directory = os.path.dirname(os.path.abspath(__file__))

# Path to the recorded video file
#video_file_path = os.path.join(base_directory, 'Footage', 'Camera.MP4')
#video_file_path = r'C:\Users\alpha\PycharmProjects\v1SpatioTemporalInferencer\test\s12.mp4'
video_file_path = r'C:\Users\alpha\Downloads\Working Directory\Extras\smoking\s13.mp4'
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

video_capture = cv2.VideoCapture(video_file_path)  # Assume webcam for real-time processing
last_detection_time = 0
alert_interval = 30  # seconds
frame_count = 0
frame_skip = 2  # Process every 2nd frame for balance

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Frame grab failed, skipping...")
        continue  # Skip to the next frame if this one fails

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip this frame

    # Resize frame to half size for a balance between speed and recognition capability
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    frame_face_locations = face_recognition.face_locations(small_frame)
    frame_face_encodings = face_recognition.face_encodings(small_frame, frame_face_locations)

    for (top, right, bottom, left), face_encoding in zip(frame_face_locations, frame_face_encodings):
        # Scale back up face locations since the frame we detected in was scaled to 1/2 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

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
                    payload = {'rollNumber': name, 'cameraName': 'Security Camera'}
                    #response = requests.post('http://localhost:5001/report/incident', json=payload)
                    #print(f"Alert sent to server: {response.text}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
