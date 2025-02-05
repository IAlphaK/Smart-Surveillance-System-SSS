import face_recognition
import cv2
import os
import numpy as np
import requests
import time
import pickle


class VideoFaceProcessor:
    def __init__(self, encoding_file='face_encodings.pkl'):
        # Load the pre-generated encodings
        with open(encoding_file, 'rb') as f:
            encoding_data = pickle.load(f)

        self.student_encodings = encoding_data['encodings']
        self.student_names = encoding_data['names']

        self.last_detection_time = 0
        self.alert_interval = 30  # seconds
        self.frame_skip = 2  # Process every 2nd frame

    def process_video(self, video_file_path):
        video_capture = cv2.VideoCapture(video_file_path)
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Frame grab failed, skipping...")
                continue

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            # Process the frame
            self._process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Detect faces and get encodings
        frame_face_locations = face_recognition.face_locations(small_frame)
        frame_face_encodings = face_recognition.face_encodings(small_frame, frame_face_locations)

        for (top, right, bottom, left), face_encoding in zip(frame_face_locations, frame_face_encodings):
            # Scale back up face locations
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Find matches
            matches = face_recognition.compare_faces(self.student_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                face_distances = face_recognition.face_distance(self.student_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.student_names[best_match_index]
                    self._handle_detection(name)

            # Draw the results
            self._draw_result(frame, name, left, top, right, bottom)

        cv2.imshow('Video', frame)

    def _handle_detection(self, name):
        current_time = time.time()
        if (current_time - self.last_detection_time) > self.alert_interval:
            self.last_detection_time = current_time
            payload = {'rollNumber': name, 'cameraName': 'Security Camera'}
            # Uncomment to enable API calls
            # response = requests.post('http://localhost:5001/report/incident', json=payload)
            # print(f"Alert sent to server: {response.text}")
            print(f"Detection: {name}")

    def _draw_result(self, frame, name, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)


if __name__ == "__main__":
    video_file_path = r'C:\Users\alpha\Downloads\Working Directory\Extras\smoking\s13.mp4'
    processor = VideoFaceProcessor('face_encodings.pkl')
    processor.process_video(video_file_path)