import face_recognition
import os
import numpy as np
import pickle


def generate_face_encodings(photo_directory):
    """
    Generate face encodings from photos in the specified directory
    and save them to a pickle file.

    Args:
        photo_directory (str): Path to directory containing student photos
    """
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
                print(f"Successfully encoded {filename}")
            except IndexError:
                print(f"No face found in {filename}, skipping...")
                continue

    # Save encodings to file
    encoding_data = {
        'encodings': student_encodings,
        'names': student_names
    }

    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump(encoding_data, f)

    print(f"Successfully saved encodings for {len(student_names)} students")


if __name__ == "__main__":
    # Base directory of the script
    base_directory = os.path.dirname(os.path.abspath(__file__))
    photo_directory = os.path.join(base_directory, 'Students')

    generate_face_encodings(photo_directory)