import requests
import time
import numpy as np
import cv2
import os

from mmaction.apis import inference_recognizer, init_recognizer

# Abdullah Paths (Absolute)
config_path = r'C:\Users\alpha\mmaction2\configs\recognition\tsn\tsn_r50_custom.py'
checkpoint_path = r'C:\Users\alpha\mmaction2\work_dirs\tsn_r50_custom\best_acc_top1_epoch_20.pth'
# For Smoking
#img_path = r'C:\Users\alpha\Downloads\Final to work on dataset\Ava Kinetics method\my_dataset\videos\smoking\s10.mp4'
# For Fighting
img_path = r'C:\Users\alpha\mmaction2\data\unknown_videos_for_model\fighting\fi016.mp4'
# For Normal
# img_path = r'C:\Users\alpha\mmaction2\data\unknown_videos_for_model\normal\TrumanShow_run_f_nm_np1_ba_med_21.mp4'
# Anomaly
# img_path = r'C:\Users\alpha\mmaction2\data\unknown_videos_for_model\normal\nofi022.mp4'

# Abubakar Paths (Absolute)
# config_path = r'C:\Users\abuba\mmaction2\configs\recognition\tsn\tsn_r50_custom.py'
# checkpoint_path = r'C:\Smart-Surveillance-System-SSS\MMaction2 fine tuned model\best_acc_top1_epoch_20.pth'
# For Smoking
# img_path = r'C:\Smart-Surveillance-System-SSS\MMaction2 fine tuned model\Final to work on dataset\Kinetics 400 method\my_dataset\videos\smoking\20241005180755649_E43686547_5_wh[]wh_di[E43686547]di_cn[5]cn_trans.MP4_Rendered.mp4'
# For Fighting
# img_path = r'C:\Smart-Surveillance-System-SSS\MMaction2 fine tuned model\Final to work on dataset\Kinetics 400 method\my_dataset\videos\fighting\fi016.mp4'

upload_directory = r'C:\Smart-Surveillance-System-SSS\Web App\server\uploads'
if not os.path.exists(upload_directory):
    os.makedirs(upload_directory)  # Create the directory if it doesn't exist

# Path to save the extracted image
image_save_path = os.path.join(upload_directory, 'extracted_frame.jpg')


# Function to extract the middle frame from the video and save it as an image
def extract_frame_from_video(video_path, save_path):
    cap = cv2.VideoCapture(img_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames in video: {frame_count}")

    # Calculate the middle frame, handle odd frame counts by flooring the division result
    middle_frame = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Unable to read frame {middle_frame} from video {img_path}")
    else:
        cv2.imwrite(image_save_path, frame)
        print(f"Middle frame extracted and saved to {image_save_path}")

    cap.release()


# Build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cpu")

# Inference the video
results = inference_recognizer(model, img_path)

# Label mapping
label_map = {0: 'Smoking', 1: 'Fighting', 2: 'Normal'}

# Extract the predicted scores from the ActionDataSample object
pred_scores = results.pred_score

# Find the label with the highest prediction score
max_score_index = int(np.argmax(pred_scores))  # Ensure the index is an integer
predicted_label = label_map.get(max_score_index, f'Unknown({max_score_index})')
predicted_score = pred_scores[max_score_index]

# Print the label and score
print(f'Predicted Label: {predicted_label}, Score: {predicted_score:.2f}')

# Extract the middle frame from the video
extract_frame_from_video(img_path, image_save_path)

# Check if the predicted label is not 'Normal' and handle the "Fighting" case with score less than 0.65
if predicted_label != 'Normal' and not (predicted_label == 'Fighting' and predicted_score < 0.65):
    # Server details
    server_url = 'http://localhost:5001/report/label'  # Define the correct endpoint for action label reporting
    camera_name = 'Camera 1'

    # Send the label, score, cameraName, and image to the server
    with open(image_save_path, 'rb') as img_file:
        files = {'image': img_file}
        payload = {'label': predicted_label, 'score': f'{predicted_score:.2f}', 'cameraName': camera_name}
        response = requests.post(server_url, data=payload, files=files)

    # Check response status
    if response.status_code == 200:
        print(f"Label and image sent to server successfully: {response.json()}")
    else:
        print(f"Failed to send label and image to server: {response.status_code}")
else:
    print(f"Request not sent to server for label '{predicted_label}' with score {predicted_score:.2f}")


# Print the scores with the associated labels
for label_idx, score in enumerate(pred_scores):
    label = label_map.get(label_idx, f'Unknown({label_idx})')
    print(f'{label}: {score:.2f}')