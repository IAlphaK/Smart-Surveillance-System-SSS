import base64

import cv2
import mmcv
import mmengine
import numpy as np
import torch
import face_recognition
import pickle
import os
import requests
import time
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmaction.apis import detection_inference
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.utils import get_str_type
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any


@dataclass
class FrameData:
    # Contains information for a single frame where a person is detected
    frame_id: int
    bbox: np.ndarray
    frame: np.ndarray

@dataclass
class PersonAction:
    # Represents the data for a single detected action sequence for a person
    person_id: int
    action: str
    start_frame: int
    frames: List[FrameData]
    last_processed: int = 0

    def add_frame(self, frame_id: int, bbox: np.ndarray, frame: np.ndarray):
        # Add a frame to the action sequence
        self.frames.append(FrameData(frame_id=frame_id, bbox=bbox, frame=frame))


class IntegratedDetector:
    def __init__(self, config_file, checkpoint_file, label_map_file, encoding_file='face_encodings.pkl',
                 device='cuda:0'):
        # Initialize the integrated detector that combines:
        # 1. Action detection using MMAction2
        # 2. Face recognition for identified subjects
        # 3. Tracking of persons across frames to maintain consistent IDs and detect ongoing actions

        # Core parameters and model setup
        self.device = device
        # Action-specific parameters for thresholding and timing
        self.action_params = {
            'smoking': {
                'confidence_threshold': 0.97,
                'temporal_window_seconds': 0.8,
                'min_confidence_seconds': 0.6,
                'initial_threshold': 0.95
            },
            'fighting': {
                'confidence_threshold': 0.92,
                'temporal_window_seconds': 1.0,
                'min_confidence_seconds': 0.7,
                'initial_threshold': 0.90
            }
        }

        # Dictionary to keep track of persons and their bounding boxes over time
        self.person_trackers = {}

        # Cleanup parameters for stale person trackers
        self.tracker_cleanup_interval = 30  # frames
        self.last_cleanup = 0

        # Frame and action tracking
        self.current_frame_id = 0
        self.fps = 30
        self.active_actions: Dict[Tuple[int, str], PersonAction] = {}
        self.processed_actions: set = set()

        # Model setups
        self._setup_action_detection(config_file, checkpoint_file, label_map_file)
        self._setup_face_recognition(encoding_file)

        # Directory for saving cropped images during processing
        os.makedirs('cropped_frames', exist_ok=True)

    def _create_or_update_action(self, action_key: Tuple[int, str], person_id: int,
                                 action: str, frame_data: FrameData) -> None:
        # Create a new action instance or update existing one for a given person-action combination
        if action_key not in self.active_actions:
            self.active_actions[action_key] = PersonAction(
                person_id=person_id,
                action=action.lower(),
                start_frame=self.current_frame_id,
                frames=[frame_data]
            )
        else:
            self.active_actions[action_key].frames.append(frame_data)

    def _setup_action_detection(self, config_file, checkpoint_file, label_map_file):
        """
        Sets up the action detection model from config and checkpoint files.
        It also handles model normalization, label maps, and related parameters.
        """
        self.config = mmengine.Config.fromfile(config_file)
        self.config.model.backbone.pretrained = None
        try:
            # Some configs may allow custom test_cfg handling
            self.config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
        except KeyError:
            pass

        # Build and load the action detection model
        self.model = MODELS.build(self.config.model)
        load_checkpoint(self.model, checkpoint_file, map_location='cpu')
        self.model.to(self.device)
        self.model.eval()

        # Load label map for interpretation of model outputs
        self.label_map = self._load_label_map(label_map_file)
        # Set detection-related parameters from the configuration
        self._setup_detection_params()

    def _cleanup_person_trackers(self):
        """
        Cleans up person trackers that have not been updated for a certain number of frames.
        Helps remove stale or no-longer-visible person entries.
        """
        current_frame = self.current_frame_id
        stale_threshold = 60  # frames

        for person_id in list(self.person_trackers.keys()):
            track = self.person_trackers[person_id]
            if not track:
                continue

            last_frame = self.current_frame_id - stale_threshold
            if isinstance(track[-1], dict) and 'frame_id' in track[-1]:
                last_frame = track[-1]['frame_id']

            # Remove trackers whose last update is too old
            if self.current_frame_id - last_frame > stale_threshold:
                del self.person_trackers[person_id]

    def _setup_detection_params(self):
        """
        Sets parameters needed for detection like image normalization, temporal sampling, etc.
        """
        self.short_side = 256
        self.clip_buffer = []
        val_pipeline = self.config.val_pipeline
        # Extract sampling parameters from the pipeline (e.g., clip length and frame interval)
        sampler = [x for x in val_pipeline if get_str_type(x['type']) == 'SampleAVAFrames'][0]
        self.clip_len = sampler['clip_len']
        self.frame_interval = sampler['frame_interval']

        self.det_score_thr = 0.9
        self.det_cat_id = 0
        self.det_config = r'C:\Users\alpha\mmaction2\demo\demo_configs\faster-rcnn_r50_fpn_2x_coco_infer.py'
        # Action detection model's human detector checkpoint
        self.det_checkpoint = ('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                               'faster_rcnn_r50_fpn_2x_coco/'
                               'faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')

        self.img_norm_cfg = dict(
            mean=np.array(self.config.model.data_preprocessor.mean),
            std=np.array(self.config.model.data_preprocessor.std),
            to_rgb=False
        )

    def _setup_face_recognition(self, encoding_file):
        """
        Loads known face encodings and associated names for face recognition.
        Allows identification of known individuals performing actions.
        """
        with open(encoding_file, 'rb') as f:
            encoding_data = pickle.load(f)
        self.known_face_encodings = encoding_data['encodings']
        self.known_face_names = encoding_data['names']

    def _load_label_map(self, label_map_file):
        """
        Loads the label map (class indices to action names) from a file.
        """
        with open(label_map_file, 'r') as f:
            lines = f.readlines()
        lines = [x.strip().split(': ') for x in lines]
        label_map = {int(x[0]): x[1] for x in lines}
        return label_map

    def process_frame(self, frame):
        """
        Processes a single frame:
        1. Increments frame count
        2. Runs action detection
        3. Updates active actions and checks if any action sequence should be processed
        """
        self.current_frame_id += 1

        # Detect actions and get annotated frame + detections
        processed_frame, detections = self._detect_actions(frame)

        # Update action sequences based on detections
        for bbox, actions, person_id in detections:
            frame_data = FrameData(
                frame_id=self.current_frame_id,
                bbox=bbox,
                frame=frame.copy()
            )

            # For each detected action of a person, update or create an action entry
            for action, score in actions:
                action_key = (person_id, action.lower())

                # If this action was processed recently, skip to avoid duplicates
                if action_key in self.processed_actions:
                    last_processed = self.active_actions.get(action_key, PersonAction(0, "", 0, [])).last_processed
                    if self.current_frame_id - last_processed < 150:
                        continue

                self._create_or_update_action(action_key, person_id, action, frame_data)

                # Check if conditions are met to process the action sequence
                self._check_and_process_action(action_key)

        return processed_frame

    def _detect_actions(self, frame):
        """
        Detects persons and their actions in the given frame using:
        1. Human detection via a Faster R-CNN model
        2. Action classification using MMAction2 model on buffered frames

        Returns:
            processed_frame: Frame annotated with detections
            detections: List of (bbox, actions, person_id)
        """
        try:
            h, w, _ = frame.shape

            # Rescale frame to short_side, maintaining aspect ratio
            new_w, new_h = mmcv.rescale_size((w, h), (self.short_side, np.Inf))
            frame_resized = mmcv.imresize(frame, (new_w, new_h))
            w_ratio, h_ratio = new_w / w, new_h / h

            # Maintain a buffer of frames for temporal action recognition
            self.clip_buffer.append(frame_resized)
            if len(self.clip_buffer) > self.clip_len * self.frame_interval:
                self.clip_buffer.pop(0)

            # If we don't have enough frames to form a clip, just return
            if len(self.clip_buffer) < self.clip_len * self.frame_interval:
                return frame, []

            frame_for_detection = frame_resized.copy()

            # Human detection step (independent of action classification)
            print("Performing Human Detection for each frame")
            human_detections, _ = detection_inference(
                self.det_config,
                self.det_checkpoint,
                [frame_for_detection],
                self.det_score_thr,
                self.det_cat_id,
                self.device
            )

            # If no humans detected, no actions
            if not human_detections or len(human_detections[0]) == 0:
                return frame, []

            det = human_detections[0]
            human_detections_tensor = torch.from_numpy(det[:, :4]).to(self.device)

            # Prepare a clip of frames for action recognition
            clip_frames = [
                self.clip_buffer[i].astype(np.float32)
                for i in range(0, len(self.clip_buffer), self.frame_interval)
            ][-self.clip_len:]

            # Normalize frames (as expected by the model)
            normalized_frames = []
            for img in clip_frames:
                img_norm = img.copy()
                mmcv.imnormalize_(
                    img_norm,
                    self.img_norm_cfg['mean'],
                    self.img_norm_cfg['std'],
                    self.img_norm_cfg['to_rgb']
                )
                normalized_frames.append(img_norm)

            # Prepare tensor input for the action model
            input_array = np.stack(normalized_frames).transpose((3, 0, 1, 2))[np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(self.device)

            # Prepare data sample with bounding boxes for persons
            datasample = ActionDataSample()
            datasample.proposals = InstanceData(bboxes=human_detections_tensor)
            datasample.set_metainfo(dict(img_shape=(new_h, new_w)))

            # Run action recognition model
            with torch.no_grad():
                result = self.model(input_tensor, [datasample], mode='predict')
                scores = result[0].pred_instances.scores

            # Periodically cleanup old trackers
            if self.current_frame_id - self.last_cleanup > self.tracker_cleanup_interval:
                self._cleanup_person_trackers()
                self.last_cleanup = self.current_frame_id

            # Visualize results and prepare return data
            detections = []
            processed_frame = frame.copy()

            for i, bbox in enumerate(human_detections_tensor):
                bbox_orig = bbox.cpu().numpy()
                bbox_orig = bbox_orig.copy()
                # Convert back to original frame coordinates
                bbox_orig[0::2] = bbox_orig[0::2] * w / new_w
                bbox_orig[1::2] = bbox_orig[1::2] * h / new_h

                # Initially draw a box in blue for a detected person
                x1, y1, x2, y2 = map(int, bbox_orig)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                actions = []
                # Evaluate action scores and pick those above initial thresholds
                for j in range(scores.shape[1]):
                    if j not in self.label_map:
                        continue

                    action_name = self.label_map[j]
                    params = self.action_params.get(action_name.lower())

                    if params and scores[i, j] > params['initial_threshold']:
                        actions.append((action_name, scores[i, j].item()))

                if actions:
                    # If an action is detected, draw a green box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Get consistent ID for this person using tracking
                    person_id = self._get_person_id(bbox_orig)
                    y_offset = y1
                    for action_name, score in actions:
                        # Overlay text with person ID and action name
                        text = f"ID{person_id} {action_name}: {score:.2f}"
                        y_offset += 20
                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(
                            processed_frame,
                            (x1, y_offset - text_h),
                            (x1 + text_w, y_offset + 5),
                            (0, 0, 0),
                            -1
                        )
                        cv2.putText(
                            processed_frame,
                            text,
                            (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )
                    detections.append((bbox_orig, actions, person_id))

            return processed_frame, detections

        except Exception as e:
            # If any error occurs during detection, return original frame
            print(f"Error in _detect_actions: {str(e)}")
            return frame, []

    def _get_person_id(self, bbox):
        """
        Assigns or retrieves a consistent person ID for a detected bounding box.
        The method attempts to match the current bounding box to previously seen persons
        by measuring the distance between the center points of new and old detections.
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        min_distance = float('inf')
        best_person_id = None

        # Compare current detection with existing trackers to find the closest match
        for person_id, track in self.person_trackers.items():
            if not track:
                continue

            last_bbox = track[-1]
            if isinstance(last_bbox, dict) and 'bbox' in last_bbox:
                last_bbox = last_bbox['bbox']
            elif not isinstance(last_bbox, np.ndarray):
                continue

            last_center_x = (last_bbox[0] + last_bbox[2]) / 2
            last_center_y = (last_bbox[1] + last_bbox[3]) / 2

            distance = np.sqrt(
                (center_x - last_center_x) ** 2 +
                (center_y - last_center_y) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                best_person_id = person_id

        # If found a close match, update that person's track
        if best_person_id is not None and min_distance < 150:
            self.person_trackers[best_person_id].append(bbox)
            # Limit track history length
            if len(self.person_trackers[best_person_id]) > 30:
                self.person_trackers[best_person_id].pop(0)
            return best_person_id

        # If no match found, assign a new person ID
        new_id = len(self.person_trackers)
        self.person_trackers[new_id] = [bbox]
        return new_id

    def _check_and_process_action(self, action_key: Tuple[int, str]) -> None:
        """
        Checks if an ongoing action sequence has enough frames or if time threshold passed
        to perform the processing step (face recognition, reporting).
        """
        action_data = self.active_actions[action_key]
        frames_count = len(action_data.frames)

        if frames_count < 4:
            # Not enough frames collected for processing
            return

        # Decide if we should process now based on length of sequence or gap
        should_process = (
            frames_count >= 35 or  # if sequence is long enough
            (self.current_frame_id - action_data.frames[-1].frame_id > 15)  # or if recent frames are no longer incoming
        )

        print(
            f"Checking action for person {action_data.person_id}: frames={frames_count}, should_process={should_process}")

        if not should_process:
            return

        # At this point, we have a sufficient sequence to process
        print(f"Processing action sequence for person {action_data.person_id}")
        try:
            self._process_action_sequence(action_key)
        except Exception as e:
            print(f"Error processing action sequence: {str(e)}")
            return

        # Mark this action as processed to avoid redundant processing
        self.processed_actions.add(action_key)
        action_data.last_processed = self.current_frame_id

    def _process_action_sequence(self, action_key: Tuple[int, str]) -> None:
        """
        Processes a completed action sequence:
        1. Extracts cropped frames
        2. Performs face recognition on them
        3. Determines the most likely person involved
        4. Sends results to the server endpoint
        """
        action_data = self.active_actions[action_key]

        # Attempt face recognition on frames associated with the action
        face_detections = []
        for idx, frame_data in enumerate(action_data.frames):
            try:
                # Crop the region of interest (person bbox) from the original frame
                x1, y1, x2, y2 = map(int, frame_data.bbox)
                cropped_frame = frame_data.frame[y1:y2, x1:x2]

                # Save for debugging/record keeping
                frame_filename = f'cropped_frames/person_{action_data.person_id}_{frame_data.frame_id}.jpg'
                cv2.imwrite(frame_filename, cropped_frame)

                # Convert to RGB for face_recognition library
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                # Extract face encodings
                face_encodings = face_recognition.face_encodings(rgb_frame, model="small")
                if len(face_encodings) > 0:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0],
                                                             tolerance=0.6)
                    if True in matches:
                        match_idx = matches.index(True)
                        face_detections.append(self.known_face_names[match_idx])
                    else:
                        face_detections.append("Unknown")
                else:
                    # No face found in this frame
                    print(f"No face encodings found in frame {frame_data.frame_id}")
                    face_detections.append("Unknown")

            except Exception as e:
                # If error occurs while processing a frame, default to "Unknown"
                print(f"Error processing frame {frame_data.frame_id}: {str(e)}")
                face_detections.append("Unknown")

        # If we got any face detections, handle them
        if face_detections:
            print(f"Total faces processed: {len(face_detections)}")
            self._handle_detection_results(action_data, face_detections)
        else:
            print("No faces were detected in the sequence")

        # Remove the processed action from active actions
        self.active_actions.pop(action_key)

    def _handle_detection_results(self, action_data: PersonAction, face_detections: List[str]) -> None:
        """
        Determines the most likely identified person from face detections and sends results to a server endpoint.
        If too many frames are unknown, defaults to 'Unknown'.
        """
        total_frames = len(face_detections)
        if total_frames == 0:
            return

        detection_counts = Counter(face_detections)

        # Print detection summaries
        print(f"\nFace Detection Results for Person {action_data.person_id} ({action_data.action}):")
        for name, count in detection_counts.items():
            print(f"{name}: {count}")

        # Determine the final identity
        unknown_ratio = detection_counts.get("Unknown", 0) / total_frames
        if unknown_ratio >= 0.55:
            detected_person = "Unknown"
        else:
            detection_counts.pop("Unknown", None)
            detected_person = detection_counts.most_common(1)[0][0] if detection_counts else "Unknown"

        # Use a middle frame as representative
        middle_idx = len(action_data.frames) // 2
        middle_frame = action_data.frames[middle_idx]

        # Prepare a representative image for reporting
        x1, y1, x2, y2 = map(int, middle_frame.bbox)
        person_image = middle_frame.frame[y1:y2, x1:x2]

        temp_image_path = f'temp_detected_person_{action_data.person_id}.jpg'
        cv2.imwrite(temp_image_path, person_image)

        camera_name = 'main office'

        # Prepare the data to send to the server
        payload = {
            'personName': detected_person,
            'label': action_data.action,
            'cameraName': camera_name,
            'score': str(len([x for x in face_detections if x == detected_person]) / total_frames)
        }

        # Post results to server endpoint
        server_url = 'http://localhost:5001/report/label'
        try:
            with open(temp_image_path, 'rb') as img_file:
                files = {'image': ('detection.jpg', img_file, 'image/jpeg')}
                response = requests.post(server_url, data=payload, files=files)
                response.raise_for_status()
                print(f"Payload sent successfully. Server response: {response.text}")
        except requests.RequestException as e:
            print(f"Failed to send payload to server: {e}")
        finally:
            # Clean up temporary image
            try:
                os.remove(temp_image_path)
            except OSError as e:
                print(f"Error removing temporary file: {e}")

        # Also save a copy of the representative frame locally
        frame_filename = f'detected_frames/person_{action_data.person_id}_{middle_frame.frame_id}.jpg'
        os.makedirs('detected_frames', exist_ok=True)
        cv2.imwrite(frame_filename, person_image)

    def process_video(self, video_path):
        """
        Processes a given video file frame-by-frame:
        1. Extracts frames
        2. Detects actions and persons
        3. Handles result display and cleanup
        4. On completion or interruption, processes any leftover action sequences
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Processing video at {self.fps} FPS. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # If no more frames, process any partially completed action sequences
                    for action_key in list(self.active_actions.keys()):
                        action_data = self.active_actions[action_key]
                        # Only process if we have at least a few frames
                        if len(action_data.frames) >= 4:
                            print(
                                f"Processing remaining action for person {action_data.person_id} with {len(action_data.frames)} frames")
                            self._process_action_sequence(action_key)
                    break

                processed_frame = self.process_frame(frame)
                # Display results
                cv2.imshow('Action Detection', processed_frame)

                delay = int(1000 / self.fps)
                # Allow user to quit
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

                # A short sleep to reduce CPU load (optional)
                time.sleep(0.001)

        finally:
            # Cleanup video capture and windows
            cap.release()
            cv2.destroyAllWindows()

def main():
    # Entry point for this script - sets up the detector and processes the video.
    CONFIG_FILE = r'C:\Users\alpha\mmaction2\configs\detection\slowfast\hpc.py'
    CHECKPOINT_FILE = 'result.pth'
    LABEL_MAP_FILE = 'labelmap.txt'
    INPUT_VIDEO = 'test/s13.mp4'
    ENCODING_FILE = 'face_encodings.pkl'

    detector = IntegratedDetector(
        config_file=CONFIG_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        label_map_file=LABEL_MAP_FILE,
        encoding_file=ENCODING_FILE
    )

    detector.process_video(INPUT_VIDEO)


if __name__ == '__main__':
    main()
