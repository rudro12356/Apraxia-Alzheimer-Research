import os
import datetime
import pandas as pd

import cv2
import mediapipe as mp

def extract_time_and_frame(filename: str):
    """
    Given a filename like:
        2015-12-07T11-22-53.150000-bgrimage.jpg

    Returns: (time_obj, frame_number)
      - time_obj: datetime.time (HH:MM:SS)
      - frame_number: int from the portion after the decimal
    """
    if not filename.endswith('-bgrimage.jpg'):
        raise ValueError(f"Filename not in expected format: {filename}")
    base = filename[:-len('-bgrimage.jpg')]  # e.g. "2015-12-07T11-22-53.150000"

    # Split on '.' -> ["2015-12-07T11-22-53", "150000"]
    if '.' not in base:
        raise ValueError(f"No '.' found in base portion: {base}")
    datetime_part, frame_str = base.split('.', 1)

    # datetime_part -> e.g. "2015-12-07T11-22-53"
    # We'll split "date" from "time" by 'T'
    if 'T' not in datetime_part:
        raise ValueError(f"No 'T' found in datetime portion: {datetime_part}")
    _, time_str = datetime_part.split('T', 1)

    # time_str might look like "11-22-53"
    parsed_time = datetime.datetime.strptime(time_str, "%H-%M-%S").time()

    frame_number = int(frame_str)
    return parsed_time, frame_number


def parse_annotation_ts(ts_str: str):
    """
    Given an annotation time string like "11-22-38.817000",
    return (time_obj, frame_number).

      - time_obj: datetime.time (HH:MM:SS)
      - frame_number: int from everything after the '.'
    """
    if '.' not in ts_str:
        raise ValueError(f"Annotation time '{ts_str}' missing '.' for frames.")

    time_part, frame_part = ts_str.split('.', 1)
    # e.g. time_part = "11-22-38", frame_part = "817000"
    dt = datetime.datetime.strptime(time_part, "%H-%M-%S")
    return dt.time(), int(frame_part)


def is_in_range(file_time, file_frame, start_time, start_frame, end_time, end_frame):
    """
    Compare (file_time, file_frame) to [start_time, start_frame, end_time, end_frame].
    Return True if the file is >= start and <= end.
    """
    ft_tuple = (file_time.hour, file_time.minute, file_time.second, file_frame)
    st_tuple = (start_time.hour, start_time.minute, start_time.second, start_frame)
    et_tuple = (end_time.hour, end_time.minute, end_time.second, end_frame)
    return st_tuple <= ft_tuple <= et_tuple


def get_images_for_action(patient_id, action_id, root_dir, annotation_root):
    """
    Returns a list of image filenames for the specified `action_id`
    in a given `patient_id`.

    Parameters:
    -----------
    patient_id  : e.g. "Patient_01"
    action_id   : e.g. "P2_3"
    root_dir    : top-level directory for patient data (contains subfolders "Patient_01/RGB", etc.)
    annotation_root : directory containing annotation files named "Patient_01.csv", "Patient_02.csv", etc.
    """
    annotation_file = os.path.join(annotation_root, f"{patient_id}.csv")
    if not os.path.isfile(annotation_file):
        # Optionally, print or raise an error if annotation does not exist for a given patient
        print(f"Warning: No annotation file for {patient_id} at {annotation_file}")
        return []

    image_dir = os.path.join(root_dir, patient_id, "RGB")
    if not os.path.isdir(image_dir):
        print(f"Warning: No RGB folder for {patient_id} at {image_dir}")
        return []

    # Read annotation
    df = pd.read_csv(annotation_file, header=None)
    df.columns = ['action_id', 'start_ts', 'end_ts', 'status', 'state']

    # Find row with matching action_id
    matching_rows = df[df['action_id'] == action_id]
    if matching_rows.empty:
        print(f"Warning: No Matching image for action ID {action_id} for {patient_id} at {image_dir}")
        return []

    # For simplicity, assume one row per action
    row = matching_rows.iloc[0]
    start_ts = row['start_ts']  # e.g. "11-22-53.150000"
    end_ts = row['end_ts']  # e.g. "11-23-05.750000"

    # Parse start/end
    start_time, start_frame = parse_annotation_ts(start_ts)
    end_time, end_frame = parse_annotation_ts(end_ts)

    matched_files = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith('.jpg'):
            try:
                file_time, file_frame = extract_time_and_frame(fname)
            except ValueError:
                continue
            if is_in_range(file_time, file_frame, start_time, start_frame, end_time, end_frame):
                matched_files.append(os.path.join(image_dir, fname))

    return matched_files


def get_images_for_action_all_patients(action_id, root_dir, annotation_root):
    """
    Iterates over all "Patient_*" subfolders in `root_dir`.
    For each patient, fetches the images for the given action.
    Returns a dictionary { patient_id: [list_of_images] }.
    """
    # Look for subfolders that start with "Patient_"
    patient_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("Patient_")
    ]

    results = {}
    for patient_id in patient_dirs:
        images = get_images_for_action(patient_id, action_id, root_dir, annotation_root)
        results[patient_id] = images
    return results


# def extract_skeleton_data(action_images, action, output_dir="output_videos"):
#     """
#     Extracts skeleton data for a given dictionary of patient images using MediaPipe's Pose model.
#
#     Parameters:
#     -----------
#     action_images: dict
#         Dictionary with 'patient_id' as key and a list of image file paths as value.
#         e.g. {
#                "patient_001": ["path/to/image1.jpg", "path/to/image2.jpg"],
#                "patient_002": ["path/to/image3.jpg"]
#              }
#     action: str
#         String label or identifier for the action being performed.
#
#     Returns:
#     --------
#     pandas.DataFrame
#         DataFrame with columns:
#           [patient_id, action, joint0_x, joint0_y, joint7_x, joint7_y, ..., joint24_x, joint24_y].
#     """
#     mp_pose = mp.solutions.pose
#     mp_drawing = mp.solutions.drawing_utils
#     pose = mp_pose.Pose(
#         static_image_mode=True,
#         enable_segmentation=False,
#         min_detection_confidence=0.5
#     )
#
#     # Landmarks of interest in the Pose model
#     landmarks_of_interest = [0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#
#     # Prepare list to build the DataFrame
#     data_rows = []
#
#     # Iterate over each patient and their corresponding images
#     for patient_id, image_paths in action_images.items():
#         if not image_paths:
#             print(f"No images found for patient {patient_id}")
#             continue  # Skip if no images
#         # --- 1) Prepare to write a video for each patient ---
#         # Read the first image to get frame size
#         first_img = cv2.imread(image_paths[0])
#         if first_img is None:
#             # Skip if we can't read the first image
#             continue
#
#         height, width, _ = first_img.shape
#         # Define codec and create a VideoWriter
#         video_name = f"{patient_id}_{action}.mp4"
#         video_path = os.path.join(output_dir, video_name)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
#
#         for img_path in image_paths:
#             image = cv2.imread(img_path)
#             if image is None:
#                 # Skip if image loading fails
#                 continue
#
#             # Convert to RGB for MediaPipe
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             # Run pose estimation
#             results = pose.process(image_rgb)
#
#             # Skip if no pose landmarks are detected
#             if not results.pose_landmarks:
#                 continue
#
#             # Collect all required landmarks
#             row_data = [patient_id, action]
#             for lm_idx in landmarks_of_interest:
#                 landmark = results.pose_landmarks.landmark[lm_idx]
#                 row_data.append([round(landmark.x, 5),round(landmark.y, 5)])
#
#             data_rows.append(row_data)
#             # Optionally draw the pose on the image
#             # mp_drawing.draw_landmarks(
#             #     image,
#             #     results.pose_landmarks,
#             #     mp_pose.POSE_CONNECTIONS
#             # )
#             # Write the (optionally annotated) frame into the video
#             out.write(image)
#
#         # Close out the VideoWriter for this patient
#         out.release()
#
#     # Build the column headers
#     columns = ["patient_id", "action"]
#     for lm_idx in landmarks_of_interest:
#         columns.append(f"joint_{lm_idx}")
#
#     # Create a DataFrame with the collected rows
#     df = pd.DataFrame(data_rows, columns=columns)
#     return df
def extract_skeleton_data(action_images, action, annotation_root, output_dir="output_videos"):
    """
    Extracts skeleton data for a given dictionary of patient images using MediaPipe's Pose model.

    Parameters:
    -----------
    action_images: dict
        Dictionary with 'patient_id' as key and a list of image file paths as value.
    action: str
        String label or identifier for the action being performed.
    annotation_root: str
        Directory containing annotation files named "Patient_01.csv", "Patient_02.csv", etc.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
          [patient_id, action, start_frame, end_frame, joint0_x, joint0_y, ..., joint24_x, joint24_y].
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=True,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    landmarks_of_interest = [0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    data_rows = []

    for patient_id, image_paths in action_images.items():
        if not image_paths:
            print(f"No images found for patient {patient_id}")
            continue

        # Read corresponding annotation file
        annotation_file = os.path.join(annotation_root, f"{patient_id}.csv")
        if not os.path.isfile(annotation_file):
            print(f"Warning: No annotation file found for {patient_id}")
            continue

        # Read annotation data
        df_annotations = pd.read_csv(annotation_file, header=None)
        df_annotations.columns = ['action_id', 'start_ts', 'end_ts', 'status', 'state']

        # Get start and end timestamps for the action
        action_row = df_annotations[df_annotations['action_id'] == action]
        if action_row.empty:
            print(f"Warning: No annotation data for {action} in {patient_id}")
            continue

        start_time, start_frame = parse_annotation_ts(action_row.iloc[0]['start_ts'])
        end_time, end_frame = parse_annotation_ts(action_row.iloc[0]['end_ts'])

        # Process images
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue  # Skip unreadable images

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                continue  # Skip if no pose detected

            row_data = [patient_id, action, start_frame, end_frame]
            for lm_idx in landmarks_of_interest:
                landmark = results.pose_landmarks.landmark[lm_idx]
                row_data.append(round(landmark.x, 5))
                row_data.append(round(landmark.y, 5))

            data_rows.append(row_data)

    # Build DataFrame
    columns = ["patient_id", "action", "start_frame", "end_frame"]
    for lm_idx in landmarks_of_interest:
        columns.append(f"joint_{lm_idx}_x")
        columns.append(f"joint_{lm_idx}_y")

    df = pd.DataFrame(data_rows, columns=columns)
    return df
