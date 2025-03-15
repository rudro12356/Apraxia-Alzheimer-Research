import project_utils as pu


if __name__ == "__main__":
    root_dir = "/Users/home/Professor Sumaiya Lab Work/Autism/Apraxia/praxis_dataset"  # Contains Patient_01/RGB, Patient_02/RGB, ...
    annotation_root = "/Users/home/Professor Sumaiya Lab Work/Autism/Apraxia/labels_time_stamps_release"  # Contains Patient_01.csv, Patient_02.csv, ...
    action_id = "P2_3"

    action_images = pu.get_images_for_action_all_patients(action_id, root_dir, annotation_root)
    df_skeleton = pu.extract_skeleton_data(action_images, action_id, annotation_root)
    # Print and save the results
    print(df_skeleton)
    df_skeleton.to_csv(f"{action_id}_skeleton.csv", index=False)