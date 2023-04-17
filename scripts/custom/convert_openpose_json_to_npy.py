import os
import json
import numpy as np


def convert(JSON_FOLDER, NPY_PATH):
    # Initialize an empty list to accumulate the pose data
    pose_data_list = []

    # Loop over all JSON files in the folder
    for JSON_FILE in sorted(os.listdir(JSON_FOLDER)):
        # Check if the file is a JSON file
        if JSON_FILE.endswith('.json'):
            # Construct the full path to the JSON file
            JSON_PATH = os.path.join(JSON_FOLDER, JSON_FILE)
            
            # Load the JSON data
            with open(JSON_PATH, 'r') as f:
                data = json.load(f)
            
            # Extract the pose data from the JSON data
            pose_data = np.array(data['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
            
            # Add the pose data to the list
            pose_data_list.append(pose_data)

    # Convert the list of pose data arrays to a single NumPy array
    pose_data_array = np.stack(pose_data_list, axis=0)

    # Save the pose data as an NPY file
    np.save(NPY_PATH, pose_data_array)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert OpenPose JSON files to NPY file")
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="keypoints.npy")
    args = parser.parse_args()

    JSON_FOLDER = args.json_dir
    NPY_PATH = os.path.join(JSON_FOLDER, "..", args.output_file)

    convert(JSON_FOLDER, NPY_PATH)
