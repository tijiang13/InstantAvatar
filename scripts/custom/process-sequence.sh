#!/bin/bash

# Check if path is specified as an argument
if [ -z "$1" ]; then
  echo "Please specify the path to the folder containing the data."
  exit 1
fi

# Check if the path exists
if [ ! -d "$1" ]; then
  echo "The specified path does not exist or is not a directory."
  exit 1
fi

# Check if gender is specified as an argument
if [ -z "$1" ]; then
  echo "Please specify the path to the folder containing the data."
  exit 1
fi

path=$(readlink -f "$1")

# Run OpenPose
if [ ! -f "$path/keypoints.npy" ]; then
  echo "Running OpenPose in $path/images"
  bash scripts/custom/run-openpose-bin.sh $path/images
fi

if [ ! -d "$path/masks" ]; then
  echo "Running mask in $path"
  python scripts/custom/run-sam.py --data_dir $path
  # python scripts/custom/run-rvm.py --data_dir $path
  python scripts/custom/extract-largest-connected-components.py --data_dir $path
fi

if [ ! -f "$path/poses.npz" ]; then
  python scripts/custom/run-romp.py --data_dir $path
fi

if [ ! -f "$path/poses_optimized.npz" ]; then
  echo "Refining SMPL..."
  python scripts/custom/refine-smpl.py --data_dir $path --gender $2 # --silhouette
fi

if [ ! -f "$path/output.mp4" ]; then
  python scripts/visualize-SMPL.py --path $path --gender $2 --pose $path/pose_optimized.npz --headless --fps 1
fi