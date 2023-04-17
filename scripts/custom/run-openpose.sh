#!/bin/bash

# Check if path is specified as an argument
if [ -z "$1" ]; then
  echo "Please specify the path to the folder containing the images."
  exit 1
fi

# Check if the path exists
if [ ! -d "$1" ]; then
  echo "The specified path does not exist or is not a directory."
  exit 1
fi

CWD=$(pwd)
cd $HOME/third_party/openpose
python main.py --image_dir $1/images
mv pose2d.npy $1/keypoints.npy
cd $CWD