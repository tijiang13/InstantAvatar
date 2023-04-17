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

# Set the path to your OpenPose installation
OPENPOSE_PATH="$HOME/third_party/openpose-cpp/"

# Set the path to the absoulte folder containing the images
IMAGE_FOLDER=$1
IMAGE_FOLDER=$(realpath $IMAGE_FOLDER)

CURRENT_DIR=$(pwd)
cd $OPENPOSE_PATH
./build/examples/openpose/openpose.bin \
  --image_dir $IMAGE_FOLDER\
  --display 0 \
  --write_json $IMAGE_FOLDER/../openpose_json/ \
  --write_images $IMAGE_FOLDER/../openpose_output/ \
  --write_images_format jpg \
  --render_pose 1 \
  --render_threshold 0.5 \
  --number_people_max 1 \
  --model_pose BODY_25
cd $CURRENT_DIR

python ./scripts/custom/convert_openpose_json_to_npy.py \
  --json_dir $IMAGE_FOLDER/../openpose_json/