#!/bin/bash

# Enable logging
LOG_FILE="train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
set -e  # Exit immediately if a command fails

echo "Process started at: $(date)"

# Base paths
BASE_VIDEO_PATHS=("vp9/run1b_2018-05-23-16-19-17.kinect_color" 
                  "vp10/run1_2018-05-24-13-14-41.kinect_color"
                  "vp10/run2_2018-05-24-14-08-46.kinect_color"
                  "vp12/run1_2018-05-24-15-44-28.kinect_color" 
                  "vp12/run2_2018-05-24-16-21-35.kinect_color")

# Loop through noise levels
for BASE_VIDEO in "${BASE_VIDEO_PATHS[@]}"; do
    # Create output folder
    DATA_FOLDER="data/${BASE_VIDEO}"
    echo "$DATA_FOLDER"
done
