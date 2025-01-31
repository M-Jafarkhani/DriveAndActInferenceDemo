#!/bin/bash

# Enable logging
LOG_FILE="train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
set -e  # Exit immediately if a command fails

echo "Process started at: $(date)"

# Base video paths
BASE_VIDEO_PATHS=(
    "vp9/run1b_2018-05-23-16-19-17.kinect_color"
    "vp10/run1_2018-05-24-13-14-41.kinect_color"
    "vp10/run2_2018-05-24-14-08-46.kinect_color"
    "vp12/run1_2018-05-24-15-44-28.kinect_color"
    "vp12/run2_2018-05-24-16-21-35.kinect_color"
)

# Model and annotation paths
MODEL_PATH="./demo_models/best_model.pth"
ANNOTATION_PATH="./demo_models/annotation_converter.pkl"

# Loop through video files
for BASE_VIDEO in "${BASE_VIDEO_PATHS[@]}"; do
    DATA_FOLDER="./data/$(dirname "$BASE_VIDEO")"
    VIDEO_PATH="$DATA_FOLDER/$(basename "$BASE_VIDEO")"
    LOG_FILE="$DATA_FOLDER/predictions.log"
    METRICS_OUTPUT="$DATA_FOLDER/metrics_results.txt"
    
    echo "Processing video: $VIDEO_PATH"
    
    # Run sliding window inference
    python3 SlidingWindow.py --video_path "$VIDEO_PATH" --model_path "$MODEL_PATH" --annotation_path "$ANNOTATION_PATH" --log_file "$LOG_FILE"
    
    # Run metrics evaluation
    python3 Metrics.py --ground_truth "$DATA_FOLDER/midlevel.chunks_90.split_0.test.csv" --prediction_log "$LOG_FILE" --file_id "$BASE_VIDEO" > "$METRICS_OUTPUT"
    
    echo "Metrics saved to: $METRICS_OUTPUT"
done