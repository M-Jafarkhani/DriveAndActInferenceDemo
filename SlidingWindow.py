import logging
import cv2
from InferenceOnSingleVideoMini import load_model, load_video_segment, run_inference_on_video_chunk

# Configure logging
def configure_logger(log_file):
    """
    Configures the logger to save predictions and details to a log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Save logs to file
            logging.StreamHandler()  # Also print to console
        ]
    )

def log_prediction(frame_idx, top1_class, confidence):
    """
    Logs the prediction for a given frame.

    Args:
        frame_idx (int): Index of the frame.
        top1_class (str): Predicted activity.
        confidence (float): Confidence of the prediction.
    """
    logging.info(f"Frame {frame_idx}: Predicted Activity: {top1_class}, Confidence: {confidence:.2f}%")

def sliding_window_inference(video_path, model_path, annotation_path, log_file, window_size=16, stride=1, resize=None, cuda_active=True):
    # Configure logger
    configure_logger(log_file)
    model, annotation_converter = load_model(model_path, annotation_path, cuda_active)
    frames = load_video_segment(video_path, start_frame=0, n_frames=0)  # Load all frames
    num_frames = len(frames)
    
    if resize:
        frames = [cv2.resize(frame, resize) for frame in frames]

    # Sliding window inference
    for start in range(0, num_frames - window_size + 1, stride):
        # Define window
        end = start + window_size
        window_frames = frames[start:end]

        # Run inference on the current window
        top1_class, top1_class_conf, _ = run_inference_on_video_chunk(window_frames, model, annotation_converter, cuda_active)

        # Assign the prediction to the first frame of the window
        for i in range(start, min(start + stride, num_frames)):
            log_prediction(i, top1_class, top1_class_conf * 100)

    logging.info("Inference complete. Predictions logged successfully.")

if __name__ == "__main__":
    # Define paths
    video_path = "./run1b_2018-05-29-14-02-47.kinect_color.mp4"
    model_path = "./demo_models/best_model.pth"
    annotation_path = "./demo_models/annotation_converter.pkl"
    log_file = "./predictions.log"

    # Perform sliding window inference and log predictions
    sliding_window_inference(
        video_path=video_path,
        model_path=model_path,
        annotation_path=annotation_path,
        log_file=log_file,
        window_size=16,
        stride=1,
        resize=(224, 224),
        cuda_active=True
    )
