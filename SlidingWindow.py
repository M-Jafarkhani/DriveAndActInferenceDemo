import logging
import argparse
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

def overlay_prediction(frame, prediction, confidence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Prediction: {prediction} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def sliding_window_inference(video_path, model_path, annotation_path, log_file, window_size=16, stride=1, resize=(224, 224), cuda_active=True):
    # Configure logger
    print("Configuring logger...")
    configure_logger(log_file)

    print("Loading model...")
    model, annotation_converter = load_model(model_path, annotation_path, cuda_active)
    print("Model loaded successfully.")

    print(f"Loading video from path: {video_path}")
    frames = load_video_segment(video_path, start_frame=0, n_frames=0)
    print(f"Number of frames loaded: {len(frames)}")
    if not frames:
        raise ValueError("No frames were loaded from the video. Check the video file or path.")

    num_frames = len(frames)

    # Create a video writer for output
    output_path = "./annotated_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = 15
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    # Initialize a list to store predictions for each frame
    frame_predictions = [None] * num_frames

    print("Starting sliding window inference...")

    # Sliding window inference
    for start in range(0, num_frames - window_size + 1, stride):
        print(f"Processing window: Frames {start} to {start + window_size - 1}")
        # Define window
        end = start + window_size
        window_frames = frames[start:end]

        resized_window = [cv2.resize(frame, resize) for frame in window_frames]
        # Run inference on the resized window
        top1_class, top1_class_conf, _ = run_inference_on_video_chunk(resized_window, model, annotation_converter, cuda_active)
        print(f"Window Prediction: {top1_class} ({top1_class_conf * 100:.2f}%)")

        # Store predictions for all frames in the window
        for i in range(start, end):
            if frame_predictions[i] is None:
                frame_predictions[i] = [(top1_class, top1_class_conf)]
            else:
                frame_predictions[i].append((top1_class, top1_class_conf))

    print("Combining predictions for overlapping frames...")

    # Combine predictions for overlapping frames
    for i, predictions in enumerate(frame_predictions):
        if predictions:
            # Select the prediction with the highest confidence
            top_prediction = max(predictions, key=lambda x: x[1])
            top1_class, top1_class_conf = top_prediction

            # Overlay prediction on the frame
            frame_with_overlay = overlay_prediction(frames[i], top1_class, top1_class_conf * 100)

            # Write to output video
            video_writer.write(frame_with_overlay)

            # Display the frame with overlay (real-time playback)
            cv2.imshow("Video with Predictions", frame_with_overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
                video_writer.release()
                cv2.destroyAllWindows()
                return

            # Log predictions
            log_prediction(i, top1_class, top1_class_conf * 100)

        if i % 50 == 0:
            print(f"Processed frame {i}/{num_frames}")

    # Release resources
    video_writer.release()
    cv2.destroyAllWindows()
    logging.info("Inference complete. Predictions logged successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--window_size", type=int, required=True, help="Sliding window size")

    args = parser.parse_args()
    sliding_window_inference(video_path=args.video_path, 
                             model_path=args.model_path, 
                             annotation_path=args.annotation_path, 
                             log_file=args.log_file,
                             window_size=args.window_size,  # Use the user-specified window size
                             stride=1,
                             resize=(224, 224),
                             cuda_active=True)