from InferenceOnSingleVideoMini import load_model, load_video_segment, run_inference_on_video_chunk
import cv2

def sliding_window_inference(video_path, model_path, annotation_path, window_size=16, stride=1, resize=None, cuda_active=True):
    model, annotation_converter = load_model(model_path, annotation_path, cuda_active)
    frames = load_video_segment(video_path, start_frame=0, n_frames=0)  # Load all frames
    num_frames = len(frames)
    
    if resize:
        frames = [cv2.resize(frame, resize) for frame in frames]

    # Initialize predictions list
    frame_predictions = []

    for start in range(0, num_frames - window_size + 1, stride):
        # Define window
        end = start + window_size
        window_frames = frames[start:end]
        top1_class, _, _ = run_inference_on_video_chunk(window_frames, model, annotation_converter, cuda_active)

        # Assign the prediction to the first frame of the window
        for i in range(start, min(start + stride, num_frames)):
            frame_predictions.append((i, top1_class))

    return frame_predictions

if __name__ == "__main__":
    # Define paths
    video_path = "./run1b_2018-05-29-14-02-47.kinect_color.mp4"
    model_path = "./demo_models/best_model.pth"
    annotation_path = "./demo_models/annotation_converter.pkl"

    # Perform sliding window inference
    predictions = sliding_window_inference(
        video_path=video_path,
        model_path=model_path,
        annotation_path=annotation_path,
        window_size=16,
        stride=1,
        resize=(224, 224),
        cuda_active=True
    )

    # Save or print results
    for frame_idx, pred in predictions:
        print(f"Frame {frame_idx}: Predicted Activity - {pred}")
