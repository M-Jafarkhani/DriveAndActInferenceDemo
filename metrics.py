import pandas as pd
import re

class Metric:
    """Utility class for different metrics"""
    
    def __init__(self, ground_truth_csv, prediction_log, file_id):
        self.ground_truth = pd.read_csv(ground_truth_csv)
        self.predictions = self.parse_predictions(prediction_log)
        self.file_id = file_id
    
    @staticmethod
    def parse_predictions(log_file):
        predictions = {}
        with open(log_file, 'r') as file:
            for line in file:
                match = re.match(r".*Frame (\d+): Predicted Activity: (.*), Confidence: (\d+\.\d+)%", line)
                if match:
                    frame_idx = int(match.group(1))
                    activity = match.group(2)
                    confidence = float(match.group(3))
                    predictions[frame_idx] = (activity, confidence)
        return predictions
    
    def midpoint_hit_criteria(self):
        """For each ground truth activity window, if the predicted activity for the midpoint frame
           matches the ground truth activity, we count it as a "correct hit."

        Returns:
            float: correct_hits / total_instances
        """
        correct_hits = 0
        total_windows = 0

        for _, row in self.ground_truth.iterrows():
            # Get the frame range for the activity
            start_frame = row['frame_start']
            end_frame = row['frame_end']
            activity = row['activity']
            file_id = row['file_id']

            # Calculate midpoint frame
            midpoint_frame = (start_frame + end_frame) // 2
            
            # Check prediction for the midpoint frame
            if self.file_id == file_id:
                if midpoint_frame in self.predictions:
                    predicted_activity, _ = self.predictions[midpoint_frame]
                    if predicted_activity == activity:
                        correct_hits += 1

                total_windows += 1

        return correct_hits / total_windows if total_windows > 0 else 0.0
    
    def iou(self):
        """
        Calculate the Intersection over Union (IoU) metric. Here, 
        the overlapping part between the ground truth window and the predicted window is intersection and
        the total area covered by both the ground truth and predicted windows is union.

        Returns:
            float: The average IoU score for each ground truth.
        """
        iou_scores = []

        for _, row in self.ground_truth.iterrows():
            # Get the frame range for the activity
            gt_start = row['frame_start']
            gt_end = row['frame_end']
            activity = row['activity']
            file_id = row['file_id']
            
            if self.file_id == file_id:
                # Find all predicted frames that match the activity
                predicted_frames = [
                    idx for idx, (predicted_activity, _) in self.predictions.items()
                    if predicted_activity == activity and gt_start <= idx <= gt_end
                ]


                if not predicted_frames:
                    iou_scores.append(0)
                    continue

                # Calculate intersection and union
                pred_start = min(predicted_frames)
                pred_end = max(predicted_frames)

                intersection_start = max(gt_start, pred_start)
                intersection_end = min(gt_end, pred_end)
                intersection = max(0, intersection_end - intersection_start + 1)

                union_start = min(gt_start, pred_start)
                union_end = max(gt_end, pred_end)
                union = max(0, union_end - union_start + 1)

                iou = intersection / union if union > 0 else 0
                iou_scores.append(iou)

        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    

if __name__ == "__main__":
    # Define file paths
    ground_truth_csv = "./data/midlevel.chunks_90.split_0.train.csv"
    prediction_log = "./predictions.log"
    file_id = "vp1/run1b_2018-05-29-14-02-47.kinect_color"
    
    # Initialize and evaluate metrics
    metrics = Metric(ground_truth_csv, prediction_log, file_id)
    print(metrics.midpoint_hit_criteria())
    print(metrics.iou())