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
            correct_hits / total_instances
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

    

if __name__ == "__main__":
    # Define file paths
    ground_truth_csv = "./data/midlevel.chunks_90.split_0.train.csv"
    prediction_log = "./predictions.log"
    file_id = "vp1/run1b_2018-05-29-14-02-47.kinect_color"
    
    # Initialize and evaluate metrics
    metrics = Metric(ground_truth_csv, prediction_log, file_id)
    print(metrics.midpoint_hit_criteria())