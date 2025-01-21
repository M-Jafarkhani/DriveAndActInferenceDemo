import pandas as pd
import re

class Metric:
    """Utility class for different metrics"""
    
    def __init__(self, ground_truth_csv, prediction_log):
        self.ground_truth = pd.read_csv(ground_truth_csv)
        self.predictions = self.parse_predictions(prediction_log)
    
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

    

if __name__ == "__main__":
    # Define file paths
    ground_truth_csv = "./data/midlevel.chunks_90.split_0.train.csv"
    prediction_log = "./predictions.log"
    
    # Initialize and evaluate metrics
    metrics = Metric(ground_truth_csv, prediction_log)