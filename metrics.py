import pandas as pd
class Metric:
    """Utility class for different metrics"""
    
    def __init__(self, ground_truth_csv):
        self.ground_truth = pd.read_csv(ground_truth_csv)
        print(self.ground_truth)

    

if __name__ == "__main__":
    # Define file paths
    ground_truth_csv = "./data/midlevel.chunks_90.split_0.train.csv"
    
    # Initialize and evaluate metrics
    metrics = Metric(ground_truth_csv)