import pandas as pd
import re

class Metric:
    """Utility class for different metrics"""
    
    def __init__(self, ground_truth_csv, prediction_log, file_id):
        self.ground_truth = pd.read_csv(ground_truth_csv)
        self.predictions = self.parse_predictions(prediction_log)
        self.segmented_predictions = self.convert_predictions_to_segments(self.predictions)
        self.file_id = file_id
        self.activity_classes = list(self.ground_truth['activity'].dropna().unique())
        self.filtered_ground_truth = self.ground_truth[self.ground_truth['file_id'] == file_id]
    
    @staticmethod
    def parse_predictions(log_file):
        predictions = []
        with open(log_file, 'r') as file:
            for line in file:
                try:
                    # Split the line to extract frame and activity information
                    parts = line.strip().split(' - ')  
                    frame_part = parts[1].split(': ')[0]  
                    frame = int(frame_part.split()[1])
                    
                    # Extract activity and confidence
                    activity_part = parts[1].split(': ', maxsplit=1)[1] 
                    activity = activity_part.split(', Confidence')[0].replace("Predicted Activity: ", "").strip() 
                    
                    # Extract confidence if it exists
                    confidence = 0.0  # Default confidence
                    if "Confidence" in activity_part:  # Check if "Confidence" exists in the string
                        confidence = float(activity_part.split('Confidence: ')[1].strip('%'))  # Extract confidence as float

                    # Append to predictions list
                    predictions.append({'frame': frame, 'activity': activity, 'confidence': confidence})
                
                except (IndexError, ValueError) as e:
                    print(f"Skipping line due to error: {line.strip()} -> {e}")

        # Convert predictions to DataFrame
        return pd.DataFrame(predictions)
    
    @staticmethod
    def convert_predictions_to_segments(predictions):
        segments = []
        current_activity = None
        current_start = None

        for _, row in predictions.iterrows():
            frame = row['frame']
            activity = row['activity']

            # Start a new segment if the activity changes
            if activity != current_activity:
                if current_activity is not None:
                    # Save the previous segment
                    segments.append({
                        'frame_start': current_start,
                        'frame_end': frame - 1,
                        'activity': current_activity
                    })
                # Start a new segment
                current_activity = activity
                current_start = frame

        # Save the last segment
        if current_activity is not None:
            segments.append({
                'frame_start': current_start,
                'frame_end': predictions.iloc[-1]['frame'],
                'activity': current_activity
            })

        return pd.DataFrame(segments)
    
    
    def evaluate_multiclass(self):
        all_activities = set(self.filtered_ground_truth['activity']).union(set(self.segmented_predictions['activity']))

        # Initialize metrics for all known activities
        metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_activities}
        matched_chunks = set()  
        
        for _, gt in self.filtered_ground_truth.iterrows():
            gt_start = gt['frame_start']
            gt_end = gt['frame_end']
            gt_activity = gt['activity']
            chunk_key = (gt['annotation_id'], gt['chunk_id'])
            gt_midpoint = (gt_start + gt_end) // 2

            matched_prediction = self.segmented_predictions[
                (self.segmented_predictions['frame_start'] <= gt_midpoint) &
                (self.segmented_predictions['frame_end'] >= gt_midpoint) &
                (self.segmented_predictions['activity'] == gt_activity)
            ]

            if not matched_prediction.empty:
                #  True Positive: A prediction exists for this midpoint
                if chunk_key not in matched_chunks:
                    metrics[gt_activity]['tp'] += 1
                    matched_chunks.add(chunk_key)  
                else:
                    metrics[gt_activity]['fp'] += 1
            else:
                # False Negative: No correct prediction found for this activity midpoint
                metrics[gt_activity]['fn'] += 1

        # Count False Positives for unmatched predictions
        for _, pred in self.segmented_predictions.iterrows():
            pred_activity = pred['activity']

            if pred_activity in metrics and metrics[pred_activity]['tp'] > 0:
                continue

            if pred_activity not in metrics:
                continue
            
            metrics[pred_activity]['fp'] += 1

        precision, recall = {}, {}
        for cls in all_activities:
            tp, fp, fn = metrics[cls]['tp'], metrics[cls]['fp'], metrics[cls]['fn']
            precision[cls] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall[cls] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            
        overall_tp = sum(metrics[cls]['tp'] for cls in all_activities)
        overall_fp = sum(metrics[cls]['fp'] for cls in all_activities)
        overall_fn = sum(metrics[cls]['fn'] for cls in all_activities)

        overall_precision = overall_tp / (overall_tp + overall_fp) * 100 if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) * 100 if (overall_tp + overall_fn) > 0 else 0

        return precision, recall, overall_precision, overall_recall
        
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

        for _, row in self.filtered_ground_truth.iterrows():
            # Get the frame range for the activity
            gt_start = row['frame_start']
            gt_end = row['frame_end']
            activity = row['activity']
            
            # Find all predicted frames that match the activity
            predicted_frames = self.predictions[
                                    (self.predictions["activity"] == activity) & 
                                    (self.predictions["frame"] >= gt_start) & 
                                    (self.predictions["frame"] <= gt_end)
                                ]["frame"].tolist()


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
    
    def evaluate(self):
        mean_iou = self.iou()
        precision, recall, overall_precision, overall_recall = self.evaluate_multiclass()

        print("Precision per class (%):", {cls: f"{precision[cls]:.2f}%" for cls in precision})
        print("Recall per class (%):", {cls: f"{recall[cls]:.2f}%" for cls in recall})
        print(f"Overall Precision: {overall_precision:.2f}%")
        print(f"Overall Recall: {overall_recall:.2f}%")
        print(f"Mean IoU: {mean_iou*100:.2f}%")
    

if __name__ == "__main__":
    # Define file paths
    ground_truth_csv = "./data/midlevel.chunks_90.split_0.train.csv"
    prediction_log = "./predictions.log"
    file_id = "vp1/run1b_2018-05-29-14-02-47.kinect_color"
    
    # Initialize and evaluate metrics
    metrics = Metric(ground_truth_csv, prediction_log, file_id)
    results = metrics.evaluate()