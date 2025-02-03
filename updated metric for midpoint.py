import pandas as pd
import re

class Metric:
    """Utility class for different metrics"""
    
    def __init__(self, ground_truth_csv, prediction_log, file_id):
        self.ground_truth = pd.read_csv(ground_truth_csv)

        # ✅ Filter ground truth for the correct file_id
        self.filtered_ground_truth = self.ground_truth[self.ground_truth['file_id'] == file_id].reset_index(drop=True)

        self.predictions = self.parse_predictions(prediction_log)
        self.segmented_predictions = self.convert_predictions_to_segments(self.predictions)
        self.file_id = file_id
        self.activity_classes = list(self.filtered_ground_truth['activity'].dropna().unique())
    
    @staticmethod
    def parse_predictions(log_file):
        predictions = []
        with open(log_file, 'r') as file:
            for line in file:
                try:
                    if not line.startswith("Frame") or "Predicted Activity:" not in line or "Confidence:" not in line:
                        print(f"Skipping malformed line: {line.strip()}")
                        continue

                    frame = int(line.split(":")[0].replace("Frame", "").strip())
                    activity = line.split("Predicted Activity: ")[1].split(", Confidence")[0].strip()
                    confidence_str = line.split("Confidence: ")[1].strip().replace("%", "")
                    confidence = float(confidence_str)

                    predictions.append({'frame': frame, 'activity': activity, 'confidence': confidence})

                except (IndexError, ValueError) as e:
                    print(f"Skipping line due to error: {line.strip()} -> {e}")

        return pd.DataFrame(predictions) if predictions else pd.DataFrame(columns=['frame', 'activity', 'confidence'])
    
    @staticmethod
    def convert_predictions_to_segments(predictions):
        if predictions.empty:
            return pd.DataFrame(columns=['frame_start', 'frame_end', 'activity'])

        segments = []
        current_activity = None
        current_start = None

        for _, row in predictions.iterrows():
            frame = row['frame']
            activity = row['activity']

            if activity != current_activity:
                if current_activity is not None:
                    segments.append({
                        'frame_start': current_start,
                        'frame_end': frame - 1,
                        'activity': current_activity
                    })
                current_activity = activity
                current_start = frame

        if current_activity is not None:
            segments.append({
                'frame_start': current_start,
                'frame_end': predictions.iloc[-1]['frame'],
                'activity': current_activity
            })

        return pd.DataFrame(segments)
    
    def evaluate_multiclass(self):
        all_activities = set(self.filtered_ground_truth['activity']).union(set(self.segmented_predictions['activity']))
        metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_activities}
        matched_groundtruths = set()  
        matched_predictions = set()

        # Fix: Prevent FN overcounting by tracking annotation_id**
        seen_annotations = set()

        for _, gt in self.filtered_ground_truth.iterrows():
            gt_start = gt['frame_start']
            gt_end = gt['frame_end']
            gt_activity = gt['activity']
            annotation_id = gt['annotation_id']  #  Track by annotation ID
            gt_midpoint = (gt_start + gt_end) // 2  

            matched_prediction = self.segmented_predictions[
                (self.segmented_predictions['frame_start'] <= gt_midpoint) &
                (self.segmented_predictions['frame_end'] >= gt_midpoint) &
                (self.segmented_predictions['activity'] == gt_activity)
            ]

            if not matched_prediction.empty:
                if annotation_id not in matched_groundtruths:
                    metrics[gt_activity]['tp'] += 1  #  Count only one TP per predicted segment
                    matched_groundtruths.add(annotation_id)  
            else:
                if annotation_id not in seen_annotations:  #  Prevent duplicate FN counting
                    metrics[gt_activity]['fn'] += 1  #  Ensure FN is counted only ONCE per annotation
                    seen_annotations.add(annotation_id)  # Store the annotation_id to prevent duplicate FNs
                    print(f" FN Detected: {gt_activity} (annotation_id={gt['annotation_id']}, frames={gt_start}-{gt_end})")  # ✅ Log FN
                

        
        seen_fp_annotations = set()

        for _, pred in self.segmented_predictions.iterrows():
            pred_activity = pred['activity']
            pred_midpoint = (pred['frame_start'] + pred['frame_end']) // 2  

            if pred_midpoint in matched_predictions:
                continue  

            matched_gt = self.filtered_ground_truth[
                (self.filtered_ground_truth['frame_start'] <= pred_midpoint) &
                (self.filtered_ground_truth['frame_end'] >= pred_midpoint) &
                (self.filtered_ground_truth['activity'] == pred_activity)
            ]

            if matched_gt.empty:
                if pred_activity not in seen_fp_annotations:  # ✅ Prevent duplicate FP counting per annotation
                    metrics[pred_activity]['fp'] += 1  
                    matched_predictions.add(pred_midpoint)
                    seen_fp_annotations.add(pred_activity)  # ✅ Track FP annotation to prevent multiple counting
                    print(f"⚠️ FP Detected: {pred_activity} (frames={pred['frame_start']}-{pred['frame_end']})")  # ✅ Log FP

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

    def evaluate(self):
        precision, recall, overall_precision, overall_recall = self.evaluate_multiclass()

        print("Precision per class (%):", {cls: f"{precision[cls]:.2f}%" for cls in precision})
        print("Recall per class (%):", {cls: f"{recall[cls]:.2f}%" for cls in recall})
        print(f"Overall Precision: {overall_precision:.2f}%")
        print(f"Overall Recall: {overall_recall:.2f}%")
        
        return {
            "Precision per class (%)": precision,
            "Recall per class (%)": recall,
            "Overall Precision": f"{overall_precision:.2f}%",
            "Overall Recall": f"{overall_recall:.2f}%"
        }

if __name__ == "__main__":
    ground_truth_csv = "./test_data/midlevel.chunks_90.split_1.test.csv"
    prediction_log = "results/sliding window/vp12/run2_2018-05-24-16-21-35.kinect_color_predictions_w32.log"
    file_id = "vp12/run2_2018-05-24-16-21-35.kinect_color"  # Filtering for this file_id

    metrics = Metric(ground_truth_csv, prediction_log, file_id)
    results = metrics.evaluate()
