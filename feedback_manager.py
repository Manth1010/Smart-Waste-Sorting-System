import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np

from config import Config


class FeedbackManager:
    """Manages user feedback for continuous learning"""

    def __init__(self):
        self.config = Config()
        self.feedback_images_dir = self.config.FEEDBACK_IMAGES_DIR
        self.feedback_labels_dir = self.config.FEEDBACK_LABELS_DIR
        self.feedback_log_path = self.config.FEEDBACK_DATA_DIR / "feedback_log.json"

        # Ensure directories exist
        self.feedback_images_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_labels_dir.mkdir(parents=True, exist_ok=True)

        # Load existing feedback log
        self.feedback_log = self._load_feedback_log()

    def _load_feedback_log(self) -> List[Dict]:
        """Load feedback log from file"""
        if self.feedback_log_path.exists():
            try:
                with open(self.feedback_log_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_feedback_log(self):
        """Save feedback log to file"""
        with open(self.feedback_log_path, 'w') as f:
            json.dump(self.feedback_log, f, indent=2, default=str)

    def add_feedback(self, image_array: np.ndarray, predicted_class: str,
                    corrected_class: str, confidence: float) -> str:
        """
        Add user feedback to the dataset

        Args:
            image_array: Image as numpy array (RGB)
            predicted_class: What the model predicted
            corrected_class: What the user corrected it to
            confidence: Model's confidence in prediction

        Returns:
            feedback_id: Unique identifier for this feedback
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        feedback_id = f"feedback_{timestamp}"

        # Save image
        image_filename = f"{feedback_id}.jpg"
        image_path = self.feedback_images_dir / image_filename

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr)

        # Create YOLO format label (assuming full image bounding box for classification)
        # For classification, we use a centered bounding box covering most of the image
        class_mapping = self.config.get_class_mapping()
        class_idx = class_mapping.get(corrected_class, 0)  # Default to first class if unknown

        # Create label file (YOLO format: class_idx x_center y_center width height)
        label_filename = f"{feedback_id}.txt"
        label_path = self.feedback_labels_dir / label_filename

        # Use a large bounding box covering most of the image
        with open(label_path, 'w') as f:
            f.write(f"{class_idx} 0.5 0.5 0.8 0.8\n")

        # Log feedback
        feedback_entry = {
            'id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'predicted_class': predicted_class,
            'corrected_class': corrected_class,
            'confidence': confidence,
            'image_path': str(image_path),
            'label_path': str(label_path)
        }

        self.feedback_log.append(feedback_entry)
        self._save_feedback_log()

        return feedback_id

    def get_feedback_count(self) -> int:
        """Get total number of feedback samples"""
        return len(self.feedback_log)

    def should_retrain(self) -> bool:
        """Check if we have enough feedback to trigger retraining"""
        return self.get_feedback_count() >= self.config.FEEDBACK_RETRAIN_THRESHOLD

    def get_feedback_samples(self) -> List[Tuple[Path, Path]]:
        """Get list of (image_path, label_path) pairs for feedback samples"""
        samples = []
        for entry in self.feedback_log:
            img_path = Path(entry['image_path'])
            label_path = Path(entry['label_path'])
            if img_path.exists() and label_path.exists():
                samples.append((img_path, label_path))
        return samples

    def clear_feedback_log(self):
        """Clear feedback log (useful after retraining)"""
        self.feedback_log = []
        self._save_feedback_log()

    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback data"""
        if not self.feedback_log:
            return {'total_samples': 0}

        corrections = {}
        for entry in self.feedback_log:
            pred = entry['predicted_class']
            corr = entry['corrected_class']
            key = f"{pred}->{corr}"
            corrections[key] = corrections.get(key, 0) + 1

        return {
            'total_samples': len(self.feedback_log),
            'corrections': corrections,
            'should_retrain': self.should_retrain()
        }
