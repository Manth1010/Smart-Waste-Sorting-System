import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import Config


class YOLOv7Detector:
    """YOLO-based detector using ultralytics engine for inference.

    - If models/yolov7_custom.pt exists (your trained weights), it will be used.
    - Otherwise, falls back to a small public model (yolov8n.pt) for demo only.
      Note: public classes won't match your 5 waste classes.
    """

    def __init__(self, device: Optional[str] = None):
        self.config = Config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_mapping = self.config.get_class_mapping()
        self.category_mapping = self.config.get_category_mapping()

        # Reverse mapping: index -> class name
        self.index_to_class = {idx: cls_name for cls_name, idx in self.class_mapping.items()}

        self._load_model()

    def _load_model(self) -> None:
        weights_path = Path(self.config.MODELS_DIR) / "yolov7_custom.pt"
        if weights_path.exists():
            self.model = YOLO(str(weights_path))
        else:
            raise FileNotFoundError(f"Custom model weights not found at {weights_path}. Please train and save the model first.")

    def _prepare_image(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
        return image_rgb

    def _aggregate_predictions(self, detections: List[Dict]) -> Tuple[str, float, Dict[str, float]]:
        # Aggregate per-class max confidence among detections limited to configured classes
        per_class_conf: Dict[str, float] = {cls_name: 0.0 for cls_name in self.class_mapping.keys()}

        for det in detections:
            cls_name = det.get('name')
            conf = float(det.get('confidence', 0.0))
            if cls_name in per_class_conf:
                per_class_conf[cls_name] = max(per_class_conf[cls_name], conf)

        # Pick top class; if all zeros, pick the max among all detections regardless of mapping
        predicted_class = max(per_class_conf.items(), key=lambda kv: kv[1])[0] if per_class_conf else None
        confidence = per_class_conf[predicted_class] if predicted_class is not None else 0.0

        if (predicted_class is None) or (confidence == 0.0 and len(detections) > 0):
            # If no configured class matched, fallback to best detection
            best = max(detections, key=lambda d: float(d.get('confidence', 0.0)))
            predicted_class = best.get('name', 'Unknown')
            confidence = float(best.get('confidence', 0.0))
            # Keep per_class_conf as-is; ensure predicted class exists in dict
            if predicted_class not in per_class_conf:
                per_class_conf[predicted_class] = confidence

        return predicted_class, confidence, per_class_conf

    def predict(self, image_rgb: np.ndarray) -> Dict:
        """Run YOLOv8 detection and produce classification-style output."""
        img = self._prepare_image(image_rgb)

        with torch.no_grad():
            results_list = self.model.predict(source=img, verbose=False)

        detections: List[Dict] = []
        if results_list:
            res = results_list[0]
            names = res.names  # id -> name mapping
            boxes = res.boxes
            if boxes is not None:
                for b in boxes:
                    cls_id = int(b.cls.item())
                    conf = float(b.conf.item())
                    xyxy = b.xyxy[0].tolist()
                    name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    detections.append({
                        'name': name,
                        'confidence': conf,
                        'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        'category': self.category_mapping.get(name, 'unknown')
                    })

        predicted_class, confidence, all_predictions = self._aggregate_predictions(detections)

        predicted_category = self.category_mapping.get(predicted_class, 'unknown')

        return {
            'predicted_class': predicted_class,
            'predicted_category': predicted_category,
            'confidence': float(confidence),
            'all_predictions': all_predictions,
            'detections': detections,
        }


