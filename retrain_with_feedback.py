#!/usr/bin/env python3
"""
Retraining script that incorporates feedback data into the YOLO model training.

This script:
1. Loads existing trained model
2. Includes feedback data with original training data
3. Performs incremental training (fine-tuning)
4. Saves updated model weights
"""

import torch
import os
import shutil
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# Get class mapping from config
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from config import Config
from feedback_manager import FeedbackManager

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install it with: pip install ultralytics")
    sys.exit(1)


class FeedbackRetrainer:
    """Retrains YOLO model with feedback data"""

    def __init__(self):
        self.config = Config()
        self.feedback_manager = FeedbackManager()
        self.project_root = Path(__file__).parent

        # Dataset paths
        self.dataset_dir = self.project_root / 'data' / 'retrain_dataset'
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'

        # Train/val/test splits
        self.train_images_dir = self.images_dir / 'train'
        self.val_images_dir = self.images_dir / 'val'
        self.train_labels_dir = self.labels_dir / 'train'
        self.val_labels_dir = self.labels_dir / 'val'

        # Class mapping
        self.class_mapping = self.config.get_class_mapping()
        self.num_classes = len(self.class_mapping)
        self.class_names = sorted(list(self.class_mapping.keys()))

        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")

    def create_dataset_structure(self):
        """Create retraining dataset directory structure"""
        dirs = [
            self.images_dir,
            self.labels_dir,
            self.train_images_dir,
            self.val_images_dir,
            self.train_labels_dir,
            self.val_labels_dir
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ Created retraining dataset structure at: {self.dataset_dir}")

    def find_image_label_pairs(self, images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
        """Find matching image-label pairs"""
        pairs = []

        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(images_dir.glob(f"*{ext}"))

        for img_path in images:
            # Look for corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                print(f"Warning: No label file for {img_path.name}")

        return pairs

    def collect_training_data(self):
        """Collect original training data and feedback data"""
        all_pairs = []

        # Get original training data (from existing yolo_dataset if available)
        original_dataset = self.project_root / 'data' / 'yolo_dataset'
        if original_dataset.exists():
            original_train_images = original_dataset / 'images' / 'train'
            original_train_labels = original_dataset / 'labels' / 'train'

            if original_train_images.exists() and original_train_labels.exists():
                original_pairs = self.find_image_label_pairs(original_train_images, original_train_labels)
                all_pairs.extend(original_pairs)
                print(f"Found {len(original_pairs)} original training samples")

        # Get feedback data
        feedback_pairs = self.feedback_manager.get_feedback_samples()
        all_pairs.extend(feedback_pairs)
        print(f"Found {len(feedback_pairs)} feedback samples")

        if not all_pairs:
            raise ValueError("No training data found (original + feedback)")

        print(f"Total training samples: {len(all_pairs)}")

        # Split into train/val
        train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)

        # Copy files to retraining directories
        for img_path, label_path in train_pairs:
            shutil.copy2(img_path, self.train_images_dir / img_path.name)
            shutil.copy2(label_path, self.train_labels_dir / label_path.name)

        for img_path, label_path in val_pairs:
            shutil.copy2(img_path, self.val_images_dir / img_path.name)
            shutil.copy2(label_path, self.val_labels_dir / label_path.name)

        print(f"Retraining - Train: {len(train_pairs)}, Val: {len(val_pairs)}")
        return len(train_pairs), len(val_pairs)

    def create_dataset_yaml(self):
        """Create dataset.yaml file for retraining"""
        yaml_content = f"""# YOLO Retraining Dataset Configuration
# Paths relative to this file

path: {self.dataset_dir.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

# Classes
nc: {self.num_classes}  # number of classes
names: {self.class_names}  # class names
"""

        yaml_path = self.dataset_dir / 'dataset.yaml'
        yaml_path.write_text(yaml_content)
        print(f"✅ Created retraining dataset.yaml at: {yaml_path}")
        return yaml_path

    def retrain(self, epochs: int = 10, batch_size: int = 16, img_size: int = 640):
        """
        Retrain YOLO model with feedback data

        Args:
            epochs: Number of retraining epochs
            batch_size: Batch size
            img_size: Image size
        """
        # Collect and prepare data
        train_count, val_count = self.collect_training_data()
        dataset_yaml = self.create_dataset_yaml()

        # Load existing model for fine-tuning
        weights_path = self.config.MODELS_DIR / 'yolov7_custom.pt'
        if not weights_path.exists():
            print(f"❌ No existing model found at {weights_path}")
            print("Please train the model first using train_yolo.py")
            return None

        print(f"✅ Loading existing model: {weights_path}")
        model = YOLO(str(weights_path))

        if torch.cuda.is_available():
            device = "cuda:0"
            torch.cuda.set_device(0)
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("⚠️ CUDA not available, using CPU.")

        # Retrain the model (fine-tuning)
        print(f"\n🔄 Starting retraining with feedback data...")
        print(f"📊 Dataset: {train_count} train, {val_count} validation samples")
        print(f"⚙️  Config: {epochs} epochs, batch_size={batch_size}, img_size={img_size}")
        print(f"🎯 Fine-tuning existing model for continuous learning")
        print(f"⏳ This may take several minutes...")

        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='waste_classification_retrain',
            project=str(self.project_root / 'runs'),
            device=device,
            workers=4,
            patience=20,
            save=True,
            plots=True,
            # Fine-tuning parameters
            lr0=0.001,  # Lower learning rate for fine-tuning
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1
        )

        # Find best weights
        best_weights = Path(results.save_dir) / 'weights' / 'best.pt'

        if best_weights.exists():
            # Backup old model
            backup_path = self.config.MODELS_DIR / 'yolov7_custom_backup.pt'
            if weights_path.exists():
                shutil.copy2(weights_path, backup_path)
                print(f"📦 Backed up old model to: {backup_path}")

            # Copy new weights
            shutil.copy2(best_weights, weights_path)
            print(f"\n✅ Retraining complete!")
            print(f"   Updated model saved to: {weights_path}")

            # Clear feedback after successful retraining
            self.feedback_manager.clear_feedback_log()
            print("🧹 Cleared feedback log after successful retraining")

            return weights_path
        else:
            print(f"\n⚠️ Best weights not found at: {best_weights}")
            print(f"   Check runs directory: {results.save_dir}")
            return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Retrain YOLO model with feedback data')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of retraining epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Image size for retraining (default: 640)')

    args = parser.parse_args()

    retrainer = FeedbackRetrainer()

    # Create dataset structure
    retrainer.create_dataset_structure()

    # Retrain model
    result = retrainer.retrain(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    if result:
        print("\n" + "=" * 60)
        print("🎉 Retraining complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check retraining results in: runs/waste_classification_retrain/")
        print("2. Reload model in API: POST /reload_model")
        print("3. Test improved predictions")
    else:
        print("\n❌ Retraining failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
