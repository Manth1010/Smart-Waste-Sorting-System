#!/usr/bin/env python3
"""
Train YOLO model using ultralytics for waste classification

This script:
1. Organizes your dataset into train/val/test splits
2. Creates YOLO dataset structure
3. Trains YOLOv8 model (or YOLOv7 compatible)
4. Exports trained weights to models/yolov7_custom.pt
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

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install it with: pip install ultralytics")
    sys.exit(1)


class YOLOTrainer:
    """Train YOLO model for waste classification"""
    
    def __init__(self):
        self.config = Config()
        self.project_root = Path(__file__).parent
        
        # Dataset paths
        self.dataset_dir = self.project_root / 'data' / 'yolo_dataset'
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # Train/val/test splits
        self.train_images_dir = self.images_dir / 'train'
        self.val_images_dir = self.images_dir / 'val'
        self.test_images_dir = self.images_dir / 'test'
        self.train_labels_dir = self.labels_dir / 'train'
        self.val_labels_dir = self.labels_dir / 'val'
        self.test_labels_dir = self.labels_dir / 'test'
        
        # Class mapping
        self.class_mapping = self.config.get_class_mapping()
        self.num_classes = len(self.class_mapping)
        self.class_names = sorted(list(self.class_mapping.keys()))
        self.index_to_class = {idx: cls_name for cls_name, idx in self.class_mapping.items()}

        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
    
    def create_dataset_structure(self):
        """Create YOLO dataset directory structure"""
        dirs = [
            self.images_dir,
            self.labels_dir,
            self.train_images_dir,
            self.val_images_dir,
            self.test_images_dir,
            self.train_labels_dir,
            self.val_labels_dir,
            self.test_labels_dir
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created dataset structure at: {self.dataset_dir}")
    
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
    
    def split_dataset(self, images_dir: Path, labels_dir: Path,
                     train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Split dataset into train/val/test"""

        pairs = self.find_image_label_pairs(images_dir, labels_dir)

        if not pairs:
            raise ValueError(f"No image-label pairs found in {images_dir}")

        print(f"Found {len(pairs)} image-label pairs")

        # Check class distribution
        class_counts = {}
        for _, label_path in pairs:
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        class_idx = int(line.split()[0])
                        class_name = self.index_to_class.get(class_idx, f"class_{class_idx}")
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except:
                continue

        print("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")

        # Shuffle and split
        train_pairs, temp_pairs = train_test_split(pairs, test_size=(1 - train_ratio), random_state=42)
        val_pairs, test_pairs = train_test_split(temp_pairs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

        # Copy files to respective directories
        for img_path, label_path in train_pairs:
            shutil.copy2(img_path, self.train_images_dir / img_path.name)
            shutil.copy2(label_path, self.train_labels_dir / label_path.name)

        for img_path, label_path in val_pairs:
            shutil.copy2(img_path, self.val_images_dir / img_path.name)
            shutil.copy2(label_path, self.val_labels_dir / label_path.name)

        for img_path, label_path in test_pairs:
            shutil.copy2(img_path, self.test_images_dir / img_path.name)
            shutil.copy2(label_path, self.test_labels_dir / label_path.name)

        print("✅ Dataset split complete!")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        yaml_content = f"""# YOLO Dataset Configuration
# Paths relative to this file

path: {self.dataset_dir.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')
test: images/test    # test images (relative to 'path')

# Classes
nc: {self.num_classes}  # number of classes
names: {self.class_names}  # class names
"""
        
        yaml_path = self.dataset_dir / 'dataset.yaml'
        yaml_path.write_text(yaml_content)
        print(f"✅ Created dataset.yaml at: {yaml_path}")
        return yaml_path
    
    def train(self, epochs: int = 100, batch_size: int = 16, img_size: int = 640, model_size: str = 'n'):
        """
        Train YOLO model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size (640, 1280, etc.)
            model_size: Model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
        """
        dataset_yaml = self.create_dataset_yaml()

        import os
        last_weights = "runs/waste_classification5/weights/last.pt"

        if os.path.exists(last_weights):
            print(f"✅ Resuming from checkpoint: {last_weights}")
            model = YOLO(last_weights)
        else:
            model_name = f'yolov8{model_size}.pt'
            print(f"\n🆕 Starting fresh training with model: {model_name}")
            model = YOLO(model_name)
        # Load pre-trained YOLOv8 model
        #model_name = f'yolov8{model_size}.pt'
        #print(f"\n🚀 Loading model: {model_name}")
        #model = YOLO(model_name)

        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            torch.cuda.set_device(0)  # ensure RTX 4050 is used
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("⚠️ CUDA not available, using CPU.")

        
        # Train the model
        print(f"\n🏋️ Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {img_size}")
        print(f"   Dataset: {dataset_yaml}")
        
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='waste_classification',
            project=str(self.project_root / 'runs'),
            device=device,  # ✅ forces GPU training
            workers=4,      # use multiple workers for faster data loading
            patience=50,
            save=True,
            plots=True
        )
        # Find best weights
        best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
        
        if best_weights.exists():
            # Copy to models directory
            output_path = self.config.MODELS_DIR / 'yolov7_custom.pt'
            shutil.copy2(best_weights, output_path)
            print(f"\n✅ Training complete!")
            print(f"   Best weights saved to: {output_path}")
            return output_path
        else:
            print(f"\n⚠️ Best weights not found at: {best_weights}")
            print(f"   Check runs directory: {results.save_dir}")
            return None
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO model for waste classification')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                       help='Directory containing YOLO format .txt label files')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: n)')
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer()
    
    # Create dataset structure
    trainer.create_dataset_structure()
    
    # Split dataset
    trainer.split_dataset(Path(args.images), Path(args.labels))
    
    # Train model
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_size=args.model_size
    )
    
    print("\n" + "=" * 60)
    print("🎉 Training pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check training results in: runs/waste_classification/")
    print("2. Best weights saved to: models/yolov7_custom.pt")
    print("3. Start API: python api/app.py")


if __name__ == "__main__":
    main()



