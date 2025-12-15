#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations to YOLO format (.txt files)

This script converts LabelMe format JSON files (one per image) to YOLO format.
LabelMe format: {"shapes": [{"label": "...", "points": [[x1,y1], [x2,y2]], ...}], "imageWidth": ..., "imageHeight": ...}
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

# Get class mapping from config
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from config import Config


class LabelMeToYOLOConverter:
    """Convert LabelMe JSON annotations to YOLO format"""
    
    def __init__(self):
        self.config = Config()
        self.class_mapping = self.config.get_class_mapping()
        # Create reverse mapping: class_name -> class_id
        self.class_name_to_id = {name: idx for name, idx in self.class_mapping.items()}
        
        # Label mapping: map LabelMe labels to our class names
        self.label_mapping = {
            'bananapeel': 'Banana_Peel',
            'banana_peel': 'Banana_Peel',
            'banana peel': 'Banana_Peel',
            'orangepeel': 'Orange_Peel',
            'orange_peel': 'Orange_Peel',
            'orange peel': 'Orange_Peel',
            'plastic': 'Plastic',
            'paper': 'Paper',
            'wood': 'Wood'
        }
        
        print(f"Class mapping: {self.class_name_to_id}")
        print(f"Label mapping: {self.label_mapping}")
    
    def normalize_label(self, label: str) -> str:
        """Normalize LabelMe label to our class name"""
        # Remove category info (e.g., "Bananapeel, organic" -> "Bananapeel")
        label_clean = label.split(',')[0].strip().lower()
        
        # Map to our class name
        for key, value in self.label_mapping.items():
            if key in label_clean:
                return value
        
        # Direct check if already matches
        for class_name in self.class_name_to_id.keys():
            if class_name.lower() in label_clean or label_clean in class_name.lower():
                return class_name
        
        return None
    
    def convert_points_to_yolo(self, points: list, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert LabelMe points (two corner points) to YOLO format
        
        LabelMe: points = [[x1, y1], [x2, y2]]
        YOLO: center_x, center_y, width, height (all normalized 0-1)
        """
        if len(points) != 2:
            raise ValueError(f"Expected 2 points, got {len(points)}")
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Ensure x1 < x2 and y1 < y2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Calculate center and dimensions
        center_x = (x_min + x_max) / 2.0 / img_width
        center_y = (y_min + y_max) / 2.0 / img_height
        width_norm = (x_max - x_min) / img_width
        height_norm = (y_max - y_min) / img_height
        
        # Ensure normalized values are within [0, 1]
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        return center_x, center_y, width_norm, height_norm
    
    def convert_json_file(self, json_path: Path, output_dir: Path) -> bool:
        """Convert a single LabelMe JSON file to YOLO .txt file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get image dimensions
            img_width = data.get('imageWidth')
            img_height = data.get('imageHeight')
            
            if not img_width or not img_height:
                print(f"Warning: No image dimensions in {json_path}")
                return False
            
            # Get corresponding image filename
            image_path = data.get('imagePath', '')
            if not image_path:
                # Try to infer from JSON filename
                image_path = json_path.stem + '.jpg'
            
            # Create output txt file
            txt_filename = Path(image_path).stem + '.txt'
            txt_path = output_dir / txt_filename
            
            # Process shapes/annotations
            shapes = data.get('shapes', [])
            valid_annotations = []
            
            for shape in shapes:
                label = shape.get('label', '')
                shape_type = shape.get('shape_type', '')
                
                if shape_type != 'rectangle':
                    print(f"Warning: Skipping non-rectangle shape in {json_path}")
                    continue
                
                # Normalize label to our class name
                class_name = self.normalize_label(label)
                
                if not class_name or class_name not in self.class_name_to_id:
                    print(f"Warning: Unmapped label '{label}' -> skipped in {json_path}")
                    continue
                
                # Get points
                points = shape.get('points', [])
                if len(points) != 2:
                    print(f"Warning: Invalid points in {json_path}")
                    continue
                
                # Convert to YOLO format
                try:
                    center_x, center_y, width_norm, height_norm = self.convert_points_to_yolo(
                        points, img_width, img_height
                    )
                    
                    class_id = self.class_name_to_id[class_name]
                    valid_annotations.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                    )
                except Exception as e:
                    print(f"Error converting bbox in {json_path}: {e}")
                    continue
            
            # Write YOLO format file
            if valid_annotations:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.writelines(valid_annotations)
                print(f"✓ {json_path.name} -> {txt_filename} ({len(valid_annotations)} objects)")
                return True
            else:
                # Create empty file if no valid annotations
                txt_path.touch()
                print(f"⚠ {json_path.name} -> {txt_filename} (empty, no valid annotations)")
                return False
                
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            return False
    
    def convert_directory(self, json_dir: str, output_dir: str):
        """Convert all JSON files in a directory"""
        json_dir = Path(json_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")
        
        # Find all JSON files
        json_files = list(json_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {json_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files")
        print(f"Output directory: {output_dir}")
        print("-" * 60)
        
        success_count = 0
        for json_file in json_files:
            if self.convert_json_file(json_file, output_dir):
                success_count += 1
        
        print("-" * 60)
        print(f"✅ Conversion complete!")
        print(f"   Processed: {len(json_files)} files")
        print(f"   Success: {success_count} files")
        print(f"   Output: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON annotations to YOLO format')
    parser.add_argument('--json_dir', type=str, required=True,
                       help='Directory containing LabelMe JSON files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for YOLO .txt files')
    
    args = parser.parse_args()
    
    converter = LabelMeToYOLOConverter()
    converter.convert_directory(args.json_dir, args.output)


if __name__ == "__main__":
    main()



