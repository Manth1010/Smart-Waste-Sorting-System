import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
from tqdm import tqdm
import json

from config import Config

class DataPreprocessor:
    """Data preprocessing pipeline for waste classification"""
    
    def __init__(self):
        self.config = Config()
        self.class_mapping = self.config.get_class_mapping()
        self.category_mapping = self.config.get_category_mapping()
        
    def load_and_preprocess_image(self, image_path, target_size=None):
        """Load and preprocess a single image"""
        if target_size is None:
            target_size = self.config.IMAGE_SIZE
            
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def create_dataframe(self, data_dir):
        """Create a dataframe with image paths and labels"""
        data = []
        
        for class_name in self.class_mapping.keys():
            class_dir = Path(data_dir) / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist")
                continue
                
            for image_path in class_dir.glob("*.jpg") + class_dir.glob("*.jpeg") + class_dir.glob("*.png"):
                data.append({
                    'image_path': str(image_path),
                    'class': class_name,
                    'category': self.category_mapping[class_name]
                })
        
        return pd.DataFrame(data)
    
    def split_data(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['category']
        )
        
        # Second split: separate validation set from train_val
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=train_val_df['category']
        )
        
        return train_df, val_df, test_df
    
    def organize_data(self, source_dir, target_base_dir):
        """Organize data into train/validation/test directories"""
        # Create dataframe
        df = self.create_dataframe(source_dir)
        
        if df.empty:
            raise ValueError("No images found in the source directory")
        
        print(f"Found {len(df)} images")
        print(f"Class distribution:")
        print(df['class'].value_counts())
        print(f"Category distribution:")
        print(df['category'].value_counts())
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        # Copy files to organized structure
        for split_name, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            target_dir = Path(target_base_dir) / split_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for _, row in tqdm(split_df.iterrows(), desc=f"Organizing {split_name} data"):
                source_path = Path(row['image_path'])
                target_path = target_dir / row['class'] / source_path.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(source_path, target_path)
        
        # Save metadata
        metadata = {
            'total_images': len(df),
            'train_images': len(train_df),
            'validation_images': len(val_df),
            'test_images': len(test_df),
            'classes': list(self.class_mapping.keys()),
            'class_mapping': self.class_mapping,
            'category_mapping': self.category_mapping
        }
        
        with open(Path(target_base_dir) / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def get_data_generators(self, batch_size=None):
        """Create data generators for training"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        # Data augmentation for training
        train_transform = A.Compose([
            A.Resize(self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation/test transforms
        val_transform = A.Compose([
            A.Resize(self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return train_transform, val_transform

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Organize your raw data
    source_directory = input("Enter path to your raw data directory: ")
    if os.path.exists(source_directory):
        metadata = preprocessor.organize_data(source_directory, preprocessor.config.PROCESSED_DATA_DIR)
        print("Data organization completed!")
        print(f"Metadata saved: {metadata}")
    else:
        print("Source directory does not exist. Please provide a valid path.")
