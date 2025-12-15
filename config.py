import os
import json
from pathlib import Path

class Config:
    """Configuration class for Smart Waste Sorting System"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    TRAIN_DATA_DIR = DATA_DIR / "train"
    TEST_DATA_DIR = DATA_DIR / "test"
    VALIDATION_DATA_DIR = DATA_DIR / "validation"
    FEEDBACK_DATA_DIR = DATA_DIR / "feedback"
    FEEDBACK_IMAGES_DIR = FEEDBACK_DATA_DIR / "images"
    FEEDBACK_LABELS_DIR = FEEDBACK_DATA_DIR / "labels"
    
    # Model configuration
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Classes
    CLASSES = {
        'organic': ['Banana_Peel', 'Orange_Peel'],
        'inorganic': ['Plastic', 'Paper', 'Wood']
    }
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    DEBUG = True
    
    # Camera configuration
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    # Feedback configuration
    FEEDBACK_RETRAIN_THRESHOLD = 40  # Retrain when 40 feedback samples collected
    FEEDBACK_RETRAIN_EPOCHS = 10     # Number of epochs for feedback retraining
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.TRAIN_DATA_DIR,
            cls.TEST_DATA_DIR,
            cls.VALIDATION_DATA_DIR,
            cls.FEEDBACK_DATA_DIR,
            cls.FEEDBACK_IMAGES_DIR,
            cls.FEEDBACK_LABELS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_class_mapping(cls):
        """Get class to index mapping"""
        all_classes = []
        for category, items in cls.CLASSES.items():
            all_classes.extend(items)
        
        return {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    
    @classmethod
    def get_category_mapping(cls):
        """Get category mapping for each class"""
        category_mapping = {}
        for category, items in cls.CLASSES.items():
            for item in items:
                category_mapping[item] = category
        return category_mapping

# Create directories when module is imported
Config.create_directories()
