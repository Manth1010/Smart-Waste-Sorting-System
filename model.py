import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from pathlib import Path

from config import Config

class WasteClassificationModel:
    """CNN model for waste classification"""
    
    def __init__(self, model_type='efficientnet', num_classes=5):
        self.config = Config()
        self.model_type = model_type
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_custom_cnn(self):
        """Create a custom CNN architecture"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.config.IMAGE_SIZE, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self):
        """Create model using transfer learning"""
        if self.model_type == 'efficientnet':
            base_model = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.config.IMAGE_SIZE, 3)
            )
        elif self.model_type == 'resnet':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.config.IMAGE_SIZE, 3)
            )
        elif self.model_type == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.config.IMAGE_SIZE, 3)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """Build the model based on type"""
        if self.model_type in ['efficientnet', 'resnet', 'mobilenet']:
            self.model = self.create_transfer_learning_model()
        else:
            self.model = self.create_custom_cnn()
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return self.model
    
    def get_callbacks(self):
        """Get training callbacks"""
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=str(self.config.MODELS_DIR / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(
                filename=str(self.config.LOGS_DIR / 'training_log.csv'),
                append=True
            )
        ]
        
        return callbacks_list
    
    def create_data_generators(self):
        """Create data generators for training and validation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_DATA_DIR,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.config.VALIDATION_DATA_DIR,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators()
        
        # Get callbacks
        callbacks_list = self.get_callbacks()
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            train_generator,
            epochs=self.config.EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        self.model.save(str(self.config.MODELS_DIR / 'final_model.h5'))
        
        return self.history
    
    def evaluate(self):
        """Evaluate the model on test data"""
        if self.model is None:
            # Load best model
            self.model = tf.keras.models.load_model(
                str(self.config.MODELS_DIR / 'best_model.h5')
            )
        
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.config.TEST_DATA_DIR,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate model
        test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
            test_generator, verbose=1
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f}")
        
        # Generate predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Get class names
        class_names = list(test_generator.class_indices.keys())
        
        # Generate classification report
        report = classification_report(
            true_classes, predicted_classes, 
            target_names=class_names, output_dict=True
        )
        
        # Save evaluation results
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_top3_accuracy': float(test_top3_accuracy),
            'classification_report': report
        }
        
        with open(self.config.LOGS_DIR / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Plot top-3 accuracy
        axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
        axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        axes[1, 0].set_title('Model Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        
        # Plot learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.config.LOGS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predict class for a single image"""
        if self.model is None:
            self.model = tf.keras.models.load_model(
                str(self.config.MODELS_DIR / 'best_model.h5')
            )
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.config.IMAGE_SIZE
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class names
        class_names = list(self.config.get_class_mapping().keys())
        predicted_class = class_names[predicted_class_idx]
        predicted_category = self.config.get_category_mapping()[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'predicted_category': predicted_category,
            'confidence': float(confidence),
            'all_predictions': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }

if __name__ == "__main__":
    # Example usage
    model = WasteClassificationModel(model_type='efficientnet')
    
    # Build and train model
    model.build_model()
    print("Model built successfully!")
    print(f"Model summary:")
    model.model.summary()
    
    # Train the model
    history = model.train()
    
    # Evaluate the model
    results = model.evaluate()
    
    # Plot training history
    model.plot_training_history()
    
    print("Training completed!")
