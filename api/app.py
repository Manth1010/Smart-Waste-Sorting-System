from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import logging
from datetime import datetime
import os
from pathlib import Path

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from config import Config
from yolov7_wrapper import YOLOv7Detector
from feedback_manager import FeedbackManager

TEMPLATES_PATH = Path(__file__).parent.parent / 'templates'
app = Flask(__name__, template_folder=str(TEMPLATES_PATH))
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
model = None
config = Config()
feedback_manager = FeedbackManager()

class ModelManager:
    """Manages YOLOv8 model loading and inference"""
    
    def __init__(self):
        self.detector = None
        self.class_mapping = config.get_class_mapping()
        self.category_mapping = config.get_category_mapping()
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model (custom weights if available)"""
        try:
            self.detector = YOLOv7Detector()
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {str(e)}")
            raise e

    def reload_model(self):
        """Reload model (useful after retraining)"""
        logger.info("Reloading model...")
        self.load_model()
        logger.info("Model reloaded successfully")
    
    def predict(self, image_array):
        """Make prediction on image using YOLOv7"""
        if self.detector is None:
            raise ValueError("Model not loaded")
        
        # YOLOv7 expects RGB uint8; ensure image_array is RGB already
        result = self.detector.predict(image_array)
        result['timestamp'] = datetime.now().isoformat()
        return result

# Initialize model manager
try:
    model_manager = ModelManager()
    logger.info("Model manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model manager: {str(e)}")
    model_manager = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict waste class from uploaded image"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        result = model_manager.predict(image_rgb)
        
        logger.info(f"Prediction made: {result['predicted_class']} ({result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict waste class from base64 encoded image"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        result = model_manager.predict(image_rgb)
        
        logger.info(f"Prediction made: {result['predicted_class']} ({result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict waste class for multiple images"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images selected'}), 400
        
        results = []
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # Read image
                image_bytes = file.read()
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Make prediction
                result = model_manager.predict(image_rgb)
                result['filename'] = file.filename
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        logger.info(f"Batch prediction completed: {len(results)} images processed")
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        return jsonify({
            'model_type': 'YOLOv8',
            'image_size': config.IMAGE_SIZE,
            'classes': list(model_manager.class_mapping.keys()),
            'categories': list(set(model_manager.category_mapping.values())),
            'class_mapping': model_manager.class_mapping,
            'category_mapping': model_manager.category_mapping
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/sorting_decision', methods=['POST'])
def sorting_decision():
    """Make sorting decision based on prediction"""
    try:
        data = request.get_json()
        if 'prediction' not in data:
            return jsonify({'error': 'No prediction provided'}), 400

        prediction = data['prediction']
        predicted_category = prediction.get('predicted_category')
        confidence = prediction.get('confidence', 0)

        # Define confidence threshold
        confidence_threshold = 0.7

        if confidence < confidence_threshold:
            return jsonify({
                'action': 'manual_review',
                'reason': f'Low confidence ({confidence:.3f} < {confidence_threshold})',
                'predicted_category': predicted_category,
                'confidence': confidence
            })

        # Define sorting actions
        sorting_actions = {
            'organic': 'sort_to_organic_bin',
            'inorganic': 'sort_to_inorganic_bin'
        }

        action = sorting_actions.get(predicted_category, 'unknown_category')

        return jsonify({
            'action': action,
            'predicted_category': predicted_category,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in sorting decision: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """Real-time prediction for video frames"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500

        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make prediction
        result = model_manager.predict(image_rgb)

        # Add prediction ID for feedback tracking
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        response = {
            'prediction_id': prediction_id,
            'predicted_class': result['predicted_class'],
            'predicted_category': result['predicted_category'],
            'confidence': result['confidence'],
            'timestamp': result['timestamp']
        }

        logger.info(f"Real-time prediction: {result['predicted_class']} ({result['confidence']:.3f})")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in real-time prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for model improvement"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500

        data = request.get_json()
        required_fields = ['image', 'predicted_class', 'corrected_class', 'prediction_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing {field}'}), 400

        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get prediction details
        predicted_class = data['predicted_class']
        corrected_class = data['corrected_class']
        confidence = data.get('confidence', 0.0)
        prediction_id = data['prediction_id']

        # Add feedback
        feedback_id = feedback_manager.add_feedback(
            image_rgb, predicted_class, corrected_class, confidence
        )

        logger.info(f"Feedback submitted: {feedback_id} ({predicted_class} -> {corrected_class})")

        # Check if retraining should be triggered
        should_retrain = feedback_manager.should_retrain()
        current_count = feedback_manager.get_feedback_count()

        print(f"📊 Feedback Stats: {current_count}/{config.FEEDBACK_RETRAIN_THRESHOLD} samples collected")
        print(f"🎯 Retraining threshold: {'REACHED' if should_retrain else 'NOT REACHED'}")

        response = {
            'feedback_id': feedback_id,
            'prediction_id': prediction_id,
            'message': 'Feedback submitted successfully',
            'should_retrain': should_retrain,
            'current_feedback_count': current_count,
            'remaining_for_retrain': max(0, config.FEEDBACK_RETRAIN_THRESHOLD - current_count),
            'timestamp': datetime.now().isoformat()
        }

        # Auto-trigger retraining if threshold reached
        if should_retrain:
            try:
                print("🚀 Starting automatic retraining with feedback data...")
                print(f"📈 Training for {config.FEEDBACK_RETRAIN_EPOCHS} epochs")

                # Start retraining in background
                import subprocess
                import sys

                script_path = Path(__file__).parent.parent / 'retrain_with_feedback.py'
                if script_path.exists():
                    process = subprocess.Popen([
                        sys.executable, str(script_path),
                        '--epochs', str(config.FEEDBACK_RETRAIN_EPOCHS)
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    response['retraining_started'] = True
                    response['retraining_process_id'] = process.pid
                    response['retraining_epochs'] = config.FEEDBACK_RETRAIN_EPOCHS
                    logger.info(f"Auto-retraining started with PID: {process.pid}")

                    print(f"🔄 Retraining process started (PID: {process.pid})")
                    print("⏳ Model is now retraining in the background...")
                else:
                    response['retraining_error'] = 'Retraining script not found'
                    print("❌ Retraining script not found!")
            except Exception as e:
                logger.error(f"Error starting auto-retraining: {str(e)}")
                response['retraining_error'] = str(e)
                print(f"❌ Error starting retraining: {str(e)}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback_stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    try:
        stats = feedback_manager.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Trigger model retraining with feedback data"""
    try:
        if not feedback_manager.should_retrain():
            return jsonify({
                'error': 'Not enough feedback data for retraining',
                'current_count': feedback_manager.get_feedback_count(),
                'required_count': config.FEEDBACK_RETRAIN_THRESHOLD
            }), 400

        # Import training module
        import subprocess
        import sys

        # Run retraining script
        script_path = Path(__file__).parent.parent / 'retrain_with_feedback.py'
        if not script_path.exists():
            return jsonify({'error': 'Retraining script not found'}), 500

        # Run retraining in background
        process = subprocess.Popen([
            sys.executable, str(script_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # For now, return immediately (could implement async  handling)
        return jsonify({
            'message': 'Retraining started',
            'process_id': process.pid,
            'feedback_samples': feedback_manager.get_feedback_count()
        })

    except Exception as e:
        logger.error(f"Error starting retraining: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload model after retraining"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model manager not available'}), 500

        model_manager.reload_model()
        return jsonify({
            'message': 'Model reloaded successfully',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import time

    # Open browser at localhost (assuming API_HOST is '0.0.0.0' or similar)
    url = f'http://localhost:{config.API_PORT}'
    webbrowser.open(url)

    # Run  the app
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )
