import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import requests
from queue import Queue
import base64

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from config import Config

class CameraManager:
    """Manages camera operations for real-time waste detection"""
    
    def __init__(self, camera_id=0, api_url="http://localhost:5000"):
        self.config = Config()
        self.camera_id = camera_id
        self.api_url = api_url
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.results_queue = Queue(maxsize=50)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOGS_DIR / 'camera.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.detection_interval = 1.0  # seconds
        self.confidence_threshold = 0.7
        self.last_detection_time = 0
        
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
            
            self.logger.info(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            return False
    
    def capture_frame(self):
        """Capture a frame from camera"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for detection"""
        # Resize frame
        frame_resized = cv2.resize(frame, self.config.IMAGE_SIZE)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
    
    def send_prediction_request(self, image_array):
        """Send prediction request to API"""
        try:
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare request data
            data = {'image': f'data:image/jpeg;base64,{image_base64}'}
            
            # Send request
            response = requests.post(
                f"{self.api_url}/predict_base64",
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error sending prediction request: {str(e)}")
            return None
    
    def process_detection(self, frame):
        """Process detection on frame"""
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self.last_detection_time < self.detection_interval:
            return None
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Send prediction request
        result = self.send_prediction_request(processed_frame)
        
        if result:
            self.last_detection_time = current_time
            
            # Add frame info to result
            result['frame_timestamp'] = current_time
            result['frame_shape'] = frame.shape
            
            # Add to results queue
            self.results_queue.put(result)
            
            self.logger.info(
                f"Detection: {result['predicted_class']} "
                f"({result['predicted_category']}) - "
                f"Confidence: {result['confidence']:.3f}"
            )
            
            return result
        
        return None
    
    def camera_thread(self):
        """Camera capture thread"""
        while self.is_running:
            frame = self.capture_frame()
            if frame is not None:
                # Add frame to queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Process detection
                self.process_detection(frame)
            
            time.sleep(0.033)  # ~30 FPS
    
    def start_camera(self):
        """Start camera and detection"""
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        
        # Start camera thread
        self.camera_thread_obj = threading.Thread(target=self.camera_thread)
        self.camera_thread_obj.daemon = True
        self.camera_thread_obj.start()
        
        self.logger.info("Camera started successfully")
        return True
    
    def stop_camera(self):
        """Stop camera and detection"""
        self.is_running = False
        
        if self.camera_thread_obj.is_alive():
            self.camera_thread_obj.join(timeout=2)
        
        if self.cap is not None:
            self.cap.release()
        
        self.logger.info("Camera stopped")
    
    def get_latest_result(self):
        """Get latest detection result"""
        if not self.results_queue.empty():
            return self.results_queue.get()
        return None
    
    def get_latest_frame(self):
        """Get latest frame"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def save_detection_image(self, frame, result, save_dir=None):
        """Save detection image with annotation"""
        if save_dir is None:
            save_dir = self.config.LOGS_DIR / 'detections'
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp and prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{result['predicted_class']}_{result['confidence']:.3f}.jpg"
        
        # Draw annotation on frame
        annotated_frame = frame.copy()
        text = f"{result['predicted_class']} ({result['confidence']:.3f})"
        cv2.putText(
            annotated_frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Save image
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), annotated_frame)
        
        self.logger.info(f"Detection image saved: {filepath}")
        return filepath

class ConveyorBeltDetector:
    """Detector for conveyor belt waste sorting"""
    
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.sorting_decisions = []
        self.logger = logging.getLogger(__name__)
        
    def get_sorting_decision(self, prediction_result):
        """Get sorting decision based on prediction"""
        try:
            # Send request to API for sorting decision
            response = requests.post(
                f"{self.camera_manager.api_url}/sorting_decision",
                json={'prediction': prediction_result},
                timeout=5
            )
            
            if response.status_code == 200:
                decision = response.json()
                decision['timestamp'] = datetime.now().isoformat()
                self.sorting_decisions.append(decision)
                return decision
            else:
                self.logger.error(f"Sorting decision request failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting sorting decision: {str(e)}")
            return None
    
    def trigger_sorting_mechanism(self, decision):
        """Trigger physical sorting mechanism"""
        # This would interface with actual hardware
        # For now, just log the decision
        
        action = decision.get('action', 'unknown')
        category = decision.get('predicted_category', 'unknown')
        
        self.logger.info(f"Sorting mechanism triggered: {action} for {category}")
        
        # Here you would add code to control actual sorting hardware
        # For example:
        # - Control servo motors
        # - Activate pneumatic systems
        # - Send signals to PLC systems
        
        return True
    
    def process_conveyor_belt(self):
        """Main processing loop for conveyor belt"""
        while self.camera_manager.is_running:
            # Get latest detection result
            result = self.camera_manager.get_latest_result()
            
            if result:
                # Get sorting decision
                decision = self.get_sorting_decision(result)
                
                if decision:
                    # Trigger sorting mechanism
                    self.trigger_sorting_mechanism(decision)
                    
                    # Save detection image
                    frame = self.camera_manager.get_latest_frame()
                    if frame is not None:
                        self.camera_manager.save_detection_image(frame, result)
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

if __name__ == "__main__":
    # Example usage
    camera_manager = CameraManager(camera_id=0)
    conveyor_detector = ConveyorBeltDetector(camera_manager)
    
    try:
        # Start camera
        if camera_manager.start_camera():
            print("Camera started. Press 'q' to quit.")
            
            # Start conveyor belt processing
            conveyor_thread = threading.Thread(target=conveyor_detector.process_conveyor_belt)
            conveyor_thread.daemon = True
            conveyor_thread.start()
            
            # Main loop for display
            while True:
                frame = camera_manager.get_latest_frame()
                if frame is not None:
                    # Display frame
                    cv2.imshow('Waste Detection', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("Stopping camera...")
    
    finally:
        camera_manager.stop_camera()
        print("Camera stopped.")
