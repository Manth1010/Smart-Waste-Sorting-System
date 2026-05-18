# Smart Waste Sorting System

An AI-powered smart waste sorting system built using YOLOv8 and Flask that automatically detects, classifies, and sorts waste materials with real-time prediction, intelligent feedback learning, and automated model retraining.

---

## Project Overview

Waste segregation is a major challenge in modern waste management systems. Manual sorting is inefficient, slow, and prone to human error.

This project solves that problem using computer vision and deep learning by automatically identifying waste items, classifying them into appropriate categories, and continuously improving model performance through a feedback-driven retraining pipeline.

---

## Key Features

- Real-time waste object detection using YOLOv8
- Image upload-based prediction
- Base64 image prediction support
- Batch prediction for multiple images
- Real-time video frame prediction
- Automatic waste category classification
- Confidence-based sorting decision engine
- Manual review fallback for uncertain predictions
- User feedback correction mechanism
- Automated model retraining pipeline
- Dynamic model reloading
- REST API architecture
- Health monitoring endpoints
- Feedback analytics support

---

## Tech Stack

### Backend
- Python
- Flask
- Flask-CORS

### AI / Computer Vision
- YOLOv8
- OpenCV
- NumPy
- PIL (Pillow)

### Machine Learning Workflow
- Feedback Learning
- Automated Retraining
- Real-Time Inference

### Utilities
- Logging
- JSON Processing
- Base64 Image Handling
- Pathlib

---

## 🏗 System Architecture

```text
User Interface / Camera Input
           ↓
Flask REST API Layer
           ↓
Image Preprocessing
           ↓
YOLOv8 Detection Model
           ↓
Waste Classification
           ↓
Category Mapping
(Organic / Inorganic)
           ↓
Sorting Decision Engine
           ↓
Auto Sorting / Manual Review
           ↓
Feedback Collection
           ↓
Feedback Storage
           ↓
Retraining Pipeline
           ↓
Updated Model Deployment
```

---

## 📂 Project Structure

```bash
Smart-Waste-Sorting-System/
│
├── api/
│   └── app.py
│
├── src/
│   ├── config.py
│   ├── yolov8_wrapper.py
│   ├── feedback_manager.py
│
├── templates/
│   └── index.html
│
├── models/
├── logs/
├── retrain_with_feedback.py
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---------|--------|---------|
| `/` | GET | Main UI |
| `/health` | GET | Application health check |
| `/predict` | POST | Predict waste class from uploaded image |
| `/predict_base64` | POST | Predict using base64 image |
| `/batch_predict` | POST | Predict multiple images |
| `/predict_realtime` | POST | Real-time frame prediction |
| `/sorting_decision` | POST | Sorting action logic |
| `/feedback` | POST | Submit corrected feedback |
| `/feedback_stats` | GET | Feedback analytics |
| `/retrain` | POST | Trigger retraining |
| `/reload_model` | POST | Reload updated model |
| `/model_info` | GET | Model details |

---

## Installation

### Clone Repository
```bash
git clone https://github.com/Manth1010/Smart-Waste-Sorting-System.git
cd Smart-Waste-Sorting-System
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
python api/app.py
```

---

## Example Use Cases

- Smart recycling systems
- Industrial waste segregation
- AI-powered smart bins
- Recycling plant automation
- Computer vision object classification
- Intelligent waste monitoring

---

## Future Enhancements

- IoT hardware integration
- Conveyor belt automation
- Robotic waste sorting
- Cloud deployment
- Analytics dashboard
- Multi-category waste classification
- Performance optimization

---

## Skills Demonstrated

This project showcases:

- Computer Vision
- Deep Learning
- YOLOv8 Object Detection
- Flask API Development
- REST Architecture
- Real-Time AI Inference
- Model Retraining Pipelines
- Feedback Learning Systems
- Production-style AI Deployment

---

