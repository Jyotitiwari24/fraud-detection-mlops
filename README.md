**Real-Time Fraud Detection System**

This project is a real-time fraud detection application built to understand how a machine learning model is trained, deployed as an API, and monitored after deployment.

The goal of this project is learning end-to-end ML + basic MLOps concepts, not just model accuracy.

**What this project does**

Trains a fraud detection model using synthetic transaction data
Serves predictions through a FastAPI REST API
Logs and tracks predictions for monitoring
Shows basic statistics through monitoring endpoints
Supports local and Docker-based execution

**Key Features**

Fraud classification using XGBoost
REST API for real-time predictions
Prediction monitoring (counts, fraud rate, confidence)
MLflow experiment tracking (optional)
Simple HTML dashboard for visualization
Docker support for deployment practice

**Why I built this**

Most ML projects stop after training a model.
This project focuses on what happens after deployment:
How predictions are served
How results are tracked
How model behavior is monitored over time

 **Tech Stack**

Machine Learning: scikit-learn, XGBoost
API: FastAPI, Uvicorn
Tracking: MLflow
Data: Pandas, NumPy
Deployment: Docker, Docker Compose
Frontend: HTML, CSS, JavaScript

**Prerequisites**

Python 3.8+
Git
Docker (optional)

 **Getting Started**
1️ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

2️ Generate sample data
python generate_synthetic_data.py

Synthetic data is used only for learning purposes.

3️ Train the model
python train_model.py

Optional (with MLflow):

python train_with_mlflow.py
mlflow ui

Open: http://localhost:5000

4️ Start the API server
uvicorn api:app --reload

API: http://localhost:8000

Docs (Swagger): http://localhost:8000/docs

5️ View dashboard

Open dashboard.html in a browser to see:
Prediction statistics
Fraud counts
Monitoring summaries

**API Endpoints**
Prediction
POST /predict

Batch Prediction
POST /predict/batch

Monitoring
GET /monitoring/stats
GET /monitoring/analysis
GET /monitoring/drift

Health Check
GET /health

**Project Structure**
fraud-detection/
├── api.py                     # FastAPI application
├── train_model.py             # Model training
├── train_with_mlflow.py       # Training with MLflow
├── monitoring.py              # Monitoring logic
├── generate_synthetic_data.py # Data generation
├── test_api.py                # API tests
├── dashboard.html             # Simple dashboard
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── models/                    # Saved models
├── monitoring/                # Prediction logs
└── mlruns/                    # MLflow runs

**Testing**
python test_api.py

 **Model Results (approx.)**
Precision: ~94%

Recall: ~92%

F1-score: ~93%

ROC-AUC: ~98%

Metrics are based on synthetic data and used only for learning.

 **Docker (Optional)**
docker-compose up -d
docker-compose logs -f
docker-compose down

 **Possible Improvements**

Authentication for API
Better input validation
Prometheus & Grafana monitoring
CI/CD pipeline
Real dataset integration

**Production Issues Resolved**

Fixed Docker container unhealthy state by implementing internal Python healthcheck

Resolved monitoring system crash due to incorrect function signature

Handled NoneType label issue in prediction logging

Rebased and refactored project structure into modular API, monitoring, and deployment layers

This project demonstrates:
End-to-end ML workflow
Model deployment using FastAPI
Basic monitoring concepts
Experiment tracking with MLflow
Docker-based deployment
Practical MLOps fundamentals

**Note**
This is a learning and portfolio project created to practice real-world ML deployment concepts.
It is not intended for production use.
