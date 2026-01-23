from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import json
from datetime import datetime
import logging
from monitoring import FraudMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with monitoring",
    version="2.0.0"
)

# Initialize monitor
monitor = FraudMonitor()


# Define what a transaction looks like
class Transaction(BaseModel):
    """A single transaction to check for fraud"""
    Time: float = Field(..., description="Seconds since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in dollars")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    actual_fraud: Optional[bool] = Field(None, description="True label if known (for monitoring)")
    risk_level: Optional[str] = "low"

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 12345.0,
                "Amount": 150.50,
                "V1": -1.5, "V2": 2.3, "V3": 0.5, "V4": -0.8,
                "V5": 1.2, "V6": -0.3, "V7": 0.7, "V8": 0.2,
                "V9": -1.1, "V10": 0.9, "V11": -0.4, "V12": 1.6,
                "V13": -0.2, "V14": 0.8, "V15": -0.5, "V16": 1.3,
                "V17": -0.6, "V18": 0.4, "V19": -0.9, "V20": 1.1,
                "V21": -0.7, "V22": 0.6, "V23": -1.2, "V24": 0.3,
                "V25": -0.1, "V26": 0.5, "V27": -0.4, "V28": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """What the API returns"""
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    timestamp: str
    message: str


# Load model and scaler when API starts
@app.on_event("startup")
def load_model():
    """Load the trained model and scaler"""
    global model, scaler, metadata
    
    try:
        logger.info("Loading model and scaler...")
        model = joblib.load('models/fraud_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"✓ Model loaded (trained on {metadata['training_date']})")
        logger.info("✓ Monitoring system initialized")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


# Health check endpoint
@app.get("/")
def root():
    """Check if API is running"""
    return {
        "status": "online",
        "message": "Fraud Detection API v2.0 with Monitoring",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict - Check a single transaction",
            "batch": "/predict/batch - Check multiple transactions",
            "stats": "/monitoring/stats - Get monitoring statistics",
            "analysis": "/monitoring/analysis - Get detailed analysis",
            "health": "/health - Get model info"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "monitoring": monitor.get_statistics()
    }




@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    try:
        # Extract features in correct order
        feature_order = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]
        
        features = [getattr(transaction, feature) for feature in feature_order]
        features_array = np.array(features).reshape(1, -1)
        
        # Scale and predict
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        is_fraud = bool(prediction == 1)
        
        if is_fraud:
            message = f"⚠️ FRAUD DETECTED! Risk: {risk_level.upper()} ({probability*100:.1f}% confidence)"
        else:
            message = f"✓ Transaction appears legitimate (fraud probability: {probability*100:.1f}%)"
        
        # Create response
        result = {
            'is_fraud': is_fraud,
            'fraud_probability': round(probability, 4),
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        
        # Log to monitoring system
        transaction_dict = transaction.dict()
        actual_label = transaction_dict.pop('actual_fraud', 0)

        monitor.log_prediction(
                amount=transaction.Amount,
                prob=probability,
                is_fraud=is_fraud,
                risk_level=transaction.risk_level
)


        logger.info(f"Prediction: fraud={is_fraud}, prob={probability:.4f}, amount=${transaction.Amount:.2f}")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]):
    """Predict multiple transactions at once"""
    
    try:
        results = []
        
        for txn in transactions:
            result = predict_fraud(txn)
            results.append(result.dict())
        
        total = len(results)
        fraud_count = sum(1 for r in results if r['is_fraud'])
        
        return {
            "total_transactions": total,
            "fraud_detected": fraud_count,
            "fraud_rate": round(fraud_count / total * 100, 2),
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/monitoring/stats")
def get_monitoring_stats():
    return monitor.get_stats()


@app.get("/monitoring/analysis")
def get_monitoring_analysis():
    return monitor.get_analysis()


@app.get("/monitoring/drift")
def check_drift(window_size: int = 1000):
    """Check for model drift"""
    return monitor.detect_drift(window_size)

# Run with: uvicorn api:app --reload
if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)