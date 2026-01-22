import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os
from datetime import datetime

# Create mlruns folder if it doesn't exist
os.makedirs('mlruns', exist_ok=True)

print("="*60)
print("TRAINING WITH MLFLOW TRACKING")
print("="*60)

# Set MLflow tracking URI (where to save experiments)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud-detection")

# Start  MLflow run (this tracks one training session)
with mlflow.start_run(run_name=f"xgboost-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
    
    print("\n[1/8] Loading data...")
    df = pd.read_csv('creditcard_data.csv')
    print(f"✓ Loaded {len(df)} transactions")
    
    # Log dataset info
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("fraud_count", int(df['Class'].sum()))
    mlflow.log_param("fraud_percentage", float(df['Class'].mean() * 100))
    
    print("\n[2/8] Preparing features...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    mlflow.log_param("num_features", X.shape[1])
    
    print("\n[3/8] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    
    print("\n[4/8] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n[5/8] Handling imbalanced data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    mlflow.log_param("balancing_method", "SMOTE")
    
    print("\n[6/8] Training XGBoost model...")
    
    # Model hyperparameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Log all hyperparameters
    for param, value in params.items():
        mlflow.log_param(param, value)
    
    model = XGBClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    print("✓ Model trained!")
    
    print("\n[7/8] Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate all metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log all metrics to MLflow
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("true_positives", int(tp))
    mlflow.log_metric("false_positives", int(fp))
    mlflow.log_metric("true_negatives", int(tn))
    mlflow.log_metric("false_negatives", int(fn))
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    print("\n[8/8] Saving model to MLflow...")
    # Log the model to MLflow
    mlflow.xgboost.log_model(
        model, 
        "model",
        registered_model_name="fraud-detection-model"
    )
    
    # save scaler as artifact
    scaler_path = "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path)
    os.remove(scaler_path)  # Clean up
    
    # Feature importance
    feature_names = X.columns.tolist()

    # Log feature importance
    importance_dict = {
      feature: float(importance)
      for feature, importance in zip(feature_names, model.feature_importances_)
             }
    mlflow.log_dict(importance_dict, "feature_importance.json")
    
    print("✓ Model and artifacts saved to MLflow!")
    
    # Get the run ID for reference
    run_id = mlflow.active_run().info.run_id
    print(f"\n✓ MLflow Run ID: {run_id}")
    print(f"✓ Model URI: runs:/{run_id}/model")

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print("\nTo view results:")
print("1. Run: mlflow ui")
print("2. Open: http://localhost:5000")
print("3. Browse your experiments!")
print("\nNext: Add monitoring")