import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    auc
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

print("="*60)
print("FRAUD DETECTION MODEL TRAINING")
print("="*60)

# Step 1: Load data
print("\n[1/7] Loading data...")
df = pd.read_csv('creditcard_data.csv')
print(f"✓ Loaded {len(df)} transactions")

# Step 2: Prepare features and target
print("\n[2/7] Preparing features...")

# Separate features (X) and target (y)
X = df.drop('Class', axis=1)  # All columns except 'Class'
y = df['Class']  # Just the 'Class' column

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target distribution: {y.value_counts().to_dict()}")

# Step 3: Split into train and test sets
print("\n[3/7] Splitting data...")
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42,  # Makes split reproducible
    stratify=y  # Keep same fraud ratio in both sets
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Step 4: Scale the features
print("\n[4/7] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled (mean=0, std=1)")

# Step 5: Handle imbalanced data with SMOTE
print("\n[5/7] Handling imbalanced data...")
print(f"Before SMOTE - Fraud: {sum(y_train)}, Legit: {len(y_train)-sum(y_train)}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - Fraud: {sum(y_train_balanced)}, Legit: {len(y_train_balanced)-sum(y_train_balanced)}")
print("✓ Data is now balanced!")

# Step 6: Train the model
print("\n[6/7] Training XGBoost model...")
print("This might take a minute...")


model = XGBClassifier(
    max_depth=6,  # How deep trees can grow
    learning_rate=0.1,  # How fast it learns
    n_estimators=100,  # Number of trees
    random_state=42,
    eval_metric='logloss'  # What to optimize
)

# Train the model
model.fit(X_train_balanced, y_train_balanced)
print("✓ Model trained!")

# Step 7: Evaluate the model
print("\n[7/7] Evaluating model...")

# Make predictions on test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of fraud

# Calculate metrics
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"  True Negatives (correctly identified legit): {tn}")
print(f"  False Positives (legit flagged as fraud): {fp}")
print(f"  False Negatives (fraud missed): {fn}")
print(f"  True Positives (correctly caught fraud): {tp}")

# Important metrics for fraud detection
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"\nKey Metrics:")
print(f"  Precision: {precision:.4f} (When we say fraud, we're right {precision*100:.2f}% of time)")
print(f"  Recall: {recall:.4f} (We catch {recall*100:.2f}% of all fraud)")
print(f"  F1-Score: {f1:.4f} (Balanced measure)")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f} (Overall discrimination)")

# Detailed classification report
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
axes[0].set_xticklabels(['Legitimate', 'Fraud'])
axes[0].set_yticklabels(['Legitimate', 'Fraud'])

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-10:]  # Top 10 features

axes[1].barh(range(len(indices)), importances[indices], color='steelblue')
axes[1].set_yticks(range(len(indices)))
axes[1].set_yticklabels([feature_names[i] for i in indices])
axes[1].set_title('Top 10 Important Features', fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved evaluation plot to 'model_evaluation.png'")

# Save the model and scaler
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Create a folder for model artifacts
import os
os.makedirs('models', exist_ok=True)

# Save model
model_path = 'models/fraud_model.pkl'
joblib.dump(model, model_path)
print(f"✓ Model saved to {model_path}")

# Save scaler
scaler_path = 'models/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved to {scaler_path}")

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'model_type': 'XGBoost',
    'num_features': X.shape[1],
    'train_size': len(X_train),
    'test_size': len(X_test),
    'metrics': {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Metadata saved to models/metadata.json")

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Look at model_evaluation.png to see performance")
print("2. Model is ready to use for predictions")
print("3. Next: Build an API to serve predictions")