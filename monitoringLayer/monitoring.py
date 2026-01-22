import json
import pandas as pd
import numpy as np
from datetime import datetime
import os


class FraudMonitor:
    def __init__(self):
        self.predictions = []
        self.fraud_scores = []
        self.timestamps = []

        os.makedirs("monitoring", exist_ok=True)
        self.log_file = "monitoring/predictions.jsonl"

        self.stats = {
            "total_predictions": 0,
            "fraud_detected": 0,
            "legitimate_passed": 0,
            "high_risk_count": 0,
            "total_amount_flagged": 0.0,
            "fraud_rate": 0.0,
            "avg_confidence": 0.0,
            "last_updated": None,
        }

    # 🔥 THIS MUST BE OUTSIDE __init__
    def log_prediction(self, amount, prob, is_fraud, risk_level):
        self.predictions.append(int(is_fraud))
        self.fraud_scores.append(float(prob))
        self.timestamps.append(datetime.utcnow())

        self.stats["total_predictions"] += 1

        if is_fraud:
            self.stats["fraud_detected"] += 1
            self.stats["total_amount_flagged"] += float(amount)
        else:
            self.stats["legitimate_passed"] += 1

        if risk_level == "high":
            self.stats["high_risk_count"] += 1

        self.stats["fraud_rate"] = (
            self.stats["fraud_detected"] / self.stats["total_predictions"]
        )

        self.stats["avg_confidence"] = float(np.mean(self.fraud_scores))
        self.stats["last_updated"] = self.timestamps[-1].isoformat()

        log_entry = {
            "timestamp": self.timestamps[-1].isoformat(),
            "transaction_amount": float(amount),
            "fraud_probability": float(prob),
            "predicted_fraud": bool(is_fraud),
            "risk_level": risk_level,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # API: /monitoring/stats
    def get_stats(self):
        return self.stats

    # API: /monitoring/analysis
    def get_analysis(self):
        if not os.path.exists(self.log_file):
            return {"status": "no data"}

        rows = []
        with open(self.log_file) as f:
            for line in f:
                rows.append(json.loads(line))

        if not rows:
            return {"status": "no data"}

        df = pd.DataFrame(rows)

        return {
            "total_predictions": len(df),
            "fraud_rate": round(df["predicted_fraud"].mean() * 100, 2),
            "avg_probability": round(df["fraud_probability"].mean(), 4),
            "risk_levels": df["risk_level"].value_counts().to_dict(),
            "total_amount_flagged": round(
                df[df["predicted_fraud"]]["transaction_amount"].sum(), 2
            ),
            "last_10": df.tail(10).to_dict(orient="records"),
        }
