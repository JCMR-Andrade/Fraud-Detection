import os
import json
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

# =========================
# CONFIG (TUNING + COST)
# =========================
TARGET_RECALL_FRAUD = 0.85  # meta de recall mínimo para fraudes (classe 1)
COST_FALSE_POSITIVE = 10    # custo de investigar falso positivo
COST_FALSE_NEGATIVE = 1000  # custo de perder uma fraude (FN)

# =========================
# SETUP
# =========================
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    filename="logs/execution.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

execution_log = {
    "start_time": datetime.utcnow().isoformat(),
    "steps": [],
    "config": {
        "target_recall_fraud": TARGET_RECALL_FRAUD,
        "cost_false_positive": COST_FALSE_POSITIVE,
        "cost_false_negative": COST_FALSE_NEGATIVE
    }
}

# =========================
# DOWNLOAD DATASET
# =========================
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
dataset_path = "data/creditcard.csv"

if not os.path.exists(dataset_path):
    logging.info("Downloading dataset...")
    response = requests.get(dataset_url, timeout=60)
    response.raise_for_status()
    with open(dataset_path, "wb") as f:
        f.write(response.content)
    execution_log["steps"].append("dataset_downloaded")
else:
    execution_log["steps"].append("dataset_already_exists")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(dataset_path)

execution_log["rows"] = df.shape[0]
execution_log["columns"] = df.shape[1]

X = df.drop(columns=["Class"])
y = df["Class"]

# =========================
# PREPROCESSING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

execution_log["train_size"] = len(X_train)
execution_log["test_size"] = len(X_test)

# =========================
# MODELS (INCLUDING NAIVE)
# =========================
models = {
    "NaiveBaseline": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
        # n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
        # n_jobs=-1
    )
}

# =========================
# HELPERS (INLINE LOGIC ONLY)
# =========================
results = {}
pr_curves = {}       # para plot comparativo PR
cost_rows = []       # para salvar custos em csv/json

# =========================
# TRAIN & EVALUATE
# =========================
for name, model in models.items():
    logging.info(f"Training model: {name}")
    model.fit(X_train, y_train)

    # ---------- DEFAULT THRESHOLD (predict)
    y_pred_default = model.predict(X_test)

    # =========================
    # PROBABILITIES (RAW)
    # =========================
    if hasattr(model, "predict_proba"):
        y_proba_raw = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba_raw)
        ap = average_precision_score(y_test, y_proba_raw)
    else:
        y_proba_raw = None
        roc_auc = None
        ap = None

    results[name] = {"roc_auc": roc_auc, "average_precision": ap}
    execution_log[f"{name}_roc_auc"] = roc_auc
    execution_log[f"{name}_average_precision"] = ap

    # =========================
    # CALIBRATION (ISOTONIC)
    # =========================
    y_proba_cal = None
    if y_proba_raw is not None:
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method="isotonic",
            cv=3
        )
        calibrated_model.fit(X_train, y_train)
        y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

        execution_log[f"{name}_calibration"] = "isotonic"

        # ----- Calibration curve (raw vs calibrated)
        prob_true_raw, prob_pred_raw = calibration_curve(
            y_test, y_proba_raw, n_bins=10
        )
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_test, y_proba_cal, n_bins=10
        )

        plt.figure(figsize=(7, 7))
        plt.plot(prob_pred_raw, prob_true_raw, marker="o", label="Uncalibrated")
        plt.plot(prob_pred_cal, prob_true_cal, marker="o", label="Calibrated")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Observed Frequency")
        plt.title(f"Calibration Curve - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/calibration_curve_{name}.png")
        plt.close()

    # =========================
    # CLASSIFICATION REPORT (DEFAULT)
    # =========================
    report_default = classification_report(y_test, y_pred_default, output_dict=True)
    execution_log[f"{name}_classification_report_default"] = report_default

    with open(f"outputs/classification_report_default_{name}.json", "w") as f:
        json.dump(report_default, f, indent=4)

    cm_default = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm_default.ravel()

    plt.figure()
    sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Default) - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"outputs/confusion_matrix_default_{name}.png")
    plt.close()

    cost_default = (fp * COST_FALSE_POSITIVE) + (fn * COST_FALSE_NEGATIVE)

    # =========================
    # PR CURVE + THRESHOLD TUNING
    # >>> USANDO PROBABILIDADE CALIBRADA
    # =========================
    tuned_threshold = None
    tuned_cost = None
    tuned_recall_fraud = None
    tuned_precision_fraud = None

    if y_proba_cal is not None:
        precision, recall, thresholds = precision_recall_curve(
            y_test, y_proba_cal
        )

        pr_curves[name] = {
            "precision": precision,
            "recall": recall,
            "ap": ap
        }

        candidate_idxs = [
            i for i in range(len(thresholds))
            if recall[i + 1] >= TARGET_RECALL_FRAUD
        ]

        if candidate_idxs:
            best_i = max(
                candidate_idxs,
                key=lambda i: precision[i + 1]
            )
            tuned_threshold = float(thresholds[best_i])
        else:
            best_i = int(np.argmax(recall[1:])) - 1
            best_i = max(0, min(best_i, len(thresholds) - 1))
            tuned_threshold = float(thresholds[best_i])

        y_pred_tuned = (y_proba_cal >= tuned_threshold).astype(int)

        report_tuned = classification_report(
            y_test, y_pred_tuned, output_dict=True
        )

        tuned_precision_fraud = float(
            report_tuned.get("1", {}).get("precision", 0.0)
        )
        tuned_recall_fraud = float(
            report_tuned.get("1", {}).get("recall", 0.0)
        )

        execution_log[f"{name}_tuned_threshold"] = tuned_threshold
        execution_log[f"{name}_classification_report_tuned"] = report_tuned

        with open(f"outputs/classification_report_tuned_{name}.json", "w") as f:
            json.dump(report_tuned, f, indent=4)

        cm_tuned = confusion_matrix(y_test, y_pred_tuned)
        tn2, fp2, fn2, tp2 = cm_tuned.ravel()

        plt.figure()
        sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix (Tuned) - {name}\nthreshold={tuned_threshold:.4f}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"outputs/confusion_matrix_tuned_{name}.png")
        plt.close()

        tuned_cost = (fp2 * COST_FALSE_POSITIVE) + (fn2 * COST_FALSE_NEGATIVE)

        # PR curve individual
        plt.figure()
        plt.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Calibrated)")
        plt.legend()
        plt.savefig(f"outputs/precision_recall_curve_{name}.png")
        plt.close()

    # =========================
    # COST TABLE
    # =========================
    cost_rows.append({
        "model": name,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "default_fp": int(fp),
        "default_fn": int(fn),
        "default_cost": float(cost_default),
        "tuned_threshold": tuned_threshold,
        "tuned_precision_fraud": tuned_precision_fraud,
        "tuned_recall_fraud": tuned_recall_fraud,
        "tuned_cost": float(tuned_cost) if tuned_cost is not None else None
    })

# =========================
# FEATURE IMPORTANCE (RF)
# =========================
rf_model = models["RandomForest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), X.columns[indices], rotation=45)
plt.title("Top 10 Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# =========================
# PR CURVE COMPARATIVA (TODOS OS MODELOS)
# =========================
if len(pr_curves) > 0:
    plt.figure()
    for name, d in pr_curves.items():
        plt.plot(d["recall"], d["precision"], label=f"{name} (AP={d['ap']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Comparison)")
    plt.legend()
    plt.savefig("outputs/precision_recall_curve_comparison.png")
    plt.close()

# =========================
# COST COMPARISON PLOT
# =========================
cost_df = pd.DataFrame(cost_rows)

# salvar outputs
cost_df.to_csv("outputs/model_costs.csv", index=False)
with open("outputs/model_costs.json", "w") as f:
    json.dump(cost_rows, f, indent=4)

# gráfico: custo default vs tuned (quando existe)
plt.figure(figsize=(10, 6))
x = np.arange(len(cost_df))
width = 0.35

default_costs = cost_df["default_cost"].values
tuned_costs = cost_df["tuned_cost"].fillna(np.nan).values

plt.bar(x - width / 2, default_costs, width, label="Default Cost")
plt.bar(x + width / 2, tuned_costs, width, label="Tuned Cost")

plt.xticks(x, cost_df["model"].values, rotation=20)
plt.ylabel("Cost")
plt.title("Cost Simulation: Default vs Tuned Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/cost_comparison.png")
plt.close()

# =========================
# SAVE JSON LOG
# =========================
execution_log["end_time"] = datetime.utcnow().isoformat()
execution_log["cost_summary"] = cost_rows

with open("logs/execution.json", "w") as f:
    json.dump(execution_log, f, indent=4)

logging.info("Pipeline execution completed successfully.")
