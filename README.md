# **Fraud Detection – Machine Learning Pipeline**

An end-to-end machine learning pipeline for **fraud detection**, designed to
simulate a production-oriented analytics workflow in a financial context.

Rather than focusing exclusively on model performance, this project emphasizes
**decision-making under risk**, **cost trade-offs**, and **probability reliability** —
core challenges in real-world fraud, risk, and compliance systems.

---

## **Objective**

Detect fraudulent transactions using supervised machine learning models,
handling extreme class imbalance and prioritizing evaluation criteria aligned
with **risk management and operational impact**, rather than accuracy alone.

---

## **Dataset**

The project uses a public credit card transaction dataset (UCI / Kaggle),
automatically downloaded by the pipeline to ensure full reproducibility.

The dataset is **already labeled for supervised learning**:

* **Class = 0** → legitimate transaction
* **Class = 1** → fraudulent transaction

Fraud events represent **less than 0.2% of all transactions**, making this a
highly imbalanced classification problem.

The feature set includes:

* **Time**: elapsed time since the first transaction
* **Amount**: transaction value
* **V1 to V28**: anonymized variables obtained via PCA transformation, designed
  to preserve confidentiality while retaining predictive structure

Because labels are already provided, **no heuristic or rule-based fraud detection
is required**. The project focuses on **how models behave once probabilities are
turned into decisions**, not on label discovery.

---

## **Data Preparation Scope**

This project intentionally applies **minimal upfront data treatment**.

Beyond standard scaling and **stratified train/test splitting**, no extensive
feature engineering or exploratory cleaning is performed. This choice is
**deliberate**.

In real-world fraud systems, early stages often include:

* data quality validation
* missing value treatment
* outlier handling
* domain-driven feature engineering
* temporal and behavioral aggregation

Here, those steps are intentionally left out to **isolate the impact of
evaluation strategy, threshold selection, cost modeling, and calibration**.
By starting from a clean, anonymized dataset, the pipeline highlights how
**decision logic — not feature engineering — can dominate real-world outcomes**.

---

## **Approach and Techniques**

The pipeline was built incrementally, following a **decision-oriented approach**
rather than a model-centric one.

It includes:

* **Logistic Regression** as a transparent and interpretable baseline
* **Random Forest** as a non-linear ensemble model
* Explicit comparison against a **naive baseline**
* **Precision–Recall** as the primary evaluation framework
* **Threshold tuning** to enforce a minimum fraud recall constraint
* **Cost simulation** based on false positives and false negatives
* **Probability calibration** to align scores with observed risk
* **Feature importance analysis** for interpretability

This structure mirrors how fraud models are typically evaluated and deployed in
production environments.

---

## **Tech Stack**

* **Python**
* **Pandas / NumPy**
* **Scikit-learn**
* **Matplotlib / Seaborn**
* **Structured logging (TXT + JSON)**

---

## **Pipeline Overview**

```text
┌────────────────────┐
│   Data Download    │
│ (Public Dataset)   │
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Data Preparation   │
│  • Scaling         │
│  • Stratified Split│
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Model Training     │
│  • Logistic Reg.   │
│  • Random Forest   │
│  • Naive Baseline  │
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Evaluation         │
│  • PR Curve        │
│  • ROC             │
│  • Confusion Matrix│
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Threshold Tuning   │
│  • Recall target   │
│  • Cost-aware      │
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Calibration        │
│  • Isotonic        │
│  • Risk alignment  │
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Decision Outputs   │
│  • Cost comparison │
│  • Logs & reports  │
└────────────────────┘
```

---

## **Final Model Decision**

After evaluating all models against a naive baseline, the **Random Forest
classifier** was selected as the final solution.

This decision was driven by:

* Superior behavior on the **Precision–Recall curve**
* A more favorable redistribution of **false negatives and false positives**
* The **lowest total expected cost** under simulated risk assumptions

Instead of relying on a default probability threshold, the model’s threshold was
**explicitly tuned to enforce a minimum fraud recall**, aligning model behavior
with operational risk tolerance.

> In highly imbalanced problems such as fraud detection,
> **threshold choice and cost modeling often matter more than algorithm choice**.

---

## **Calibration and Risk Alignment**

After threshold tuning, **probability calibration** was applied to assess whether
model scores reflected real-world fraud risk.

Calibration does not aim to improve ranking metrics such as ROC or Average
Precision. Instead, it adjusts the **probability scale itself**, making scores
more reliable for operational decisions.

Key observations:

* **Logistic Regression** showed limited gains from calibration
* **Random Forest** exhibited overconfidence prior to calibration, which was
  mitigated using **isotonic regression**
* The **naive baseline** provided no meaningful signal for calibration

As a result, thresholds became more stable and cost simulations more reliable.

## 🚀 How to Run
```bash
pip install -r requirements.txt
python fraud_detection_pipeline.py
```
---

## **Related Article**

A detailed walkthrough of the project — from the initial idea to the final
risk-based decision — will be published on Medium:

**https://medium.com/@julio.pimp/por-que-accuracy-falha-em-projetos-reais-de-detec%C3%A7%C3%A3o-de-fraudes-ae405622aa2c**

