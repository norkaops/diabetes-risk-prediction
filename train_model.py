import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, roc_auc_score, recall_score, precision_score,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------- CONFIGURATION ----------------
DATA_PATH = os.path.join("data", "diabetes_prediction_dataset.csv")
MODELS_DIR = "models"
THRESH_FILE = os.path.join(MODELS_DIR, "best_threshold.txt")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
def bmi_to_cat(x):
    if x < 18.5: return "underweight"
    elif x < 25: return "normal"
    elif x < 30: return "overweight"
    else: return "obese"

def hba1c_to_cat(x):
    if x < 5.7: return "normal"
    elif x < 6.5: return "prediabetes"
    else: return "diabetes"

def run_training_pipeline():
    print("--- STARTING TRAINING ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH).drop_duplicates().dropna()

    # Feature Engineering
    df["BMI_cat"] = df["bmi"].apply(bmi_to_cat)
    df["HbA1c_cat"] = df["HbA1c_level"].apply(hba1c_to_cat)

    target_col = "diabetes"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    categorical_cols = ["gender", "hypertension", "heart_disease", "smoking_history", "BMI_cat", "HbA1c_cat"]

    # Preprocessing
    print("Preprocessing...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        verbose_feature_names_out=False
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Balancing
    print("Balancing (SMOTE)...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

    # Training
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
    )
    xgb_model.fit(X_train_res, y_train_res)

    # Optimization
    print("Optimizing Threshold...")
    y_proba = xgb_model.predict_proba(X_test_proc)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    best_thr = 0.5
    best_recall = 0
    final_prec = 0
    
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        rec = recall_score(y_test, y_pred_thr, zero_division=0)
        prec = precision_score(y_test, y_pred_thr, zero_division=0)
        if prec >= 0.60:
            if rec > best_recall:
                best_recall = rec
                best_thr = thr
                final_prec = prec

    print(f"Selected Threshold: {best_thr:.4f}")

    # Metrics & Images
    y_pred_final = (y_proba >= best_thr).astype(int)
    acc = accuracy_score(y_test, y_pred_final)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred_final)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(classification_report(y_test, y_pred_final))

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Diabetes"])
    disp.plot(cmap="Blues", values_format='d')
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.savefig(os.path.join(MODELS_DIR, "roc_curve_xgb.png"))
    plt.close()

    # Save Standard Models
    print("Saving Models...")
    with open(THRESH_FILE, "w") as f:
        f.write(str(best_thr))
        
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    
    print("Done! Models saved.")

if __name__ == "__main__":
    run_training_pipeline()