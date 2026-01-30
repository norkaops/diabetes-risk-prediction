# 🏥 Diabetes Risk Prediction & Evaluation System

A comprehensive Machine Learning application built with **Streamlit** and **XGBoost** to predict the likelihood of diabetes in patients based on their medical history and demographic details.

## 🌟 Key Features

* **Risk Prediction:** Uses an advanced XGBoost classifier to estimate the probability of diabetes with high accuracy.
* **Explainable AI (XAI):**
    * **SHAP Waterfall Plots:** Break down exactly *why* a specific patient got their score.
    * **Force Plots:** Visualize how features push the risk up or down.
    * **Global Importance:** See which factors (e.g., HbA1c, Glucose) matter most across the population.
* **Interactive Simulation:** "What-If" analysis slider to see how changing BMI affects risk in real-time.
* **Clinical Support:** Automatically generates tailored health suggestions based on medical guidelines.
* **Crash-Proof Architecture:** Optimized to run smoothly on lower-memory machines by using native C++ extraction for plots.

## 📊 Model Performance

The model was trained on a dataset of 100,000 patients and achieved the following metrics:

* **Accuracy:** 93.67%
* **ROC-AUC Score:** 0.9758
* **Recall (Sensitivity):** High recall optimization to minimize missed diagnoses.

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.98      0.95      0.96     17534
           1       0.60      0.84      0.70      1696

    accuracy                           0.94     19230
   macro avg       0.79      0.89      0.83     19230
weighted avg       0.95      0.94      0.94     19230

```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/norkaops/diabetes-risk-prediction.git](https://github.com/norkaops/diabetes-risk-prediction.git)
cd diabetes-risk-prediction

```


2. **Install requirements:**
```bash
pip install -r requirements.txt

```


3. **Run the application:**
```bash
streamlit run app.py

```



## 📂 Project Structure

```text
diabetes-risk-prediction/
│
├── app.py                  # The main Streamlit dashboard application
├── requirements.txt        # List of Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git configuration
│
├── models/                 # Pre-trained models and assets
│   ├── xgb_model.pkl       # Trained XGBoost Classifier
│   ├── preprocessor.pkl    # Data preprocessing pipeline
│   └── best_threshold.txt  # Optimized decision threshold
│
└── data/                   # Dataset for evaluation
    └── diabetes_prediction_dataset.csv

```

## 🩺 Medical Disclaimer

This tool is intended for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

```

```
