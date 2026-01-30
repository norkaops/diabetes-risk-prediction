# 🏥 Diabetes Risk Prediction & Evaluation System

A comprehensive Machine Learning application built with **Streamlit** and **XGBoost** to predict the likelihood of diabetes in patients based on their medical history and demographic details.

## 🌟 Key Features

* **Risk Prediction:** Uses an advanced XGBoost classifier to estimate the probability of diabetes.
* **Explainable AI (XAI):**
    * **SHAP Waterfall Plots:** Break down exactly *why* a specific patient got their score.
    * **Force Plots:** Visualize how features push the risk up or down.
    * **Global Importance:** See which factors (e.g., HbA1c, Glucose) matter most across the population.
* **Interactive Simulation:** "What-If" analysis slider to see how changing BMI affects risk in real-time.
* **Clinical Support:** Automatically generates tailored health suggestions based on medical guidelines.


## 📊 Model Performance

The model was trained on a dataset of 100,000 patients and achieved the following metrics:
* **Accuracy:** 93.67%
* **ROC-AUC Score:** 0.9758
* **Recall (Sensitivity):** High recall optimization to minimize missed diagnoses.
*  precision    recall  f1-score   support

           0       0.98      0.95      0.96     17534
           1       0.60      0.84      0.70      1696

    accuracy                           0.94     19230
   macro avg       0.79      0.89      0.83     19230
weighted avg       0.95      0.94      0.94     19230

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/norkaops/diabetes-prediction-app.git](https://github.com/your-username/diabetes-prediction-app.git)
    cd diabetes-prediction-app
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

* `app.py`: The main Streamlit dashboard.
* `models/`: Contains the trained XGBoost model (`xgb_model.pkl`) and preprocessing pipeline.
* `data/`: Contains the dataset used for generating diagnostic plots.
* diabetes-prediction-app/
│
├── app.py                  # The main Streamlit application (The big code I gave you)
├── requirements.txt        # List of libraries needed
├── README.md               # The documentation (I will write this for you below)
├── .gitignore              # Tells git what NOT to upload
│
├── models/                 # Folder containing your trained models
│   ├── xgb_model.pkl
│   ├── preprocessor.pkl
│   └── best_threshold.txt
│
└── data/                   # Folder containing the dataset
    └── diabetes_prediction_dataset.csv

## 🩺 Medical Disclaimer

This tool is intended for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
