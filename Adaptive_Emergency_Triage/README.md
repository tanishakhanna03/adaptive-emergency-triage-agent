# 🏥 Adaptive Emergency Triage Agent

An AI-based emergency triage prototype that predicts patient acuity from vital signs and performs adaptive reassessment as new vital-sign measurements become available.

The project uses de-identified MIMIC-IV-ED demo data and machine-learning models to demonstrate how an AI-assisted triage system could dynamically monitor changes in patient priority.

> ⚠️ This is an educational and research prototype only. It is not clinically validated and must not be used for real medical decision-making.

---

## 🎯 Project Objective

Emergency departments need to prioritize patients according to the urgency of their condition.

Traditional triage is generally performed during initial assessment, but a patient's condition can change while they remain in the emergency department.

This project explores an adaptive approach:

Patient Vital Signs  
↓  
Initial Acuity Prediction  
↓  
New Vital Measurements  
↓  
Automatic Reassessment  
↓  
Priority Change Detection  
↓  
Updated Triage Timeline

---

## 📊 Dataset

This prototype uses the de-identified **MIMIC-IV-ED demo dataset**.

Files used:

- `triage.csv.gz`
- `vitalsign.csv.gz`

The current demo dataset contains a limited number of patients and is used only for prototype development.

### Features used in the baseline model

- Temperature
- Heart Rate
- Respiratory Rate
- Oxygen Saturation (SpO₂)
- Systolic Blood Pressure
- Diastolic Blood Pressure

### Enhanced model

A second model also incorporates:

- Chief Complaint

Chief-complaint text is converted into numerical features using **TF-IDF**.

---

## 🤖 Machine Learning Pipeline

### Baseline Model — V1

The first model uses six vital-sign features.

Pipeline:

1. Load triage data
2. Remove records without an acuity label
3. Select vital-sign features
4. Split data into training and testing sets
5. Handle missing values using median imputation
6. Train a Random Forest classifier
7. Predict acuity levels
8. Save the trained model using Joblib

Saved model:

`triage_model.pkl`

---

## 🚀 Enhanced Model — V2

The enhanced model combines:

- Numerical vital-sign features
- Chief-complaint text

Preprocessing includes:

- Median imputation for missing numerical values
- Standard scaling
- TF-IDF vectorization for chief-complaint text

Two algorithms were compared:

- Random Forest
- Logistic Regression

The best model according to weighted F1 score was saved as:

`triage_model_v2.pkl`

---

## 📈 Model Evaluation

### Baseline V1

- Accuracy: **52.38%**
- Precision: **51.76%**
- Recall: **52.38%**
- Weighted F1 Score: **51.60%**

### Enhanced Random Forest V2

- Accuracy: **57.14%**
- Precision: **57.31%**
- Recall: **57.14%**
- Weighted F1 Score: **57.07%**

Adding chief-complaint information improved the overall performance compared with the vitals-only baseline.

### Logistic Regression Comparison

- Accuracy: **54.76%**
- Precision: **57.54%**
- Recall: **54.76%**
- Weighted F1 Score: **56.11%**

Although Random Forest achieved better overall performance, Logistic Regression showed better recall for the small Level-1 test subset in this experiment.

---

## 🔄 Adaptive Reassessment

The adaptive component uses repeated vital-sign measurements from `vitalsign.csv.gz`.

For a selected emergency-department stay:

1. Vital measurements are sorted chronologically.
2. Each new measurement is processed by the model.
3. A new acuity level is predicted.
4. The prediction is compared with the previous acuity.
5. The system identifies whether model-predicted priority has increased, decreased, or remained unchanged.

Example:

Initial Assessment  
`Acuity 2`

↓ New vital signs

`Acuity 4 → Priority Decreased`

↓ New vital signs

`Acuity 2 → Priority Increased`

This demonstrates dynamic model reassessment rather than relying only on a single initial prediction.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard with two major components.

### 🩺 Patient Assessment

Users can enter:

- Temperature
- Heart Rate
- Respiratory Rate
- Oxygen Saturation
- Systolic Blood Pressure
- Diastolic Blood Pressure

The system displays:

- Predicted acuity
- Model confidence
- Predicted urgency level

The patient can then be reassessed after changing the vital signs.

### 📈 Adaptive Patient Timeline

A de-identified ED stay can be selected to visualize repeated measurements.

The dashboard displays:

- Timestamp
- Vital signs
- Predicted acuity
- Model confidence
- Priority changes
- Acuity trend
- Vital-sign trends
- Reassessment summary

---

## 📂 Project Structure

```text
Adaptive_Emergency_Triage/
│
├── data/
│   ├── triage.csv.gz
│   └── vitalsign.csv.gz
│
├── app.py
├── train_model.py
├── triage_agent.py
├── inspect_vitals.py
├── evaluate_model.py
├── improve_model.py
│
├── triage_model.pkl
├── triage_model_v2.pkl
│
└── README.md