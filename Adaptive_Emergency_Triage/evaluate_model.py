import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. LOAD DATASET
# =========================================================

data = pd.read_csv("data/triage.csv.gz")

features = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp"
]

target = "acuity"

# Remove rows where acuity label is missing
data = data.dropna(subset=[target])

X = data[features]
y = data[target].astype(int)

print("\n====================================")
print("     TRIAGE MODEL EVALUATION")
print("====================================")

print("\nTotal labeled records:", len(data))

print("\nClass Distribution:")
print(y.value_counts().sort_index())


# =========================================================
# 2. CREATE SAME TEST SPLIT
# =========================================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

model = joblib.load("triage_model.pkl")

print("\nTrained model loaded successfully.")


# =========================================================
# 4. MAKE PREDICTIONS
# =========================================================

predictions = model.predict(X_test)


# =========================================================
# 5. CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n------------------------------------")
print("OVERALL PERFORMANCE")
print("------------------------------------")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# =========================================================
# 6. DETAILED CLASSIFICATION REPORT
# =========================================================

print("\n------------------------------------")
print("CLASSIFICATION REPORT")
print("------------------------------------")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# =========================================================
# 7. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n------------------------------------")
print("CONFUSION MATRIX")
print("------------------------------------")

print(cm)


# =========================================================
# 8. SAMPLE ACTUAL VS PREDICTED
# =========================================================

results = pd.DataFrame({
    "Actual Acuity": y_test.values,
    "Predicted Acuity": predictions
})

print("\n------------------------------------")
print("SAMPLE PREDICTIONS")
print("------------------------------------")

print(results.head(15))


print("\n====================================")
print("Evaluation completed successfully!")
print("====================================")